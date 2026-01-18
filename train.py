"""
train_novasr.py - Complete Training Script for NovaSR Denoising

Usage:
    python train_novasr.py [--mode denoise|upsample] [--epochs 10] [--batch-size 8]

Examples:
    # Train denoising model (default)
    python train_novasr.py
    
    # Train upsample-only model
    python train_novasr.py --mode upsample
    
    # Start from custom pretrained model
    python train_novasr.py --model my_model.bin --epochs 5
    
    # Custom settings with audio comparison
    python train_novasr.py --epochs 20 --batch-size 4 --lr 1e-5 --compare
"""

import argparse
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Tuple, Generator, Optional
from dataclasses import dataclass
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio.transforms as T
from torch.optim.lr_scheduler import StepLR

warnings.filterwarnings('ignore')

# ============================================================================
# INSTALL DEPENDENCIES
# ============================================================================

def install_deps():
    """Install required packages"""
    print("📦 Installing dependencies...")
    pkgs = ["torch", "torchaudio", "datasets", "soundfile", "librosa", 
            "auraloss", "einops", "huggingface_hub"]
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + pkgs)
        print("✓ Dependencies installed\n")
    except:
        print("⚠️  Some packages may not be installed")

# Install before importing
install_deps()

try:
    import auraloss
    from datasets import load_dataset, Audio as AudioFeature
    from NovaSR import FastSR
except ImportError as e:
    print(f"❌ Error: {e}")
    print("\nMake sure NovaSR package exists in current directory")
    print("Run: python create_novasr_package.py")
    sys.exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Training configuration"""
    # Mode
    mode: str = 'denoise'  # 'denoise' or 'upsample'
    
    # Model
    model_path: Optional[str] = None  # Custom pretrained model path
    from_scratch: bool = False
    
    # Dataset
    dataset: str = 'ai4bharat/Rasa'
    dataset_config: str = 'Urdu'
    max_samples: int = 8000
    min_duration: float = 4.0
    
    # Noise (for denoise mode)
    noise_types: tuple = ('white', 'pink', 'brown', 'environmental')
    snr_range: Tuple[float, float] = (0, 20)
    noise_prob: float = 0.7
    
    # Training
    batch_size: int = 8
    lr: float = 1e-5
    epochs: int = 10
    grad_clip: float = 1.0
    weight_decay: float = 0.01
    
    # Loss
    stft_weight: float = 10.0
    phase_weight: float = 1.0
    lsd_weight: float = 1.0
    
    # Schedule
    sched_step: int = 1000
    sched_gamma: float = 0.99
    
    # Logging
    log_interval: int = 50
    compare: bool = False  # Display audio comparison
    
    # Audio
    sr_high: int = 48000
    sr_low: int = 16000
    
    @property
    def crop_len(self):
        return int(self.min_duration * self.sr_high)
    
    @property
    def add_noise(self):
        return self.mode == 'denoise'

# ============================================================================
# NOISE GENERATOR
# ============================================================================

class NoiseGen:
    @staticmethod
    def white(shape, dev='cpu'):
        return torch.randn(shape, device=dev)
    
    @staticmethod
    def pink(shape, dev='cpu'):
        w = torch.randn(shape, device=dev)
        fft = torch.fft.rfft(w)
        freqs = torch.arange(fft.shape[-1], device=dev) + 1
        fft = fft / torch.sqrt(freqs.float())
        p = torch.fft.irfft(fft, n=shape[-1])
        return p / (p.std() + 1e-8)
    
    @staticmethod
    def brown(shape, dev='cpu'):
        w = torch.randn(shape, device=dev)
        fft = torch.fft.rfft(w)
        freqs = torch.arange(fft.shape[-1], device=dev) + 1
        fft = fft / freqs.float()
        b = torch.fft.irfft(fft, n=shape[-1])
        return b / (b.std() + 1e-8)
    
    @staticmethod
    def env(shape, dev='cpu'):
        l = NoiseGen.brown(shape, dev) * 0.7
        m = NoiseGen.pink(shape, dev) * 0.3
        h = NoiseGen.white(shape, dev) * 0.1
        e = l + m + h
        return e / (e.std() + 1e-8)
    
    @staticmethod
    def add_at_snr(clean, ntype, snr_db, dev='cpu'):
        gens = {'white': NoiseGen.white, 'pink': NoiseGen.pink,
                'brown': NoiseGen.brown, 'environmental': NoiseGen.env}
        
        gen = gens.get(ntype, NoiseGen.white)
        noise = gen(clean.shape, dev)
        
        sp = clean.pow(2).mean()
        np = noise.pow(2).mean()
        snr = 10 ** (snr_db / 10)
        scale = torch.sqrt(sp / (snr * np + 1e-8))
        
        return clean + scale * noise, noise * scale

# ============================================================================
# LOSSES
# ============================================================================

class PhaseLoss(nn.Module):
    def __init__(self, ffts=(512,1024,2048), hops=(128,256,512), wins=(512,1024,2048)):
        super().__init__()
        self.ffts, self.hops, self.wins = ffts, hops, wins
        for i, w in enumerate(wins):
            self.register_buffer(f"win_{i}", torch.hann_window(w))
    
    @staticmethod
    def pdiff(a, b):
        return torch.atan2(torch.sin(a-b), torch.cos(a-b))
    
    def calc(self, yh, y, nfft, hop, win, window):
        window = window.to(yh.device)
        sh = torch.stft(yh, nfft, hop, win, window, return_complex=True)
        sy = torch.stft(y, nfft, hop, win, window, return_complex=True)
        
        ph, py = torch.angle(sh), torch.angle(sy)
        mag = torch.abs(sy) + 1e-7
        mw = mag ** 0.3
        mw = mw / (mw.amax(dim=(-2,-1), keepdim=True) + 1e-7)
        
        dp = self.pdiff(ph, py)
        lpw = torch.mean(mw * (1 - torch.cos(dp)))
        
        dph = self.pdiff(ph[...,1:], ph[...,:-1])
        dpy = self.pdiff(py[...,1:], py[...,:-1])
        dif = self.pdiff(dph, dpy)
        lif = torch.mean(mw[...,1:] * (1 - torch.cos(dif)))
        
        dfh = self.pdiff(ph[...,1:,:], ph[...,:-1,:])
        dfy = self.pdiff(py[...,1:,:], py[...,:-1,:])
        dgd = self.pdiff(dfh, dfy)
        lgd = torch.mean(mw[...,1:,:] * (1 - torch.cos(dgd)))
        
        return lpw + lif + lgd
    
    def forward(self, yh, y):
        s = torch.sqrt(torch.mean(y**2, dim=-1, keepdim=True) + 1e-7)
        y, yh = y/s, yh/s.detach()
        
        tot = 0
        for i in range(len(self.ffts)):
            tot += self.calc(yh, y, self.ffts[i], self.hops[i], 
                           self.wins[i], getattr(self, f"win_{i}"))
        return tot / len(self.ffts)

def lsd(tgt, est, nfft=2048, hop=512, eps=1e-8):
    if tgt.ndim == 1:
        tgt, est = tgt.unsqueeze(0), est.unsqueeze(0)
    
    win = torch.hann_window(nfft, device=tgt.device)
    Ya = torch.stft(tgt, nfft, hop, window=win, return_complex=True).abs()
    Yh = torch.stft(est, nfft, hop, window=win, return_complex=True).abs()
    
    lr = torch.log10((Ya+eps) / (Yh+eps))
    d = torch.sqrt(torch.mean(lr**2, dim=1))
    return d.mean(dim=1).mean()

# ============================================================================
# DATA
# ============================================================================

class DataPipe:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.resample = T.Resample(cfg.sr_high, cfg.sr_low)
        self.noise = NoiseGen()
        self._load()
    
    def _load(self):
        print(f"\n📂 Loading {self.cfg.dataset} ({self.cfg.dataset_config})")
        
        try:
            self.ds = load_dataset(self.cfg.dataset, self.cfg.dataset_config,
                                  split='train', streaming=True)
            self.ds = self.ds.cast_column('audio', AudioFeature(decode=False))
            self.ds = self.ds.take(self.cfg.max_samples)
            
            print(f"✓ Dataset ready (~{self.cfg.max_samples} samples)")
            if self.cfg.add_noise:
                print(f"✓ Noise: {', '.join(self.cfg.noise_types)}")
                print(f"  SNR: {self.cfg.snr_range[0]}-{self.cfg.snr_range[1]} dB")
                print(f"  Prob: {self.cfg.noise_prob*100:.0f}%")
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Run: huggingface-cli login")
            raise
    
    def _decode(self, ad):
        import io, soundfile as sf
        try:
            if 'bytes' in ad and ad['bytes']:
                arr, sr = sf.read(io.BytesIO(ad['bytes']))
            elif 'path' in ad and ad['path']:
                arr, sr = sf.read(ad['path'])
            else:
                return None
            
            a = torch.from_numpy(arr).float()
            if a.ndim > 1:
                a = a.mean(dim=0)
            return a, sr
        except:
            return None
    
    def _add_noise(self, a48):
        if not self.cfg.add_noise or random.random() > self.cfg.noise_prob:
            return a48
        
        nt = random.choice(self.cfg.noise_types)
        snr = random.uniform(*self.cfg.snr_range)
        noisy, _ = self.noise.add_at_snr(a48, nt, snr)
        return noisy
    
    def _aug(self, a16):
        if random.random() > 0.5:
            bits = random.choice([8, 12, 14])
            levels = 2 ** bits
            a16 = torch.round(a16 * levels) / levels
            a16 = a16 * random.uniform(0.9, 1.1)
        return a16
    
    def gen(self):
        cnt, skip, noise = 0, 0, 0
        print("\n🔄 Processing...")
        
        for idx, item in enumerate(self.ds):
            try:
                res = self._decode(item['audio'])
                if res is None:
                    skip += 1
                    continue
                
                aud, sr = res
                
                if sr != self.cfg.sr_high:
                    aud = T.Resample(sr, self.cfg.sr_high)(aud)
                
                if aud.shape[-1] < self.cfg.crop_len:
                    skip += 1
                    continue
                
                n = aud.shape[-1] // self.cfg.crop_len
                for i in range(n):
                    s = i * self.cfg.crop_len
                    clean = aud[s:s+self.cfg.crop_len]
                    
                    noisy = self._add_noise(clean)
                    a16 = self.resample(noisy)
                    a16 = self._aug(a16)
                    
                    if not torch.allclose(noisy, clean, atol=1e-5):
                        noise += 1
                    
                    # Return clean (target), low-res input, and noisy version
                    yield clean, a16, noisy
                    cnt += 1
                
                if (idx+1) % 10 == 0:
                    print(f"  {idx+1} items, {cnt} segs ({noise} noisy)...", end='\r')
            except:
                skip += 1
        
        print(f"\n✓ {cnt} segments (skip {skip})")
        if self.cfg.add_noise:
            print(f"  Clean: {cnt-noise}, Noisy: {noise} ({noise/cnt*100:.1f}%)")
    
    def loader(self, bs):
        gen = self.gen()
        b48, b16, b48_noisy = [], [], []
        
        for s48, s16, s48_noisy in gen:
            b48.append(s48)
            b16.append(s16)
            b48_noisy.append(s48_noisy)
            
            if len(b48) == bs:
                yield {
                    'a48': torch.stack(b48),
                    'a16': torch.stack(b16),
                    'a48_noisy': torch.stack(b48_noisy)
                }
                b48, b16, b48_noisy = [], [], []

# ============================================================================
# TRAINER
# ============================================================================

class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._model()
        self._losses()
        self._optim()
        self._data()
        self._setup_display()
    
    def _setup_display(self):
        """Setup audio display for notebooks"""
        self.in_notebook = False
        self.IPython = None
        
        if self.cfg.compare:
            try:
                from IPython.display import display, Audio, HTML
                from IPython import get_ipython
                
                # Check if running in IPython/Jupyter
                ipython = get_ipython()
                if ipython is not None:
                    self.in_notebook = True
                    self.IPython = type('IPython', (), {
                        'display': display,
                        'Audio': Audio,
                        'HTML': HTML
                    })
                    print("✓ Audio comparison enabled")
                else:
                    print("⚠️  --compare requires Jupyter notebook")
                    self.cfg.compare = False
            except Exception as e:
                print(f"⚠️  --compare requires Jupyter notebook (error: {e})")
                self.cfg.compare = False
    
    def _model(self):
        print("\n🤖 Model...")
        self.m = FastSR()
        
        tot = sum(p.numel() for p in self.m.model.parameters())
        print(f"✓ {tot:,} params ({tot*2/1024:.1f} KB)")
        
        # Load custom model if specified
        if self.cfg.model_path:
            model_path = Path(self.cfg.model_path)
            if not model_path.exists():
                print(f"❌ Error: Model file not found: {self.cfg.model_path}")
                sys.exit(1)
            
            try:
                print(f"📥 Loading custom model: {self.cfg.model_path}")
                state_dict = torch.load(self.cfg.model_path, map_location='cpu')
                self.m.model.load_state_dict(state_dict, strict=True)
                print(f"✓ Loaded pretrained weights from {self.cfg.model_path}")
            except Exception as e:
                print(f"❌ Error loading model: {e}")
                sys.exit(1)
        
        elif self.cfg.from_scratch:
            print("⚠️  Random init...")
            def rst(m):
                if hasattr(m, 'reset_parameters'):
                    m.reset_parameters()
                elif isinstance(m, (nn.Conv1d, nn.Linear)):
                    nn.init.kaiming_normal_(m.weight)
            self.m.model.apply(rst)
            print("✓ Random weights")
        else:
            print("✓ Using default pretrained weights")
        
        self.m.model.train().cuda().float()
    
    def _losses(self):
        print("\n📊 Losses...")
        self.phase = PhaseLoss().cuda()
        self.stft = auraloss.freq.MultiResolutionSTFTLoss(
            fft_sizes=[256,512,1024,2048,4096],
            hop_sizes=[64,128,256,512,1024],
            win_lengths=[256,512,1024,2048,4096],
            w_log_mag=2.0, w_lin_mag=0.5, w_sc=1.0,
            sample_rate=self.cfg.sr_high
        ).cuda()
        print("✓ Ready")
    
    def _optim(self):
        self.opt = optim.AdamW(self.m.model.parameters(), lr=self.cfg.lr,
                              betas=(0.8,0.9), eps=1e-8, 
                              weight_decay=self.cfg.weight_decay)
        self.sch = StepLR(self.opt, self.cfg.sched_step, self.cfg.sched_gamma)
    
    def _data(self):
        print("\n" + "="*80)
        self.data = DataPipe(self.cfg)
        print("="*80)
    
    def _display_comparison(self, input_16k, predicted, original, noisy, bidx):
        """Display audio comparison in notebook"""
        if not self.cfg.compare or not self.in_notebook:
            return
        
        try:
            # Take first sample from batch and detach from graph
            inp = input_16k[0].detach().cpu().numpy()
            pred = predicted[0].detach().cpu().numpy()
            orig = original[0].detach().cpu().numpy()
            
            # Create HTML display
            html = f"""
            <div style='background:#f5f5f5; padding:15px; border-radius:8px; margin:10px 0'>
                <h4 style='margin:0 0 10px 0; color:#333'>🔊 Audio Comparison - Batch {bidx}</h4>
                <div style='display:grid; gap:10px'>
            """
            
            self.IPython.display(self.IPython.HTML(html))
            
            # Show noisy audio if in denoise mode and noise was added
            if self.cfg.add_noise and noisy is not None:
                noisy_np = noisy[0].detach().cpu().numpy()
                # Check if actually noisy (different from original) - both on CPU
                noisy_cpu = noisy[0].detach().cpu()
                orig_cpu = original[0].detach().cpu()
                if not torch.allclose(noisy_cpu, orig_cpu, atol=1e-5):
                    print("  🔊 Noisy (48kHz - with added noise):")
                    self.IPython.display(self.IPython.Audio(noisy_np, rate=self.cfg.sr_high))
            
            # Input (16kHz)
            print("  📥 Input (16kHz - downsampled):")
            self.IPython.display(self.IPython.Audio(inp, rate=self.cfg.sr_low))
            
            # Predicted (48kHz)
            print("  🤖 Predicted (48kHz - model output):")
            self.IPython.display(self.IPython.Audio(pred, rate=self.cfg.sr_high))
            
            # Original (48kHz)
            print("  ✨ Original (48kHz - ground truth):")
            self.IPython.display(self.IPython.Audio(orig, rate=self.cfg.sr_high))
            
            html_end = "</div></div>"
            self.IPython.display(self.IPython.HTML(html_end))
            
        except Exception as e:
            print(f"⚠️  Display error: {e}")
    
    def step(self, a48, a16, a48_noisy=None):
        self.opt.zero_grad()
        
        a16 = a16.cuda().unsqueeze(1)
        pred = self.m.model(a16).squeeze(1)
        
        ml = min(pred.shape[1], a48.shape[1])
        pred, a48 = pred[:,:ml], a48[:,:ml]
        
        stft = self.stft(pred.unsqueeze(0), a48.unsqueeze(0))
        phase = self.phase(pred, a48)
        l = lsd(pred, a48)
        
        tot = (stft * self.cfg.stft_weight + 
               phase * self.cfg.phase_weight + 
               l * self.cfg.lsd_weight)
        
        tot.backward()
        torch.nn.utils.clip_grad_norm_(self.m.model.parameters(), self.cfg.grad_clip)
        self.opt.step()
        self.sch.step()
        
        return tot.item(), stft.item(), phase.item(), l.item(), pred
    
    def train(self):
        print("\n" + "="*80)
        print(f"TRAINING ({self.cfg.mode.upper()} MODE)")
        print("="*80)
        
        for ep in range(self.cfg.epochs):
            print(f"\n{'='*80}")
            print(f"EPOCH {ep+1}/{self.cfg.epochs}")
            print(f"{'='*80}")
            
            loader = self.data.loader(self.cfg.batch_size)
            eloss, bcnt = 0, 0
            
            for bidx, batch in enumerate(loader):
                a48 = batch['a48'].cuda()
                a16 = batch['a16']
                a48_noisy = batch['a48_noisy']
                
                tot, stft, phase, l, pred = self.step(a48, a16)
                
                eloss += tot
                bcnt += 1
                
                if bidx % self.cfg.log_interval == 0:
                    lr = self.sch.get_last_lr()[0]
                    st = '✓' if l < 1.0 else '⚠️'
                    print(f"\n📊 Batch {bidx}")
                    print(f"  Tot: {tot:.4f} | STFT: {stft:.4f} | Phase: {phase:.4f}")
                    print(f"  LSD: {l:.4f} {st} | LR: {lr:.6f}")
                    
                    # Display audio comparison if enabled
                    self._display_comparison(a16, pred, a48, a48_noisy, bidx)
            
            avg = eloss / max(bcnt, 1)
            print(f"\n📈 Epoch {ep+1} Avg: {avg:.4f}")
        
        print("\n" + "="*80)
        print("✅ COMPLETE")
        print("="*80)
        
        return self._save()
    
    def _save(self):
        print("\n💾 Saving...")
        
        self.m.model.half().eval()
        for mod in self.m.model.modules():
            if hasattr(mod, 'prepare'):
                mod.prepare()
        
        name = f"novasr_{self.cfg.mode}.bin"
        torch.save(self.m.model.state_dict(), name)
        print(f"✓ {name}")
        
        try:
            kp = Path('/kaggle/working')
            if kp.exists():
                torch.save(self.m.model.state_dict(), kp / name)
                print(f"✓ Kaggle")
        except:
            pass
        
        return self.m

# ============================================================================
# MAIN
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description='Train NovaSR')
    p.add_argument('--mode', choices=['denoise','upsample'], default='denoise',
                   help='Training mode: denoise (with noise) or upsample (clean only)')
    p.add_argument('--model', type=str, default=None,
                   help='Path to custom pretrained model (.bin file)')
    p.add_argument('--epochs', type=int, default=10,
                   help='Number of training epochs')
    p.add_argument('--batch-size', type=int, default=8,
                   help='Batch size for training')
    p.add_argument('--lr', type=float, default=1e-5,
                   help='Learning rate')
    p.add_argument('--from-scratch', action='store_true',
                   help='Train from random initialization (ignored if --model is specified)')
    p.add_argument('--compare', action='store_true',
                   help='Display audio comparison in notebook')
    return p.parse_args()

def main():
    args = parse_args()
    
    print("\n" + "="*80)
    print("NOVASR TRAINING")
    print("="*80)
    print(f"\n🎯 Mode: {args.mode.upper()}")
    if args.model:
        print(f"📦 Custom Model: {args.model}")
    elif args.from_scratch:
        print(f"🎲 Training: From scratch (random init)")
    else:
        print(f"📦 Training: Fine-tuning default pretrained model")
    print(f"📊 Epochs: {args.epochs}")
    print(f"📦 Batch: {args.batch_size}")
    print(f"📈 LR: {args.lr}")
    if args.compare:
        print(f"🔊 Audio comparison: Enabled")
    
    cfg = Config(
        mode=args.mode,
        model_path=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        from_scratch=args.from_scratch and not args.model,  # Ignore from_scratch if model is specified
        compare=args.compare
    )
    
    trainer = Trainer(cfg)
    model = trainer.train()
    
    print("\n" + "="*80)
    print("🎉 SUCCESS!")
    print("="*80)
    print(f"\n💾 Saved: novasr_{cfg.mode}.bin")
    print("\n💡 Usage:")
    print("  from NovaSR import FastSR")
    print(f"  model = FastSR('novasr_{cfg.mode}.bin')")
    print("  out = model.infer(input.unsqueeze(1))")
    
    return model

if __name__ == "__main__":
    main()
