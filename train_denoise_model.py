"""
Script 1: Train NovaSR Denoising + Super-Resolution Model
File: train_denoise_model.py

Usage:
    python train_denoise_model.py

Output:
    novasr_denoise_upsample.bin (trained model)
"""

import subprocess
import sys
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple, Generator
from dataclasses import dataclass
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio.transforms as T
from torch.optim.lr_scheduler import StepLR

warnings.filterwarnings('ignore')

print("="*80)
print("NOVASR DENOISING + SUPER-RESOLUTION TRAINING")
print("="*80)

# ============================================================================
# INSTALL DEPENDENCIES
# ============================================================================

def install_dependencies():
    print("\n📦 Installing dependencies...")
    packages = [
        "torch", "torchaudio", 
        "datasets", "soundfile", "librosa",
        "auraloss", "einops",
        "huggingface_hub"
    ]
    
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--upgrade", "-q"] + packages
        )
        print("✓ All dependencies installed\n")
    except Exception as e:
        print(f"⚠️  Warning: {e}")
        print("Continuing anyway...\n")

install_dependencies()

# Import after installation
try:
    import auraloss
    from datasets import load_dataset, Audio as AudioFeature
    from NovaSR import FastSR
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nPlease install NovaSR first:")
    print("  pip install git+https://github.com/YourRepo/NovaSR.git")
    sys.exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Training configuration - EDIT THESE VALUES"""
    
    # Dataset
    dataset: str = 'ai4bharat/Rasa'
    dataset_config: str = 'Urdu'
    audio_column: str = 'audio'
    max_samples: int = 8000  # Limit dataset size
    min_duration_sec: float = 4.0
    
    # Noise Settings (KEY FOR DENOISING!)
    add_noise: bool = True  # Set False for upsample-only training
    noise_types: list = None
    noise_snr_range: Tuple[float, float] = (0, 20)  # 0dB=very noisy, 20dB=light noise
    noise_probability: float = 0.7  # 70% of samples get noise
    
    # Training
    train_from_scratch: bool = False  # False = fine-tune, True = random init
    batch_size: int = 8
    learning_rate: float = 1e-5  # Lower for fine-tuning
    epochs: int = 10  # 5-10 for fine-tuning, 20-30 for scratch
    gradient_clip: float = 1.0
    weight_decay: float = 0.01
    
    # Loss weights
    stft_weight: float = 10.0
    phase_weight: float = 1.0
    lsd_weight: float = 1.0
    
    # Advanced
    scheduler_step: int = 1000
    scheduler_gamma: float = 0.99
    log_interval: int = 50
    sample_interval: int = 100
    
    # Audio
    sr_high: int = 48000
    sr_low: int = 16000
    
    def __post_init__(self):
        if self.noise_types is None:
            # Default: all noise types
            self.noise_types = ['white', 'pink', 'brown', 'environmental']
    
    @property
    def crop_length(self) -> int:
        return int(self.min_duration_sec * self.sr_high)

# ============================================================================
# NOISE GENERATOR
# ============================================================================

class NoiseGenerator:
    """Generate realistic noise for training"""
    
    @staticmethod
    def white_noise(shape, device='cpu'):
        """TV static, microphone self-noise"""
        return torch.randn(shape, device=device)
    
    @staticmethod
    def pink_noise(shape, device='cpu'):
        """Natural ambient noise (rain, ocean)"""
        white = torch.randn(shape, device=device)
        fft = torch.fft.rfft(white)
        freqs = torch.arange(fft.shape[-1], device=device) + 1
        pink_fft = fft / torch.sqrt(freqs.float())
        pink = torch.fft.irfft(pink_fft, n=shape[-1])
        return pink / (pink.std() + 1e-8)
    
    @staticmethod
    def brown_noise(shape, device='cpu'):
        """Low-frequency rumble (traffic, machinery)"""
        white = torch.randn(shape, device=device)
        fft = torch.fft.rfft(white)
        freqs = torch.arange(fft.shape[-1], device=device) + 1
        brown_fft = fft / freqs.float()
        brown = torch.fft.irfft(brown_fft, n=shape[-1])
        return brown / (brown.std() + 1e-8)
    
    @staticmethod
    def environmental_noise(shape, device='cpu'):
        """Realistic mix: office, cafe, street"""
        low = NoiseGenerator.brown_noise(shape, device) * 0.7
        mid = NoiseGenerator.pink_noise(shape, device) * 0.3
        high = NoiseGenerator.white_noise(shape, device) * 0.1
        env = low + mid + high
        return env / (env.std() + 1e-8)
    
    @staticmethod
    def add_noise_at_snr(clean, noise_type, snr_db, device='cpu'):
        """
        Add noise at specified SNR (Signal-to-Noise Ratio)
        
        SNR Guide:
          0-5 dB:  Very noisy (traffic, construction)
          5-10 dB: Noisy (busy cafe)
          10-15 dB: Moderate (office)
          15-20 dB: Light (quiet room)
        """
        generators = {
            'white': NoiseGenerator.white_noise,
            'pink': NoiseGenerator.pink_noise,
            'brown': NoiseGenerator.brown_noise,
            'environmental': NoiseGenerator.environmental_noise,
        }
        
        gen = generators.get(noise_type, NoiseGenerator.white_noise)
        noise = gen(clean.shape, device=device)
        
        # Calculate scaling for desired SNR
        signal_power = clean.pow(2).mean()
        noise_power = noise.pow(2).mean()
        snr_linear = 10 ** (snr_db / 10)
        scale = torch.sqrt(signal_power / (snr_linear * noise_power + 1e-8))
        
        noisy = clean + scale * noise
        return noisy, noise * scale

# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class MultiResPhaseTrioLoss(nn.Module):
    """Phase-aware loss for natural audio quality"""
    
    def __init__(self, n_ffts=(512, 1024, 2048), hop_lengths=(128, 256, 512), win_lengths=(512, 1024, 2048)):
        super().__init__()
        self.n_ffts = n_ffts
        self.hop_lengths = hop_lengths
        self.win_lengths = win_lengths
        
        for i, win_len in enumerate(win_lengths):
            self.register_buffer(f"window_{i}", torch.hann_window(win_len))
    
    @staticmethod
    def phase_diff(a, b):
        return torch.atan2(torch.sin(a - b), torch.cos(a - b))
    
    def compute_phase_loss(self, y_hat, y, n_fft, hop_len, win_len, window):
        window = window.to(y_hat.device)
        
        s_hat = torch.stft(y_hat, n_fft, hop_len, win_len, window, return_complex=True)
        s_y = torch.stft(y, n_fft, hop_len, win_len, window, return_complex=True)
        
        p_hat = torch.angle(s_hat)
        p_y = torch.angle(s_y)
        mag = torch.abs(s_y) + 1e-7
        
        mag_w = mag ** 0.3
        mag_w = mag_w / (mag_w.amax(dim=(-2, -1), keepdim=True) + 1e-7)
        
        dp = self.phase_diff(p_hat, p_y)
        loss_pw = torch.mean(mag_w * (1 - torch.cos(dp)))
        
        dpt_hat = self.phase_diff(p_hat[..., 1:], p_hat[..., :-1])
        dpt_y = self.phase_diff(p_y[..., 1:], p_y[..., :-1])
        dif = self.phase_diff(dpt_hat, dpt_y)
        loss_if = torch.mean(mag_w[..., 1:] * (1 - torch.cos(dif)))
        
        dpf_hat = self.phase_diff(p_hat[..., 1:, :], p_hat[..., :-1, :])
        dpf_y = self.phase_diff(p_y[..., 1:, :], p_y[..., :-1, :])
        dgd = self.phase_diff(dpf_hat, dpf_y)
        loss_gd = torch.mean(mag_w[..., 1:, :] * (1 - torch.cos(dgd)))
        
        return loss_pw + loss_if + loss_gd
    
    def forward(self, y_hat, y):
        scale = torch.sqrt(torch.mean(y**2, dim=-1, keepdim=True) + 1e-7)
        y = y / scale
        y_hat = y_hat / scale.detach()
        
        total = 0
        for i in range(len(self.n_ffts)):
            total += self.compute_phase_loss(
                y_hat, y, self.n_ffts[i], self.hop_lengths[i],
                self.win_lengths[i], getattr(self, f"window_{i}")
            )
        return total / len(self.n_ffts)


def compute_lsd(target, estimate, n_fft=2048, hop=512, eps=1e-8):
    """Log-Spectral Distance (perceptual quality metric)"""
    if target.ndim == 1:
        target = target.unsqueeze(0)
        estimate = estimate.unsqueeze(0)
    
    window = torch.hann_window(n_fft, device=target.device)
    Ya = torch.stft(target, n_fft=n_fft, hop_length=hop, window=window, return_complex=True).abs()
    Yhat = torch.stft(estimate, n_fft=n_fft, hop_length=hop, window=window, return_complex=True).abs()
    
    log_ratio = torch.log10((Ya + eps) / (Yhat + eps))
    dist = torch.sqrt(torch.mean(log_ratio**2, dim=1))
    return dist.mean(dim=1).mean()

# ============================================================================
# DATA PIPELINE
# ============================================================================

class DataPipeline:
    """Load and augment training data"""
    
    def __init__(self, config: Config):
        self.config = config
        self.resampler = T.Resample(config.sr_high, config.sr_low)
        self.noise_gen = NoiseGenerator()
        self._load_dataset()
    
    def _load_dataset(self):
        print(f"\n📂 Loading dataset: {self.config.dataset} ({self.config.dataset_config})")
        
        try:
            self.dataset = load_dataset(
                self.config.dataset, 
                self.config.dataset_config,
                split='train', 
                streaming=True
            )
            
            # Manual audio decoding (avoid torchcodec)
            self.dataset = self.dataset.cast_column(
                self.config.audio_column, 
                AudioFeature(decode=False)
            )
            
            self.dataset = self.dataset.take(self.config.max_samples)
            
            print(f"✓ Dataset ready (~{self.config.max_samples} samples)")
            
            if self.config.add_noise:
                print(f"\n🔊 Noise augmentation:")
                print(f"  Types: {', '.join(self.config.noise_types)}")
                print(f"  SNR range: {self.config.noise_snr_range[0]}-{self.config.noise_snr_range[1]} dB")
                print(f"  Probability: {self.config.noise_probability * 100:.0f}%")
            else:
                print("\n✓ No noise augmentation (upsample-only mode)")
                
        except Exception as e:
            print(f"\n❌ Dataset error: {e}")
            print("\nTroubleshooting:")
            print("  1. Run: huggingface-cli login")
            print("  2. Accept terms: https://huggingface.co/datasets/ai4bharat/Rasa")
            raise
    
    def _decode_audio(self, audio_data):
        """Decode audio using soundfile"""
        import io
        import soundfile as sf
        
        try:
            if 'bytes' in audio_data and audio_data['bytes']:
                array, sr = sf.read(io.BytesIO(audio_data['bytes']))
            elif 'path' in audio_data and audio_data['path']:
                array, sr = sf.read(audio_data['path'])
            else:
                return None
            
            audio = torch.from_numpy(array).float()
            if audio.ndim > 1:
                audio = audio.mean(dim=0)
            return audio, sr
        except:
            return None
    
    def _add_noise(self, audio_48k):
        """Add noise augmentation"""
        if not self.config.add_noise:
            return audio_48k
        
        if random.random() > self.config.noise_probability:
            return audio_48k
        
        noise_type = random.choice(self.config.noise_types)
        snr_db = random.uniform(*self.config.noise_snr_range)
        noisy, _ = self.noise_gen.add_noise_at_snr(audio_48k, noise_type, snr_db)
        return noisy
    
    def _augment_lowres(self, audio_16k):
        """Quantization and amplitude variation"""
        if random.random() > 0.5:
            bits = random.choice([8, 12, 14])
            levels = 2 ** bits
            audio_16k = torch.round(audio_16k * levels) / levels
            audio_16k = audio_16k * random.uniform(0.9, 1.1)
        return audio_16k
    
    def generate_segments(self):
        """Generate training segments"""
        count = 0
        skipped = 0
        noise_count = 0
        
        print("\n🔄 Processing audio...")
        
        for idx, item in enumerate(self.dataset):
            try:
                result = self._decode_audio(item[self.config.audio_column])
                if result is None:
                    skipped += 1
                    continue
                
                audio, sr = result
                
                # Resample to 48kHz
                if sr != self.config.sr_high:
                    resampler = T.Resample(sr, self.config.sr_high)
                    audio = resampler(audio)
                
                # Skip short audio
                if audio.shape[-1] < self.config.crop_length:
                    skipped += 1
                    continue
                
                # Split into segments
                num_segments = audio.shape[-1] // self.config.crop_length
                for i in range(num_segments):
                    start = i * self.config.crop_length
                    segment_clean = audio[start:start + self.config.crop_length]
                    
                    # Add noise to 48kHz
                    segment_noisy = self._add_noise(segment_clean)
                    
                    # Downsample to 16kHz
                    segment_16k = self.resampler(segment_noisy)
                    segment_16k = self._augment_lowres(segment_16k)
                    
                    if not torch.allclose(segment_noisy, segment_clean, atol=1e-5):
                        noise_count += 1
                    
                    # Yield (clean_target, noisy_input)
                    yield segment_clean, segment_16k
                    count += 1
                
                if (idx + 1) % 10 == 0:
                    print(f"  Items: {idx+1}, Segments: {count} ({noise_count} noisy)...", end='\r')
                    
            except:
                skipped += 1
                continue
        
        print(f"\n✓ Generated {count} segments (skipped {skipped})")
        if self.config.add_noise:
            print(f"  Clean: {count-noise_count}, Noisy: {noise_count} ({noise_count/count*100:.1f}%)")
    
    def create_dataloader(self, batch_size):
        """Create batched dataloader"""
        segment_gen = self.generate_segments()
        batch_48k, batch_16k = [], []
        
        for s48, s16 in segment_gen:
            batch_48k.append(s48)
            batch_16k.append(s16)
            
            if len(batch_48k) == batch_size:
                yield {
                    'audio_48k': torch.stack(batch_48k),
                    'audio_16k': torch.stack(batch_16k)
                }
                batch_48k, batch_16k = [], []

# ============================================================================
# TRAINER
# ============================================================================

class Trainer:
    """Main training orchestrator"""
    
    def __init__(self, config: Config):
        self.config = config
        self._setup_model()
        self._setup_losses()
        self._setup_optimizer()
        self._setup_data()
    
    def _setup_model(self):
        print("\n🤖 Initializing NovaSR model...")
        self.model = FastSR()
        
        # Count parameters
        total = sum(p.numel() for p in self.model.model.parameters())
        print(f"✓ Model: {total:,} parameters ({total * 2 / 1024:.1f} KB)")
        
        if self.config.train_from_scratch:
            print("⚠️  Resetting to random weights...")
            def reset(m):
                if hasattr(m, 'reset_parameters'):
                    m.reset_parameters()
                elif isinstance(m, (nn.Conv1d, nn.Linear)):
                    nn.init.kaiming_normal_(m.weight)
                    if hasattr(m, 'bias') and m.bias is not None:
                        nn.init.zeros_(m.bias)
            self.model.model.apply(reset)
            print("✓ Random initialization")
        else:
            print("✓ Using pre-trained weights (fine-tuning)")
        
        self.model.model.train().cuda().float()
    
    def _setup_losses(self):
        print("\n📊 Initializing losses...")
        self.phase_loss = MultiResPhaseTrioLoss().cuda()
        self.stft_loss = auraloss.freq.MultiResolutionSTFTLoss(
            fft_sizes=[256, 512, 1024, 2048, 4096],
            hop_sizes=[64, 128, 256, 512, 1024],
            win_lengths=[256, 512, 1024, 2048, 4096],
            w_log_mag=2.0, w_lin_mag=0.5, w_sc=1.0,
            sample_rate=self.config.sr_high
        ).cuda()
        print("✓ Losses ready")
    
    def _setup_optimizer(self):
        self.optimizer = optim.AdamW(
            self.model.model.parameters(),
            lr=self.config.learning_rate,
            betas=(0.8, 0.9), eps=1e-8,
            weight_decay=self.config.weight_decay
        )
        self.scheduler = StepLR(
            self.optimizer, 
            self.config.scheduler_step, 
            self.config.scheduler_gamma
        )
    
    def _setup_data(self):
        print("\n" + "=" * 80)
        self.data = DataPipeline(self.config)
        print("=" * 80)
    
    def train_step(self, audio_48k, audio_16k):
        """Single training step"""
        self.optimizer.zero_grad()
        
        # Forward
        audio_16k = audio_16k.cuda().unsqueeze(1)
        pred = self.model.model(audio_16k).squeeze(1)
        
        # Align lengths
        min_len = min(pred.shape[1], audio_48k.shape[1])
        pred = pred[:, :min_len]
        audio_48k = audio_48k[:, :min_len]
        
        # Losses
        stft = self.stft_loss(pred.unsqueeze(0), audio_48k.unsqueeze(0))
        phase = self.phase_loss(pred, audio_48k)
        lsd = compute_lsd(pred, audio_48k)
        
        total = (
            stft * self.config.stft_weight +
            phase * self.config.phase_weight +
            lsd * self.config.lsd_weight
        )
        
        # Backward
        total.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.model.parameters(), 
            self.config.gradient_clip
        )
        self.optimizer.step()
        self.scheduler.step()
        
        return total.item(), stft.item(), phase.item(), lsd.item()
    
    def train(self):
        """Main training loop"""
        print("\n" + "=" * 80)
        print("STARTING TRAINING")
        print("=" * 80)
        
        for epoch in range(self.config.epochs):
            print(f"\n{'=' * 80}")
            print(f"EPOCH {epoch + 1}/{self.config.epochs}")
            print(f"{'=' * 80}")
            
            loader = self.data.create_dataloader(self.config.batch_size)
            epoch_loss = 0
            batch_count = 0
            
            for batch_idx, batch in enumerate(loader):
                audio_48k = batch['audio_48k'].cuda()
                audio_16k = batch['audio_16k']
                
                total, stft, phase, lsd = self.train_step(audio_48k, audio_16k)
                
                epoch_loss += total
                batch_count += 1
                
                if batch_idx % self.config.log_interval == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    status = '✓' if lsd < 1.0 else '⚠️'
                    print(f"\n📊 Batch {batch_idx}")
                    print(f"  Total: {total:.4f} | STFT: {stft:.4f} | Phase: {phase:.4f}")
                    print(f"  LSD: {lsd:.4f} {status} | LR: {lr:.6f}")
            
            avg = epoch_loss / max(batch_count, 1)
            print(f"\n📈 Epoch {epoch + 1} Average Loss: {avg:.4f}")
        
        print("\n" + "=" * 80)
        print("✅ TRAINING COMPLETE")
        print("=" * 80)
        
        return self._save_model()
    
    def _save_model(self):
        """Save trained model"""
        print("\n💾 Saving model...")
        
        # Prepare for inference
        self.model.model.half().eval()
        for module in self.model.model.modules():
            if hasattr(module, 'prepare'):
                module.prepare()
        
        # Save
        save_path = "novasr_denoise_upsample.bin"
        torch.save(self.model.model.state_dict(), save_path)
        print(f"✓ Saved: {save_path}")
        
        # Kaggle
        try:
            kaggle = Path('/kaggle/working')
            if kaggle.exists():
                torch.save(self.model.model.state_dict(), kaggle / save_path)
                print(f"✓ Saved to Kaggle")
        except:
            pass
        
        return self.model

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n🎯 TASK: Train model for denoising + super-resolution")
    print("📊 INPUT: Noisy 16kHz audio")
    print("🎵 OUTPUT: Clean 48kHz audio\n")
    
    # Configuration
    config = Config(
        # Dataset
        max_samples=8000,
        min_duration_sec=4.0,
        
        # Noise (KEY SETTINGS!)
        add_noise=True,  # Enable denoising
        noise_types=['white', 'pink', 'brown', 'environmental'],
        noise_snr_range=(0, 20),  # 0dB = very noisy, 20dB = light noise
        noise_probability=0.7,  # 70% samples get noise
        
        # Training
        train_from_scratch=False,  # False = fine-tune pre-trained
        learning_rate=1e-5,
        epochs=10,
        batch_size=8,
    )
    
    # Train
    trainer = Trainer(config)
    model = trainer.train()
    
    print("\n" + "=" * 80)
    print("🎉 SUCCESS!")
    print("=" * 80)
    print("\n📦 Saved model: novasr_denoise_upsample.bin")
    print("\n💡 Usage:")
    print("  from NovaSR import FastSR")
    print("  model = FastSR('novasr_denoise_upsample.bin')")
    print("  clean_48k = model.infer(noisy_16k.unsqueeze(1))")
    print("\n✨ Model does: Denoise + Upsample simultaneously!")
    
    return model


if __name__ == "__main__":
    model = main()
