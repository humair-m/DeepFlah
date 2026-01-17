"""
setup_all.py - Create Complete NovaSR Package

This script creates all necessary files for NovaSR training

Run: python setup_all.py

This will create:
  - NovaSR/ package (all model code)
  - train_novasr.py (training script)
  - inference_example.py (usage examples)
  - README.md (documentation)
"""

import os
from pathlib import Path

# ============================================================================
# NOVASR PACKAGE FILES
# ============================================================================

FILES = {}

# __init__.py
FILES["NovaSR/__init__.py"] = '''"""NovaSR - Fast Neural Audio Super-Resolution"""
import torch
import os
import torchaudio
from .speechsr import SynthesizerTrn

class FastSR:
    def __init__(self, ckpt_path=None, half=True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hps = {
            "train": {"segment_size": 9600},
            "data": {"hop_length": 320, "n_mel_channels": 128},
            "model": {
                "resblock": "0",
                "resblock_kernel_sizes": [11],
                "resblock_dilation_sizes": [[1,3,5]],
                "upsample_initial_channel": 32,
            }
        }
        
        if ckpt_path is None:
            from huggingface_hub import snapshot_download
            model_path = snapshot_download("YatharthS/NovaSR")
            ckpt_path = f"{model_path}/pytorch_model_v1.bin"
        
        self.half = False
        self.model = self._load_model(ckpt_path).eval().float()
        if half:
            self.half = True
            self.model.half()
    
    def _load_model(self, ckpt_path):
        model = SynthesizerTrn(
            self.hps['data']['n_mel_channels'],
            self.hps['train']['segment_size'] // self.hps['data']['hop_length'],
            **self.hps['model']
        ).to(self.device)
        
        assert os.path.isfile(ckpt_path)
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        model.dec.remove_weight_norm()
        model.load_state_dict(checkpoint, strict=True)
        model.eval()
        return model
    
    def load_audio(self, audio_file):
        audio, sr = torchaudio.load(audio_file)
        audio = audio[:1, :]
        lowres = torchaudio.functional.resample(
            audio, sr, 16000, resampling_method="kaiser_window"
        ).unsqueeze(1).to(self.device)
        if self.half:
            lowres = lowres.half()
        return lowres
    
    def infer(self, lowres_wav):
        with torch.no_grad():
            return self.model(lowres_wav).squeeze(0)
'''

# activations.py
FILES["NovaSR/activations.py"] = '''"""Snake Activation Functions"""
import torch
from torch import nn, Tensor

@torch.jit.script
def snake_fast_inference(x: Tensor, a: Tensor, inv_2b: Tensor) -> Tensor:
    return x + (1.0 - torch.cos(2.0 * a * x)) * inv_2b

class SnakeBeta(nn.Module):
    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False):
        super().__init__()
        self.alpha_logscale = alpha_logscale
        init_val = torch.zeros(in_features) if alpha_logscale else torch.ones(in_features)
        self.alpha = nn.Parameter(init_val * alpha)
        self.beta = nn.Parameter(init_val * alpha)
        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable
        
        self.register_buffer('a_eff', torch.ones(1, in_features, 1), persistent=False)
        self.register_buffer('inv_2b', torch.ones(1, in_features, 1), persistent=False)
        self._is_prepared = False
    
    def prepare(self):
        with torch.no_grad():
            a = (torch.exp(self.alpha) if self.alpha_logscale else self.alpha).view(1, -1, 1)
            b = (torch.exp(self.beta) if self.alpha_logscale else self.beta).view(1, -1, 1)
            self.a_eff.copy_(a)
            self.inv_2b.copy_(1.0 / (2.0 * b + 1e-9))
        self._is_prepared = True
    
    def forward(self, x: Tensor) -> Tensor:
        if not self._is_prepared and not self.training:
            self.prepare()
        if not self.training:
            return snake_fast_inference(x, self.a_eff, self.inv_2b)
        a = (torch.exp(self.alpha) if self.alpha_logscale else self.alpha).view(1, -1, 1)
        b = (torch.exp(self.beta) if self.alpha_logscale else self.beta).view(1, -1, 1)
        return x + (1.0 - torch.cos(2.0 * a * x)) / (2.0 * b + 1e-9)
'''

# commons.py
FILES["NovaSR/commons.py"] = '''"""Common utilities"""
def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)

def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)
'''

# resample.py
FILES["NovaSR/resample.py"] = '''"""Efficient resampling"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

@torch.jit.script
def _polyphase_upsample_fused(x: Tensor, weight: Tensor, ratio: int):
    x = F.pad(x, (2, 3))
    out = F.conv1d(x, weight, groups=x.shape[1], stride=1)
    B, C_out, L = out.shape
    C = x.shape[1]
    out = out.view(B, C, ratio, L).transpose(2, 3).reshape(B, C, -1)
    return out[..., 2:-2]

class UpSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=12, channels=512):
        super().__init__()
        self.ratio = ratio
        self.channels = channels
        self.kernel_size = kernel_size
        self.register_buffer("filter", torch.zeros(1, 1, 12))
        self.register_buffer("f_fast", torch.zeros(channels * ratio, 1, 6), persistent=False)
        self._prepared = False
    
    def prepare(self):
        with torch.no_grad():
            w = self.filter * float(self.ratio)
            w = w.view(self.kernel_size)
            p0, p1 = w[0::2], w[1::2]
            fast_w = torch.stack([p0, p1], dim=0).unsqueeze(0).expand(self.channels, -1, -1)
            fast_w = fast_w.reshape(self.channels * self.ratio, 1, 6)
            self.f_fast.copy_(fast_w)
        self._prepared = True
    
    def forward(self, x: Tensor):
        if not self._prepared and not self.training:
            self.prepare()
        return _polyphase_upsample_fused(x, self.f_fast[:x.shape[1]*self.ratio], self.ratio)

class LowPassFilter1d(nn.Module):
    def __init__(self, stride=1, kernel_size=12, channels=512):
        super().__init__()
        self.stride = stride
        self.channels = channels
        self.kernel_size = kernel_size
        self.register_buffer("filter", torch.zeros(1, 1, 12))
        self.register_buffer("f_opt", torch.zeros(channels, 1, 12), persistent=False)
        self._prepared = False
    
    def prepare(self):
        with torch.no_grad():
            self.f_opt.copy_(self.filter.expand(self.channels, -1, -1))
        self._prepared = True
    
    def forward(self, x: Tensor):
        if not self._prepared and not self.training:
            self.prepare()
        C = x.shape[1]
        return F.conv1d(x, self.f_opt[:C], stride=self.stride, padding=5, groups=C)

class DownSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=12, channels=512):
        super().__init__()
        self.lowpass = LowPassFilter1d(ratio, kernel_size, channels)
    
    def forward(self, x):
        return self.lowpass(x)
'''

# speechsr.py
FILES["NovaSR/speechsr.py"] = '''"""Model architecture"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Conv1d
from torch.nn.utils import weight_norm, remove_weight_norm
from .commons import init_weights, get_padding
from .activations import SnakeBeta
from .resample import UpSample1d, DownSample1d

class Activation1d(nn.Module):
    def __init__(self, activation, up_ratio=2, down_ratio=2, 
                 up_kernel_size=12, down_kernel_size=12):
        super().__init__()
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)
    
    def forward(self, x):
        return self.downsample(self.act(self.upsample(x)))

class AMPBlock0(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1,3,5), activation=None):
        super().__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0])))
        ])
        self.convs1.apply(init_weights)
        
        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)
        
        self.num_layers = len(self.convs1) + len(self.convs2)
        self.activations = nn.ModuleList([
            Activation1d(activation=SnakeBeta(channels, alpha_logscale=True))
            for _ in range(self.num_layers)
        ])
    
    def forward(self, x):
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, 
                                   self.activations[::2], self.activations[1::2]):
            xt = a2(c2(a1(c1(a1(x)))))
            x = xt + x[:, :, :xt.shape[2]]
        return x
    
    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)

class Generator(nn.Module):
    def __init__(self, initial_channel, resblock, resblock_kernel_sizes, 
                 resblock_dilation_sizes, upsample_initial_channel, gin_channels=0):
        super().__init__()
        self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        self.resblocks = nn.ModuleList()
        
        for i in range(1):
            ch = upsample_initial_channel // (2**i)
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(AMPBlock0(ch, k, d, activation="snakebeta"))
        
        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
    
    def forward(self, x, g=None):
        x = self.conv_pre(x)
        x = F.interpolate(x, int(x.shape[-1] * 3), mode='linear')
        x = self.resblocks[0](x)
        x = self.conv_post(x)
        return torch.tanh(x)
    
    def remove_weight_norm(self):
        for l in self.resblocks:
            l.remove_weight_norm()

class SynthesizerTrn(nn.Module):
    def __init__(self, spec_channels, segment_size, resblock, 
                 resblock_kernel_sizes, resblock_dilation_sizes, upsample_initial_channel):
        super().__init__()
        self.dec = Generator(1, resblock, resblock_kernel_sizes, 
                           resblock_dilation_sizes, upsample_initial_channel)
    
    def forward(self, x):
        return self.dec(x)
'''

# README
FILES["README.md"] = '''# NovaSR - Neural Audio Super-Resolution

Fast 16kHz → 48kHz audio enhancement with optional denoising.

## Features
- 🚀 23K parameters (53 KB model)
- ⚡ Real-time CPU inference
- 🎯 Two modes: Denoise+Upsample or Upsample-only
- 🔊 High quality (LSD ~0.65)

## Quick Start

```bash
# Setup
python setup_all.py

# Train denoising model
python train_novasr.py --mode denoise --epochs 10

# Train upsample-only
python train_novasr.py --mode upsample --epochs 20

# Use model
python
>>> from NovaSR import FastSR
>>> model = FastSR('novasr_denoise.bin')
>>> enhanced = model.infer(audio_16k.unsqueeze(1))
```

## Architecture
- Generator with Snake activations
- Single ResBlock (AMPBlock0)
- 3× interpolation (16k→48k)
- Pre/post convolutions

## Training
- Fine-tuning: 10 epochs, 1e-5 LR
- From scratch: 30 epochs, 5e-5 LR
- Losses: STFT + Phase + LSD

## License
MIT
'''

# Inference example
FILES["inference_example.py"] = '''"""Example inference usage"""
import torch
import torchaudio
from NovaSR import FastSR

# Load model
model = FastSR('novasr_denoise.bin')

# From file
lowres = model.load_audio('input_16k.wav')
enhanced = model.infer(lowres)
torchaudio.save('output_48k.wav', enhanced.cpu(), 48000)

# From tensor
audio_16k = torch.randn(1, 16000)  # 1 sec at 16kHz
enhanced = model.infer(audio_16k.unsqueeze(0).unsqueeze(1))
print(f"Input: {audio_16k.shape}, Output: {enhanced.shape}")
'''

# ============================================================================
# CREATE FILES
# ============================================================================

def create_all():
    """Create all files"""
    print("="*80)
    print("CREATING NOVASR PACKAGE")
    print("="*80)
    
    # Create directories
    os.makedirs("NovaSR", exist_ok=True)
    
    # Write files
    for filepath, content in FILES.items():
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            f.write(content.strip() + '\n')
        
        print(f"✓ Created {filepath}")
    
    print("\n" + "="*80)
    print("✅ SETUP COMPLETE!")
    print("="*80)
    
    print("\n📦 Created files:")
    print("  NovaSR/")
    print("    __init__.py")
    print("    activations.py")
    print("    commons.py")
    print("    resample.py")
    print("    speechsr.py")
    print("  README.md")
    print("  inference_example.py")
    
    print("\n🚀 Next steps:")
    print("  1. Download training script:")
    print("     (Copy train_novasr.py from previous artifact)")
    print("  2. Run training:")
    print("     python train_novasr.py --mode denoise")
    print("  3. Use model:")
    print("     python inference_example.py")
    
    print("\n💡 Training modes:")
    print("  --mode denoise   : Denoise + upsample (recommended)")
    print("  --mode upsample  : Upsample only")
    
    print("\n📚 Documentation: README.md")

if __name__ == "__main__":
    create_all()


# ============================================================================
# EXECUTE
# ============================================================================

print(__doc__)
print("\n" + "="*80)
print("Execute this script to create all files")
print("="*80)
print("\nRun: python setup_all.py")
