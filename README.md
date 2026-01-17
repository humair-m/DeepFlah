# NovaSR - Neural Audio Super-Resolution

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
