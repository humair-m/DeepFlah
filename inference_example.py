"""Example inference usage"""
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
