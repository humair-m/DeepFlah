"""NovaSR - Fast Neural Audio Super-Resolution"""
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
