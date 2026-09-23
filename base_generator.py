import io
import torch
from einops import rearrange
import numpy as np
import soundfile as sf
from settings import SUPPORTED_BPM


class Generator:
    def __init__(self):
        pass

    def _finalize_audio(self, audio, clip_seconds, sample_rate):
        try:
            if isinstance(audio, np.ndarray):
                audio = torch.from_numpy(audio)
            if audio.ndim == 3:
                audio = rearrange(audio, "b d n -> d (b n)")
            audio = audio.to(torch.float32).clamp(-1, 1)

            clip_samples = int(round(clip_seconds * sample_rate))
            audio = audio[:, : max(1, min(audio.shape[-1], clip_samples))].contiguous()

            fade_len = min(int(round(0.015 * sample_rate)), audio.shape[-1])
            if fade_len > 1:
                ramp = torch.linspace(
                    1.0, 0.0, steps=fade_len, device=audio.device, dtype=audio.dtype
                )
                audio[:, -fade_len:] *= ramp

            audio_np = audio.cpu().numpy()
            max_val = np.max(np.abs(audio_np))
            if max_val > 0:
                audio_np = audio_np / max_val * 0.9

            if audio_np.ndim == 2 and audio_np.shape[0] == 2:
                audio_to_write = np.ascontiguousarray(audio_np.T)
            else:
                audio_to_write = audio_np

            buf = io.BytesIO()
            sf.write(buf, audio_to_write, sample_rate, format="WAV", subtype="PCM_16")
            buf.seek(0)
            return buf.read()
        finally:
            del audio

    def _nearest_supported_bpm(self, bpm: int) -> int:
        return min(SUPPORTED_BPM, key=lambda x: abs(x - bpm))
