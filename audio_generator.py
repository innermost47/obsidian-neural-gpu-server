import threading
import gc
import time
from typing import Optional
import torch
from diffusers import StableAudioPipeline
from diffusers import (
    BitsAndBytesConfig as DiffusersBitsAndBytesConfig,
    StableAudioDiTModel,
)
from transformers import (
    BitsAndBytesConfig as TransformersBitsAndBytesConfig,
    T5EncoderModel,
)
from settings import SUPPORTED_MODELS, TARGET_SAMPLE_RATE, MIN_DURATION
from base_generator import Generator


class AudioGenerator(Generator):
    def __init__(self, model_key: str = "stable-audio-open-1.0"):
        super().__init__()
        self.model_key = model_key
        self.model_id = SUPPORTED_MODELS[model_key]
        self.pipeline = None
        self.sample_rate = TARGET_SAMPLE_RATE
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._lock = threading.Lock()
        self._generating = False

    def load(self):

        print(f"⚡ Loading {self.model_id} on {self.device}...")

        if self.device == "cuda":
            text_encoder = T5EncoderModel.from_pretrained(
                self.model_id,
                subfolder="text_encoder",
                quantization_config=TransformersBitsAndBytesConfig(load_in_8bit=True),
                torch_dtype=torch.float16,
            )
            transformer = StableAudioDiTModel.from_pretrained(
                self.model_id,
                subfolder="transformer",
                quantization_config=DiffusersBitsAndBytesConfig(load_in_8bit=True),
                torch_dtype=torch.float16,
            )
            self.pipeline = StableAudioPipeline.from_pretrained(
                self.model_id,
                text_encoder=text_encoder,
                transformer=transformer,
                torch_dtype=torch.float16,
                device_map="balanced",
            )
        else:
            raise RuntimeError("No CUDA GPU available. CPU mode is not allowed.")

        self.sample_rate = self.pipeline.vae.sampling_rate
        print(f"✅ Model loaded (sample rate: {self.sample_rate}Hz)")

    def unload(self):
        if self.pipeline is not None:
            try:
                self.pipeline.to("cpu")
            except Exception:
                pass
            try:
                del self.pipeline.text_encoder
                del self.pipeline.transformer
                del self.pipeline.vae
            except Exception:
                pass
        self.pipeline = None

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()

        try:
            import ctypes

            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except Exception:
            pass

    def generate_with_seed(
        self,
        prompt: str,
        duration: int,
        seed: int,
        bpm: Optional[int] = None,
        key: Optional[str] = None,
    ) -> bytes:
        with self._lock:
            self._generating = True
            try:
                return self._generate_with_seed(prompt, duration, seed, bpm, key)
            finally:
                self._generating = False

    def _generate_with_seed(
        self,
        prompt: str,
        duration: int,
        seed: int,
        bpm: Optional[int] = None,
        key: Optional[str] = None,
    ) -> bytes:
        try:
            if not prompt or not prompt.strip():
                raise ValueError("prompt cannot be None or empty")

            seed = (
                seed if seed is not None else torch.randint(0, 2**31 - 1, (1,)).item()
            )

            self.load()
            duration = max(MIN_DURATION, min(10, duration))
            num_inference_steps = 50
            cfg_scale = 7.0

            final_prompt = prompt.strip()
            if key:
                final_prompt += f", {key}"
            if bpm:
                final_prompt += f", {bpm} BPM"

            gen = torch.Generator(device=self.device).manual_seed(seed)
            print(f"Stable audio open 1.0 prompt: {final_prompt}")
            t0 = time.time()

            result = self.pipeline(
                final_prompt,
                negative_prompt="Low quality, distorted, noise",
                num_inference_steps=num_inference_steps,
                audio_end_in_s=duration,
                num_waveforms_per_prompt=1,
                generator=gen,
                guidance_scale=cfg_scale,
            )

            print(f"✅ Verify done in {time.time() - t0:.1f}s")

            audio = result.audios[0].float()
            audio = self._finalize_audio(audio, duration, TARGET_SAMPLE_RATE)
            return audio
        finally:
            self.unload()
