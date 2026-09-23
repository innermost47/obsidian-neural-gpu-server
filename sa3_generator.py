import threading
from typing import Optional
import time
import torch
import gc
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond_inpaint
from base_generator import Generator
from settings import MIN_DURATION, MAX_DURATION, SA3_KEEP_IN_RAM


class StableAudio3Generator(Generator):

    def __init__(self, repo_id: str, model_key: str = ""):
        super().__init__()
        self.repo_id = repo_id
        self.model_key = model_key
        self._lock = threading.Lock()
        self._generating = False
        self._cached_model = None
        self._cached_config = None

    def generate(
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
                return self._generate(prompt, duration, seed, bpm, key)
            finally:
                self._generating = False

    def _generate(
        self,
        prompt: str,
        duration: int,
        seed: int,
        bpm: Optional[int] = None,
        key: Optional[str] = None,
    ) -> bytes:

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_half = device.type == "cuda"

        if not prompt or not prompt.strip():
            raise ValueError("prompt cannot be None or empty")

        seed = seed if seed is not None else torch.randint(0, 2**31 - 1, (1,)).item()
        duration = max(MIN_DURATION, min(MAX_DURATION, duration))

        model = None
        model_config = None
        try:
            print(f"⚡ Loading {self.repo_id} (SA3) on {device}...")
            if SA3_KEEP_IN_RAM and self._cached_model is not None:
                print(f"♻️  SA3 cache hit (RAM → GPU)")
                model = self._cached_model
                model_config = self._cached_config
                model.to(device)
                if use_half:
                    model = model.to(torch.float16)
            else:
                print(f"⚡ Loading {self.repo_id} (SA3) on {device}...")
                model, model_config = get_pretrained_model(self.repo_id)
                if SA3_KEEP_IN_RAM:
                    self._cached_model = model
                    self._cached_config = model_config
                model = model.to(device)
                if use_half:
                    model = model.to(torch.float16)

            sample_rate = model_config["sample_rate"]
            sample_size = model_config["sample_size"]
            model.eval().requires_grad_(False)

            final_prompt = prompt.strip()
            if key:
                final_prompt += f", {key}"
            if bpm:
                final_prompt += f", {bpm} BPM"

            sa3_steps = 8
            sa3_cfg = 1.0
            sa3_sampler = "pingpong"

            conditioning = [
                {
                    "prompt": final_prompt,
                    "seconds_total": float(duration),
                }
            ]

            print(f"📝 SA3 prompt: {final_prompt}")
            print(
                f"🛠️ SA3 config: steps={sa3_steps}, cfg={sa3_cfg}, "
                f"sampler={sa3_sampler}, duration={duration}s"
            )
            t0 = time.time()

            output = generate_diffusion_cond_inpaint(
                model,
                steps=sa3_steps,
                cfg_scale=sa3_cfg,
                conditioning=conditioning,
                sample_size=sample_size,
                sampler_type=sa3_sampler,
                device=device,
                seed=seed,
            )

            print(f"✅ SA3 done in {time.time() - t0:.2f}s")

            audio = self._finalize_audio(output, duration, sample_rate)
            return audio

        finally:
            if model is not None:
                try:
                    if SA3_KEEP_IN_RAM and self._cached_model is not None:
                        self._cached_model.to("cpu")
                    else:
                        model.to("cpu")
                        del model
                except Exception:
                    pass
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
