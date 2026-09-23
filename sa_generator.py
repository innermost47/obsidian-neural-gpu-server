import threading
from typing import Optional
import time
import json
import math
import torch
import gc
from huggingface_hub import hf_hub_download
from stable_audio_tools.inference.generation import generate_diffusion_cond
from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict
from base_generator import Generator
from settings import FORCED_75_STEPS_MODELS


class StableAudioGenerator(Generator):
    def __init__(
        self,
        repo_id: str,
        ckpt_filename: str,
        config_filename: str = "model_config.json",
        model_key: str = "",
    ):
        super().__init__()
        self.repo_id = repo_id
        self.ckpt_filename = ckpt_filename
        self.config_filename = config_filename
        self._lock = threading.Lock()
        self._generating = False
        self.model_key = model_key

    def generate(
        self, prompt: str, bpm: int, bars: int, key: Optional[str], seed: int
    ) -> tuple[bytes, int]:
        with self._lock:
            self._generating = True
            try:
                return self._generate(prompt, bpm, bars, key, seed)
            finally:
                self._generating = False

    def _generate(
        self, prompt: str, bpm: int, bars: int, key: Optional[str], seed: int
    ) -> bytes:

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = (
            torch.bfloat16
            if (device.type == "cuda" and torch.cuda.is_bf16_supported())
            else torch.float16
        )

        ckpt_path = hf_hub_download(repo_id=self.repo_id, filename=self.ckpt_filename)
        config_path = hf_hub_download(
            repo_id=self.repo_id, filename=self.config_filename
        )

        with open(config_path) as f:
            model_config = json.load(f)

        model = create_model_from_config(model_config)
        state_dict = load_ckpt_state_dict(ckpt_path)
        model.load_state_dict(state_dict)
        model.to(device).to(dtype).eval().requires_grad_(False)
        training_config = model_config.get("training", {})
        demo_config = training_config.get("demo", {})
        steps = (
            75
            if self.model_key in FORCED_75_STEPS_MODELS
            else demo_config.get("demo_steps", 75)
        )
        cfg_scales = demo_config.get("demo_cfg_scales", [7.0])
        cfg_scale = cfg_scales[-1] if isinstance(cfg_scales, list) else 7.0
        conditioning_duration = 30.0

        print(
            f"🛠️ Auto-Config for {self.ckpt_filename}: Steps={steps}, CFG={cfg_scale}, Cond_Duration={conditioning_duration}s"
        )

        try:
            sample_rate = model_config["sample_rate"]
            snapped_bpm = (
                self._nearest_supported_bpm(bpm)
                if self.model_key in FORCED_75_STEPS_MODELS
                else None
            )
            if snapped_bpm and snapped_bpm != bpm:
                print(
                    f"⚠️ BPM {bpm} → snapped to {snapped_bpm} (stretch managed by the central server)"
                )

            clip_seconds = (60.0 / snapped_bpm) * 4 * bars if snapped_bpm else 30
            seconds_int = int(math.ceil(clip_seconds))
            min_input = getattr(model, "min_input_length", None)
            target_samples = int(seconds_int * sample_rate)
            if (
                isinstance(min_input, int)
                and min_input > 0
                and target_samples % min_input != 0
            ):
                target_samples += min_input - (target_samples % min_input)

            parts = [prompt]
            if key:
                parts.append(key)
            parts.append(f"{bars} Bars")
            (
                parts.append(f"{snapped_bpm} BPM")
                if snapped_bpm
                else parts.append(f"{bpm} BPM")
            )
            full_prompt = ", ".join(parts)
            print(f"📝 Model prompt: {full_prompt}")

            conditioning = [
                {
                    "prompt": full_prompt,
                    "seconds_start": 0.0,
                    "seconds_total": conditioning_duration,
                }
            ]

            t0 = time.time()
            audio = generate_diffusion_cond(
                model,
                conditioning=conditioning,
                negative_conditioning=None,
                steps=steps,
                cfg_scale=cfg_scale,
                batch_size=1,
                sample_size=target_samples,
                sample_rate=sample_rate,
                seed=seed,
                device=device,
                sampler_type="dpmpp-3m-sde",
                sigma_min=0.03,
                sigma_max=500,
                scale_phi=0.0,
            )
            print(f"✅ {self.ckpt_filename} done in {time.time() - t0:.2f}s")

            audio = self._finalize_audio(audio, clip_seconds, sample_rate)
            return audio, snapped_bpm

        finally:
            try:
                model.to("cpu")
            except Exception:
                pass
            del model
            try:
                del state_dict
            except NameError:
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
