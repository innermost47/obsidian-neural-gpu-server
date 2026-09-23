import os
import sys


def _bootstrap_cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="OBSIDIAN Neural GPU Server")
    parser.add_argument("--port", type=int, default=0, help="Port (overrides .env)")
    parser.add_argument("--host", default="", help="Host (overrides .env)")
    args, _ = parser.parse_known_args()

    if args.port:
        os.environ["PORT"] = str(args.port)
    if args.host:
        os.environ["HOST"] = args.host


if __name__ == "__main__":
    _bootstrap_cli()

import asyncio
from contextlib import asynccontextmanager
from typing import Any, Callable, Optional
import secrets
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse, Response
from stable_audio_tools import get_pretrained_model
from audio_generator import AudioGenerator
from models import AudioProcessRequest
from sa3_generator import StableAudio3Generator
from sa_generator import StableAudioGenerator
from settings import (
    DEFAULT_BARS,
    DEFAULT_SA3_DURATION,
    HOST,
    MAX_DURATION,
    MIN_DURATION,
    MODEL_KEY,
    PORT,
    SA3_KEEP_IN_RAM,
    STABLE_AUDIO_3_MODELS,
    STABLE_AUDIO_MODELS,
    TARGET_SAMPLE_RATE,
    MAX_SEED,
)

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


_vram_lock = asyncio.Lock()

generator: Optional[AudioGenerator] = None
stable_audio_generators: dict[str, StableAudioGenerator] = {}
stable_audio_3_generators: dict[str, StableAudio3Generator] = {}


def _load_models() -> None:
    global generator

    if generator is None:
        generator = AudioGenerator(model_key=MODEL_KEY)

    for model_key, (repo_id, ckpt, config) in STABLE_AUDIO_MODELS.items():
        stable_audio_generators[model_key] = StableAudioGenerator(
            repo_id, ckpt, config, model_key=model_key
        )
        print(f"  {model_key} : {repo_id}/{ckpt}")

    for model_key, repo_id in STABLE_AUDIO_3_MODELS.items():
        stable_audio_3_generators[model_key] = StableAudio3Generator(
            repo_id, model_key=model_key
        )
        print(f"  {model_key} : {repo_id} (SA3)")

    if SA3_KEEP_IN_RAM:
        print("🔥 Pre-loading SA3 into RAM (SA3_KEEP_IN_RAM=true)...")
        for sa3 in stable_audio_3_generators.values():
            sa3._cached_model, sa3._cached_config = get_pretrained_model(sa3.repo_id)
        print("✅ SA3 ready in RAM")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_models()
    yield


app = FastAPI(
    title="OBSIDIAN Neural GPU Server",
    description="GPU inference server for the OBSIDIAN Neural VST",
    version="1.0.0",
    lifespan=lifespan,
)


def _audio_response(wav_bytes: bytes, extra_headers: dict[str, str]) -> Response:
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={**extra_headers},
    )


def _new_seed() -> int:
    return secrets.randbelow(MAX_SEED + 1)


def _clamp_duration(requested: Optional[float]) -> float:
    return max(MIN_DURATION, min(MAX_DURATION, requested or DEFAULT_SA3_DURATION))


def _get_backend(registry: dict[str, Any], model: str) -> Any:
    backend = registry.get(model)
    if backend is None:
        raise HTTPException(status_code=503, detail=f"Model {model} not available")
    return backend


async def _run_generation(
    backend: Any, error_prefix: str, fn: Callable[..., Any], *args: Any
) -> Any:
    if backend._generating:
        raise HTTPException(
            status_code=503, detail="Already generating — try again later"
        )

    loop = asyncio.get_running_loop()
    try:
        async with _vram_lock:
            return await loop.run_in_executor(None, fn, *args)
    except Exception as e:
        print(f"❌ {error_prefix}: {e}")
        raise HTTPException(status_code=500, detail=f"{error_prefix}: {e}") from e


async def _process_sa3(request: AudioProcessRequest, seed: int) -> Response:
    sa3 = _get_backend(stable_audio_3_generators, request.model)
    duration = _clamp_duration(request.duration)

    wav_bytes = await _run_generation(
        sa3,
        "SA3 generation failed",
        sa3.generate,
        request.prompt,
        duration,
        seed,
        request.bpm,
        request.key,
    )
    return _audio_response(
        wav_bytes,
        {
            "X-Model": request.model,
            "X-Duration": str(duration),
            "X-Sample-Rate": str(TARGET_SAMPLE_RATE),
            "X-Seed": str(seed),
        },
    )


async def _process_stable_audio(request: AudioProcessRequest, seed: int) -> Response:
    sag = _get_backend(stable_audio_generators, request.model)
    if not request.bpm:
        raise HTTPException(status_code=422, detail="bpm is required for this model")

    wav_bytes, snapped_bpm = await _run_generation(
        sag,
        f"{request.model} generation failed",
        sag.generate,
        request.prompt,
        request.bpm,
        DEFAULT_BARS,
        request.key,
        seed,
    )
    return _audio_response(
        wav_bytes,
        {
            "X-Model": request.model,
            "X-BPM": str(request.bpm),
            "X-Snapped-BPM": str(snapped_bpm),
            "X-Bars": str(DEFAULT_BARS),
            "X-Seed": str(seed),
        },
    )


async def _process_default(request: AudioProcessRequest, seed: int) -> Response:
    duration = _clamp_duration(request.duration)

    wav_bytes = await _run_generation(
        generator,
        "Generation failed",
        generator.generate_with_seed,
        request.prompt,
        duration,
        seed,
        request.bpm,
        request.key,
    )
    return _audio_response(
        wav_bytes,
        {
            "X-Model": generator.model_key,
            "X-Duration": str(duration),
            "X-Sample-Rate": str(TARGET_SAMPLE_RATE),
            "X-Seed": str(seed),
        },
    )


@app.post("/process")
async def process(request: AudioProcessRequest) -> Response:
    seed = _new_seed()

    if request.model in STABLE_AUDIO_3_MODELS:
        return await _process_sa3(request, seed)
    if request.model in STABLE_AUDIO_MODELS:
        return await _process_stable_audio(request, seed)
    return await _process_default(request, seed)


@app.get("/", response_class=PlainTextResponse)
async def root() -> str:
    return "Service OK"


if __name__ == "__main__":

    if not torch.cuda.is_available():
        print("❌ No CUDA GPU detected. CPU mode is not allowed in the GPU Server")
        sys.exit(1)

    print(f"\n{'=' * 55}")
    print("  OBSIDIAN Neural GPU Server")
    print(f"  Host   : {HOST}:{PORT}")
    print(f"{'=' * 55}\n")

    uvicorn.run(app, host=HOST, port=PORT, log_level="info", backlog=2048)
