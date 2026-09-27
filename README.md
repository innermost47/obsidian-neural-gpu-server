# OBSIDIAN Neural GPU Server

Self-hosted GPU server for the **9 GPU engines** of the [OBSIDIAN Neural](https://github.com/innermost47/ai-dj) plugin.

Run it on your own machine, point the plugin to it, and generate. No account, no API key, no cloud.

> 🪟 **Windows** and 🐧 **Linux** with an NVIDIA GPU, 🍎 **macOS** on Apple Silicon (experimental).

---

## Requirements

- One of:
  - **Windows 10 / 11** or **Linux** with an **NVIDIA GPU** and an up-to-date driver
  - **macOS** on **Apple Silicon** (M1 or later), experimental. Intel Macs are not supported. The more unified memory the better: models are not quantized on macOS.
- [Python 3.10](https://www.python.org/downloads/): with the `py` launcher on Windows, as `python3.10` on Linux and macOS (macOS: `brew install python@3.10`)
- A [Hugging Face](https://huggingface.co) account and access token
- Plenty of free disk space for the models (Stable Audio 3 Medium alone is ~9 GB)

Before installing, accept the model licenses on Hugging Face (logged in with the same account as your token):

- [stabilityai/stable-audio-open-1.0](https://huggingface.co/stabilityai/stable-audio-open-1.0)
- [stabilityai/stable-audio-3-medium](https://huggingface.co/stabilityai/stable-audio-3-medium)

---

## Install

1. Clone this repository.
2. Create a `.env` file at the project root with your Hugging Face token:

   ```
   HF_TOKEN=hf_your_token_here
   ```

3. Run the installer:
   - Windows: `install.bat`
   - Linux / macOS: `chmod +x install.sh && ./install.sh`

The script creates a local virtual environment (`.venv`), installs everything, checks that your GPU is detected, and downloads all the models. It can be run again safely: nothing already installed or downloaded is fetched twice.

---

## Run

Windows:

```bat
.venv\Scripts\python.exe main.py
```

Linux / macOS:

```bash
.venv/bin/python main.py
```

Optional flags (same on every platform):

```bash
.venv/bin/python main.py --host 0.0.0.0 --port 8000
```

Use `--host 0.0.0.0` if the plugin runs on another computer of your network (and allow the port in your firewall).

---

## Connect the plugin

In OBSIDIAN Neural: **Settings → Server/API** → enter the server URL (e.g. `http://localhost:8000`), then pick one of the 9 engines.

---

## Engines

stable-audio-open-1.0 · Stable Audio 3 Medium · Foundation-1 · Audialab EDM Elements · RC Infinite Pianos · RC Vocal Textures · SAO Instrumental · StableBeaT · gluten_v1

---

## Troubleshooting

**`Le fichier de pagination est insuffisant` / `paging file is too small` (error 1455), or the server closes silently while loading a model**
Windows ran out of virtual memory. Increase the page file: `Win+R` → `sysdm.cpl` → Advanced → Performance → Settings → Advanced → Virtual memory → _System managed size_ (or at least 32 GB), then restart. Also check that your system drive has enough free space.

**`flash_attn not installed` / `No module named 'flash_attn'`**
Harmless. The server falls back to standard PyTorch attention.

**`No supported GPU detected`**
Windows / Linux: update your NVIDIA driver and check that `nvidia-smi` works in a terminal. macOS: make sure you are on an Apple Silicon Mac and that `python3.10` is an arm64 build, not an x86_64 one running under Rosetta.

**macOS: generations come out silent, as pure noise, or with `NaN` errors**
Models run in float16 on the Apple GPU by default to save memory. Add `MPS_DTYPE=float32` to your `.env` file and restart the server. This doubles memory usage.

**macOS: generation is slower than on an NVIDIA GPU**
Expected. A few operators are not implemented on the Apple GPU yet and fall back to the CPU (`PYTORCH_ENABLE_MPS_FALLBACK` is enabled by the server), and stable-audio-open-1.0 is loaded without 8-bit quantization, which is CUDA-only.

---

## API

`POST /process` with a JSON body:

```json
{
  "prompt": "deep house chord stabs",
  "model": "stable-audio-open-1.0",
  "bpm": 124,
  "duration": 10,
  "key": "A minor"
}
```

Returns a WAV file. The seed used for the generation is returned in the `X-Seed` header.

`GET /` returns `Service OK`.

---

## License

The AI models are downloaded separately and remain subject to their own terms, including the [Stability AI Community License](https://stability.ai/license).
