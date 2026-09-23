# OBSIDIAN Neural GPU Server

Self-hosted GPU server for the **9 GPU engines** of the [OBSIDIAN Neural](https://github.com/innermost47/ai-dj) plugin.

Run it on your own machine, point the plugin to it, and generate. No account, no API key, no cloud.

> 🪟 **Windows only for now.** Linux and macOS contributions are welcome.

---

## Requirements

- Windows 10 / 11
- An **NVIDIA GPU** with an up-to-date driver
- [Python 3.10](https://www.python.org/downloads/) (with the `py` launcher)
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

3. Run `install.bat`.

The script creates a local virtual environment (`.venv`), installs everything, checks that your GPU is detected, and downloads all the models. It can be run again safely: nothing already installed or downloaded is fetched twice.

---

## Run

```bat
.venv\Scripts\python.exe main.py
```

Optional flags:

```bat
.venv\Scripts\python.exe main.py --host 0.0.0.0 --port 8000
```

Use `--host 0.0.0.0` if the plugin runs on another computer of your network (and allow the port in the Windows firewall).

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

**`No CUDA GPU detected`**
Update your NVIDIA driver and check that `nvidia-smi` works in a terminal.

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
