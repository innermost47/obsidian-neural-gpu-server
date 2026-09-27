#!/usr/bin/env bash
# ===========================================================================
#  OBSIDIAN Neural GPU Server - Linux (NVIDIA CUDA) and macOS (Apple Silicon) setup
# ===========================================================================
set -euo pipefail
cd "$(dirname "$0")"

VENV_DIR=".venv"
TORCH_INDEX="https://download.pytorch.org/whl/cu126"
export PYTHONUNBUFFERED=1

# Hugging Face cache: defaults to ~/.cache/huggingface
# Uncomment to keep models inside the project folder:
# export HF_HOME="$(pwd)/hf_cache"

# Linux only, optional: URL or path of a flash-attn wheel matching
# torch 2.7 / CUDA 12.6 / Python 3.10. Leave empty to skip flash-attn.
FLASH_ATTN_WHL=""

on_error() {
    echo
    echo "[FAILED] Setup stopped on an error (see above)."
}
trap on_error ERR

# ---------------------------------------------------------------------------
#  Checks
# ---------------------------------------------------------------------------
echo
echo "=== Checks ==="

OS_NAME="$(uname -s)"
case "$OS_NAME" in
    Linux)
        PLATFORM="linux"
        ;;
    Darwin)
        PLATFORM="macos"
        if [[ "$(uname -m)" != "arm64" ]]; then
            echo "[ERROR] Intel Macs are not supported: an Apple Silicon (M1 or later) Mac is required."
            exit 1
        fi
        ;;
    *)
        echo "[ERROR] Unsupported system: $OS_NAME. On Windows, use install.bat."
        exit 1
        ;;
esac
echo "[OK] Platform: $PLATFORM"

if ! command -v python3.10 >/dev/null 2>&1; then
    echo "[ERROR] python3.10 not found."
    echo "        Install Python 3.10 (with the venv module) and run again."
    if [[ "$PLATFORM" == "macos" ]]; then
        echo "        macOS (Homebrew): brew install python@3.10"
    else
        echo "        Debian/Ubuntu: sudo apt install python3.10 python3.10-venv"
    fi
    exit 1
fi

if [[ "$PLATFORM" == "macos" ]]; then
    if [[ "$(python3.10 -c 'import platform; print(platform.machine())')" != "arm64" ]]; then
        echo "[ERROR] python3.10 is an x86_64 (Rosetta) build: the Apple GPU (MPS) needs an arm64 Python."
        echo "        Install the arm64 build (e.g. Homebrew in /opt/homebrew) and run again."
        exit 1
    fi
elif ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[ERROR] nvidia-smi not found: NVIDIA driver missing or not installed."
    exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "[WARNING] ffmpeg not found in PATH. Some audio conversions may fail."
    if [[ "$PLATFORM" == "macos" ]]; then
        echo "          macOS (Homebrew): brew install ffmpeg"
    else
        echo "          Debian/Ubuntu: sudo apt install ffmpeg"
    fi
fi

# HF_TOKEN lookup order: environment variable, then .env, then prompt.
if [[ -z "${HF_TOKEN:-}" && -f ".env" ]]; then
    HF_TOKEN="$(grep -E '^[[:space:]]*HF_TOKEN[[:space:]]*=' .env | tail -n 1 \
        | cut -d '=' -f 2- | tr -d '\r' | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' \
        -e 's/^"\(.*\)"$/\1/' -e "s/^'\(.*\)'$/\1/")" || true
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "[INFO] HF_TOKEN not found in environment or .env file."
    read -r -s -p "Hugging Face token: " HF_TOKEN
    echo
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "[ERROR] HF_TOKEN is required to download the models."
    echo "        Add a line HF_TOKEN=your_token to the .env file and run again."
    exit 1
fi
# huggingface_hub reads HF_TOKEN directly from the environment.
export HF_TOKEN
echo "[OK] Hugging Face token loaded."

# ---------------------------------------------------------------------------
#  Virtual environment
# ---------------------------------------------------------------------------
echo
echo "=== Virtual environment ==="

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    python3.10 -m venv "$VENV_DIR"
fi
PY="$VENV_DIR/bin/python"

"$PY" -m pip install --upgrade pip packaging setuptools wheel

# ---------------------------------------------------------------------------
#  Dependencies
# ---------------------------------------------------------------------------
echo
if [[ "$PLATFORM" == "macos" ]]; then
    echo "=== PyTorch (Apple Silicon / MPS) ==="
    # The default PyPI wheels for macOS arm64 include MPS support.
    "$PY" -m pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1
else
    echo "=== PyTorch CUDA 12.6 ==="
    "$PY" -m pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url "$TORCH_INDEX"
fi

echo
echo "=== requirements.txt ==="
if [[ "$PLATFORM" == "macos" ]]; then
    # bitsandbytes 8-bit loading is only used on CUDA: skip it on macOS.
    MAC_REQUIREMENTS="$(mktemp)"
    grep -v -i -E '^[[:space:]]*bitsandbytes' requirements.txt > "$MAC_REQUIREMENTS"
    "$PY" -m pip install --no-cache-dir -r "$MAC_REQUIREMENTS"
    rm -f "$MAC_REQUIREMENTS"
else
    "$PY" -m pip install --no-cache-dir -r requirements.txt
fi

echo
echo "=== stable-audio-tools and extras ==="
"$PY" -m pip install --no-cache-dir stable-audio-tools einops safetensors pytorch_lightning

if [[ "$PLATFORM" == "macos" ]]; then
    echo "[INFO] flash-attn skipped (not available on macOS)."
elif [[ -n "$FLASH_ATTN_WHL" ]]; then
    echo
    echo "=== flash-attn ==="
    "$PY" -m pip install --no-cache-dir --no-deps "$FLASH_ATTN_WHL"
else
    echo "[INFO] flash-attn skipped (FLASH_ATTN_WHL is empty)."
fi

# Check that PyTorch sees the GPU
if [[ "$PLATFORM" == "macos" ]]; then
    "$PY" -c "import torch, sys; ok = torch.backends.mps.is_available(); print('MPS OK:', 'Apple Silicon GPU' if ok else 'NO'); sys.exit(0 if ok else 1)"
else
    "$PY" -c "import torch, sys; ok = torch.cuda.is_available(); print('CUDA OK:', torch.cuda.get_device_name(0) if ok else 'NO'); sys.exit(0 if ok else 1)"
fi

# ---------------------------------------------------------------------------
#  Model pre-download
# ---------------------------------------------------------------------------
echo
echo "=== Models ==="

"$PY" -c "from diffusers import StableAudioPipeline; StableAudioPipeline.from_pretrained('stabilityai/stable-audio-open-1.0'); print('SAO 1.0 cached')"

"$PY" -c "from huggingface_hub import hf_hub_download as d; d(repo_id='RoyalCities/Foundation-1', filename='Foundation_1.safetensors'); d(repo_id='RoyalCities/Foundation-1', filename='model_config.json'); print('Foundation-1 cached')"

for model_name in audialab-edm-elements rc-infinite-pianos rc-vocal-textures sao-instrumental stablebeat gluten-v1; do
    "$PY" -c "from huggingface_hub import hf_hub_download as d; r='innermost47/obsidian-neural-models'; d(repo_id=r, filename='${model_name}.safetensors'); d(repo_id=r, filename='${model_name}_model_config.json'); print('${model_name} cached')"
done

"$PY" -c "from huggingface_hub import hf_hub_download as d; r='stabilityai/stable-audio-3-medium'; d(repo_id=r, filename='model_config.json'); d(repo_id=r, filename='model.safetensors'); print('SA3 Medium cached')"

# ---------------------------------------------------------------------------
trap - ERR
echo
echo "=== Setup complete ==="
echo "Start the server with: $VENV_DIR/bin/python main.py"