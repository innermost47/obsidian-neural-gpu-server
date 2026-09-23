@echo off
rem ===========================================================================
rem  OBSIDIAN Neural GPU Server - Windows setup
rem ===========================================================================
setlocal EnableExtensions
cd /d "%~dp0"

set "VENV_DIR=.venv"
set "TORCH_INDEX=https://download.pytorch.org/whl/cu126"
set "PYTHONUNBUFFERED=1"

rem Hugging Face cache: defaults to %USERPROFILE%\.cache\huggingface
rem Uncomment to keep models inside the project folder:
rem set "HF_HOME=%~dp0hf_cache"

rem Optional: URL or path of a Windows flash-attn wheel matching
rem torch 2.7 / CUDA 12.6 / Python 3.10. Leave empty to skip flash-attn.
set "FLASH_ATTN_WHL="

rem ---------------------------------------------------------------------------
rem  Checks
rem ---------------------------------------------------------------------------
echo.
echo === Checks ===

py -3.10 --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python 3.10 not found via the "py" launcher.
    echo         Install Python 3.10 from python.org and run again.
    goto :error
)

where nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo [ERROR] nvidia-smi not found: NVIDIA driver missing or not installed.
    goto :error
)

where ffmpeg >nul 2>&1
if errorlevel 1 (
    echo [WARNING] ffmpeg not found in PATH. Some audio conversions may fail.
    echo           Recommended install: ffmpeg.org
)

rem HF_TOKEN lookup order: environment variable, then .env, then prompt.
if not defined HF_TOKEN if exist ".env" (
    for /f "usebackq eol=# tokens=1,* delims==" %%A in (".env") do (
        if /i "%%A"=="HF_TOKEN" set "HF_TOKEN=%%~B"
    )
)
if not defined HF_TOKEN (
    echo [INFO] HF_TOKEN not found in environment or .env file.
    set /p "HF_TOKEN=Hugging Face token: "
)
if not defined HF_TOKEN (
    echo [ERROR] HF_TOKEN is required to download the models.
    echo         Add a line HF_TOKEN=your_token to the .env file and run again.
    goto :error
)
echo [OK] Hugging Face token loaded.
rem huggingface_hub reads HF_TOKEN directly from the environment.

rem ---------------------------------------------------------------------------
rem  Virtual environment
rem ---------------------------------------------------------------------------
echo.
echo === Virtual environment ===

if not exist "%VENV_DIR%\Scripts\python.exe" (
    py -3.10 -m venv "%VENV_DIR%" || goto :error
)
set "PY=%VENV_DIR%\Scripts\python.exe"

"%PY%" -m pip install --upgrade pip packaging setuptools wheel || goto :error

rem ---------------------------------------------------------------------------
rem  Dependencies
rem ---------------------------------------------------------------------------
echo.
echo === PyTorch CUDA 12.6 ===
"%PY%" -m pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url %TORCH_INDEX% || goto :error

echo.
echo === requirements.txt ===
"%PY%" -m pip install --no-cache-dir -r requirements.txt || goto :error

echo.
echo === stable-audio-tools and extras ===
"%PY%" -m pip install --no-cache-dir stable-audio-tools einops safetensors pytorch_lightning || goto :error

if defined FLASH_ATTN_WHL (
    echo.
    echo === flash-attn ===
    "%PY%" -m pip install --no-cache-dir --no-deps "%FLASH_ATTN_WHL%" || goto :error
) else (
    echo [INFO] flash-attn skipped ^(FLASH_ATTN_WHL is empty^).
)

rem Check that PyTorch sees CUDA
"%PY%" -c "import torch, sys; ok = torch.cuda.is_available(); print('CUDA OK:', torch.cuda.get_device_name(0) if ok else 'NO'); sys.exit(0 if ok else 1)" || goto :error

rem ---------------------------------------------------------------------------
rem  Model pre-download
rem ---------------------------------------------------------------------------
echo.
echo === Models ===

"%PY%" -c "from diffusers import StableAudioPipeline; StableAudioPipeline.from_pretrained('stabilityai/stable-audio-open-1.0'); print('SAO 1.0 cached')" || goto :error

"%PY%" -c "from huggingface_hub import hf_hub_download as d; d(repo_id='RoyalCities/Foundation-1', filename='Foundation_1.safetensors'); d(repo_id='RoyalCities/Foundation-1', filename='model_config.json'); print('Foundation-1 cached')" || goto :error

for %%M in (audialab-edm-elements rc-infinite-pianos rc-vocal-textures sao-instrumental stablebeat gluten-v1) do (
    "%PY%" -c "from huggingface_hub import hf_hub_download as d; r='innermost47/obsidian-neural-models'; d(repo_id=r, filename='%%M.safetensors'); d(repo_id=r, filename='%%M_model_config.json'); print('%%M cached')" || goto :error
)

"%PY%" -c "from huggingface_hub import hf_hub_download as d; r='stabilityai/stable-audio-3-medium'; d(repo_id=r, filename='model_config.json'); d(repo_id=r, filename='model.safetensors'); print('SA3 Medium cached')" || goto :error

rem ---------------------------------------------------------------------------
echo.
echo === Setup complete ===
echo Start the server with: %VENV_DIR%\Scripts\python.exe main.py
pause
exit /b 0

:error
echo.
echo [FAILED] Setup stopped on an error (see above).
pause
exit /b 1