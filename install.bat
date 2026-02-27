@echo off
:: ─────────────────────────────────────────────────────────────────────────────
:: Unsloth Fine-Tuning Lab — Installer (Windows)
:: ─────────────────────────────────────────────────────────────────────────────
setlocal EnableDelayedExpansion

echo.
echo ╔══════════════════════════════════════════════════╗
echo ║    🦥  Unsloth Fine-Tuning Lab  — Installer      ║
echo ╚══════════════════════════════════════════════════╝
echo.

set VENV_DIR=%~dp0venv

:: ── Detect CUDA version ───────────────────────────────────────────────────────
set CUDA_MAJOR=0
for /f "tokens=*" %%i in ('nvidia-smi ^| findstr /C:"CUDA Version"') do (
    for /f "tokens=3" %%v in ("%%i") do (
        for /f "tokens=1 delims=." %%m in ("%%v") do set CUDA_MAJOR=%%m
    )
)

if "%CUDA_MAJOR%"=="0" (
    echo ⚠️  No CUDA detected — installing CPU-only builds.
    set TORCH_INDEX=https://download.pytorch.org/whl/cpu
    set UNSLOTH_EXTRA=
    set LLAMA_CUDA=0
) else if %CUDA_MAJOR% GEQ 13 (
    echo ✅ Detected CUDA %CUDA_MAJOR%.x
    set TORCH_INDEX=https://download.pytorch.org/whl/cu130
    set UNSLOTH_EXTRA=cu130-torch2100
    set LLAMA_CUDA=1
) else if %CUDA_MAJOR%==12 (
    echo ✅ Detected CUDA %CUDA_MAJOR%.x
    set TORCH_INDEX=https://download.pytorch.org/whl/cu128
    set UNSLOTH_EXTRA=cu128-torch250
    set LLAMA_CUDA=1
) else if %CUDA_MAJOR%==11 (
    echo ✅ Detected CUDA %CUDA_MAJOR%.x
    set TORCH_INDEX=https://download.pytorch.org/whl/cu118
    set UNSLOTH_EXTRA=cu118-torch220
    set LLAMA_CUDA=1
) else (
    echo ⚠️  CUDA too old — falling back to CPU.
    set TORCH_INDEX=https://download.pytorch.org/whl/cpu
    set UNSLOTH_EXTRA=
    set LLAMA_CUDA=0
)

:: ── Create / reuse venv ───────────────────────────────────────────────────────
if not exist "%VENV_DIR%" (
    echo 📦 Creating virtual environment...
    python -m venv "%VENV_DIR%"
) else (
    echo ♻️  Reusing existing venv
)

call "%VENV_DIR%\Scripts\activate.bat"
pip install --upgrade pip --quiet

:: ── Step 1: PyTorch ───────────────────────────────────────────────────────────
echo.
echo ━━━ Step 1/5: Installing PyTorch ━━━
pip install torch torchvision torchaudio --index-url %TORCH_INDEX%

:: ── Step 2: Unsloth ───────────────────────────────────────────────────────────
echo.
echo ━━━ Step 2/5: Installing Unsloth ━━━
if not "!UNSLOTH_EXTRA!"=="" (
    pip install "unsloth[!UNSLOTH_EXTRA!] @ git+https://github.com/unslothai/unsloth.git"
    if errorlevel 1 pip install unsloth
) else (
    pip install unsloth
)

:: ── Step 3: llama-cpp-python ─────────────────────────────────────────────────
echo.
echo ━━━ Step 3/5: Installing llama-cpp-python ━━━
if "%LLAMA_CUDA%"=="1" (
    set CMAKE_ARGS=-DGGML_CUDA=on
    pip install llama-cpp-python
    if errorlevel 1 (
        echo ⚠️  CUDA build failed — falling back to CPU build
        set CMAKE_ARGS=
        pip install llama-cpp-python
    )
) else (
    pip install llama-cpp-python
)

:: ── Step 4: Flash Attention 2 (optional) ─────────────────────────────────────
echo.
echo ━━━ Step 4/5: Flash Attention 2 (optional) ━━━
if "%LLAMA_CUDA%"=="1" (
    pip install wheel ninja
    set MAX_JOBS=2
    pip install flash-attn --no-build-isolation
    if errorlevel 1 echo ⚠️  flash-attn failed — skipping. xformers will be used instead.
) else (
    echo    Skipping (no CUDA^).
)

:: ── Step 5: Everything else ───────────────────────────────────────────────────
echo.
echo ━━━ Step 5/5: Installing remaining dependencies ━━━
pip install -r requirements.txt

:: ── Config setup ─────────────────────────────────────────────────────────────
if not exist "config.json" (
    copy config.example.json config.json >nul
    echo.
    echo 📝 Created config.json — edit it to set your data directory.
)

echo.
echo ╔══════════════════════════════════════════════════╗
echo ║  ✅  Installation complete!                       ║
echo ║                                                  ║
echo ║  Activate venv:  venv\Scripts\activate           ║
echo ║  Run the app:    python main.py                  ║
echo ╚══════════════════════════════════════════════════╝
echo.