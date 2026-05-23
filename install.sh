#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Unsloth Fine-Tuning Lab — Installer
# ─────────────────────────────────────────────────────────────────────────────

VENV_DIR="$(pwd)/venv"

# ── Helper: clear pip cache ───────────────────────────────────────────────────
clear_pip_cache() {
    pip cache purge --quiet 2>/dev/null || true
    rm -rf "$HOME/.cache/pip" 2>/dev/null
    rm -rf /tmp/pip-* /tmp/easy_install-* /tmp/pip-build-* 2>/dev/null
}

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║    🦥  Unsloth Fine-Tuning Lab  — Installer      ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# ── Detect CUDA version ───────────────────────────────────────────────────────
detect_cuda_version() {
    if command -v nvcc &>/dev/null; then
        nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' | head -1
    elif command -v nvidia-smi &>/dev/null; then
        nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1
    else
        echo ""
    fi
}

# ── Detect GPU VRAM in GB ─────────────────────────────────────────────────────
detect_vram_gb() {
    if command -v nvidia-smi &>/dev/null; then
        nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null \
            | head -1 | awk '{printf "%d", $1/1024}'
    else
        echo "0"
    fi
}

CUDA_VERSION=$(detect_cuda_version)
VRAM_GB=$(detect_vram_gb)

if [ -z "$CUDA_VERSION" ]; then
    echo "⚠️  No CUDA detected — installing CPU-only builds."
    echo "   (Training will be very slow without a GPU.)"
    CUDA_MAJOR=""
    CUDA_MINOR=""
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
    UNSLOTH_EXTRA=""
    LLAMA_CMAKE=""
else
    CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
    CUDA_MINOR=$(echo "$CUDA_VERSION" | cut -d. -f2)
    echo "✅ Detected CUDA $CUDA_VERSION  |  GPU VRAM: ${VRAM_GB}GB"

    if   [ "$CUDA_MAJOR" -ge 13 ]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu130"
        UNSLOTH_EXTRA="cu130-torch2100"
    elif [ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -ge 8 ]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu128"
        UNSLOTH_EXTRA="cu128-torch250"
    elif [ "$CUDA_MAJOR" -eq 12 ]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu124"
        UNSLOTH_EXTRA="cu124-torch260"
    elif [ "$CUDA_MAJOR" -eq 11 ]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu118"
        UNSLOTH_EXTRA="cu118-torch220"
    else
        echo "⚠️  CUDA $CUDA_VERSION is too old — falling back to CPU."
        CUDA_MAJOR=""
        TORCH_INDEX="https://download.pytorch.org/whl/cpu"
        UNSLOTH_EXTRA=""
    fi
    LLAMA_CMAKE="-DGGML_CUDA=on"
fi

echo ""

# ── Create / reuse venv ───────────────────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "📦 Creating virtual environment at $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
    if [ $? -ne 0 ]; then
        echo "❌ Failed to create venv. Is python3-venv installed?"
        echo "   Try: sudo apt install python3-venv"
        exit 1
    fi
else
    echo "♻️  Reusing existing venv at $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
pip install --upgrade pip --quiet --no-cache-dir

# ── Step 1: PyTorch ───────────────────────────────────────────────────────────
echo ""
echo "━━━ Step 1/5: PyTorch ━━━"
pip install torch torchvision torchaudio --index-url "$TORCH_INDEX" --no-cache-dir
if [ $? -ne 0 ]; then
    echo "❌ PyTorch install failed. Aborting."
    exit 1
fi
echo "🧹 Clearing pip cache after Step 1..."
clear_pip_cache

# ── Step 2: Unsloth ───────────────────────────────────────────────────────────
echo ""
echo "━━━ Step 2/5: Unsloth ━━━"
if [ -n "$UNSLOTH_EXTRA" ]; then
    pip install "unsloth[$UNSLOTH_EXTRA] @ git+https://github.com/unslothai/unsloth.git" --no-cache-dir
    if [ $? -ne 0 ]; then
        echo "⚠️  Versioned Unsloth install failed — trying generic install..."
        pip install unsloth --no-cache-dir
    fi
else
    pip install unsloth --no-cache-dir
fi
if [ $? -ne 0 ]; then
    echo "❌ Unsloth install failed. Aborting."
    exit 1
fi
echo "🧹 Clearing pip cache after Step 2..."
clear_pip_cache

# ── Step 3: llama-cpp-python ─────────────────────────────────────────────────
echo ""
echo "━━━ Step 3/5: llama-cpp-python ━━━"
if [ -n "$LLAMA_CMAKE" ]; then
    CMAKE_ARGS="$LLAMA_CMAKE" pip install llama-cpp-python --no-cache-dir
    if [ $? -ne 0 ]; then
        echo "⚠️  CUDA build failed — falling back to CPU-only llama-cpp-python"
        pip install llama-cpp-python --no-cache-dir
        if [ $? -ne 0 ]; then
            echo "⚠️  llama-cpp-python install failed — local GGUF inference won't work."
        fi
    fi
else
    pip install llama-cpp-python --no-cache-dir
fi
echo "🧹 Clearing pip cache after Step 3..."
clear_pip_cache

# ── Step 4: Flash Attention 2 (optional) ─────────────────────────────────────
echo ""
echo "━━━ Step 4/5: Flash Attention 2 (optional) ━━━"

SKIP_FLASH=0
if [ -z "$CUDA_MAJOR" ]; then
    echo "   Skipping — no CUDA."
    SKIP_FLASH=1
elif [ "$VRAM_GB" -le 12 ] 2>/dev/null; then
    echo "   Skipping — GPU has ${VRAM_GB}GB VRAM (≤12GB). Compiling flash-attn would OOM."
    echo "   xformers (installed in Step 5) will be used instead."
    SKIP_FLASH=1
fi

if [ "$SKIP_FLASH" -eq 0 ]; then
    pip install wheel ninja --quiet --no-cache-dir
    echo "   Building flash-attn with MAX_JOBS=2 (this can take 10-30 min)..."
    (MAX_JOBS=2 pip install flash-attn --no-build-isolation --no-cache-dir)
    if [ $? -ne 0 ]; then
        echo "⚠️  flash-attn build failed — skipping. xformers will be used instead."
    else
        echo "✅ flash-attn installed successfully."
    fi
fi
echo "🧹 Clearing pip cache after Step 4..."
clear_pip_cache
rm -rf /tmp/torch_extensions /tmp/llama-* /tmp/flash-attn-* 2>/dev/null

# ── Step 5: Everything else ───────────────────────────────────────────────────
echo ""
echo "━━━ Step 5/5: Remaining dependencies ━━━"
pip install -r requirements.txt --no-cache-dir
if [ $? -ne 0 ]; then
    echo "❌ requirements.txt install failed."
    exit 1
fi
echo "🧹 Clearing pip cache after Step 5..."
clear_pip_cache

# ── Config setup ─────────────────────────────────────────────────────────────
if [ ! -f "config.json" ]; then
    cp config.example.json config.json
    echo ""
    echo "📝 Created config.json — edit it to set your data directory and preferences."
fi

# ── Final cleanup ─────────────────────────────────────────────────────────────
echo ""
echo "🧹 Final cleanup of build artifacts..."
rm -rf build/ dist/ *.egg-info 2>/dev/null
echo "✅ Cleanup complete."

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  ✅  Installation complete!                       ║"
echo "║                                                  ║"
echo "║  Activate venv:  source venv/bin/activate        ║"
echo "║  Run the app:    python3 main.py                 ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""