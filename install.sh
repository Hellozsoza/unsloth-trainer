#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Unsloth Fine-Tuning Lab — Installer
# ─────────────────────────────────────────────────────────────────────────────
set -e

VENV_DIR="$(pwd)/venv"

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║    🦥  Unsloth Fine-Tuning Lab  — Installer      ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# ── Detect CUDA version ───────────────────────────────────────────────────────
detect_cuda() {
    if command -v nvcc &>/dev/null; then
        nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+' | head -1
    elif command -v nvidia-smi &>/dev/null; then
        nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1
    else
        echo ""
    fi
}

CUDA_VERSION=$(detect_cuda)

if [ -z "$CUDA_VERSION" ]; then
    echo "⚠️  No CUDA detected — installing CPU-only builds."
    echo "   (Training will be very slow without a GPU.)"
    CUDA_MAJOR=""
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
    UNSLOTH_EXTRA=""
    LLAMA_CMAKE=""
else
    CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
    CUDA_MINOR=$(echo "$CUDA_VERSION" | cut -d. -f2)
    echo "✅ Detected CUDA $CUDA_VERSION (major: $CUDA_MAJOR)"

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
        echo "⚠️  CUDA $CUDA_VERSION is older than 11 — falling back to CPU."
        TORCH_INDEX="https://download.pytorch.org/whl/cpu"
        UNSLOTH_EXTRA=""
        LLAMA_CMAKE=""
    fi
    LLAMA_CMAKE="DGGML_CUDA=on"
fi

echo ""

# ── Create / reuse venv ───────────────────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "📦 Creating virtual environment at $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
else
    echo "♻️  Reusing existing venv at $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
pip install --upgrade pip --quiet

# ── Step 1: PyTorch ───────────────────────────────────────────────────────────
echo ""
echo "━━━ Step 1/5: Installing PyTorch ($TORCH_INDEX) ━━━"
pip install torch torchvision torchaudio --index-url "$TORCH_INDEX"

# ── Step 2: Unsloth ───────────────────────────────────────────────────────────
echo ""
echo "━━━ Step 2/5: Installing Unsloth ━━━"
if [ -n "$UNSLOTH_EXTRA" ]; then
    pip install "unsloth[$UNSLOTH_EXTRA] @ git+https://github.com/unslothai/unsloth.git" || \
    pip install unsloth
else
    pip install unsloth
fi

# ── Step 3: llama-cpp-python ─────────────────────────────────────────────────
echo ""
echo "━━━ Step 3/5: Installing llama-cpp-python ━━━"
if [ -n "$LLAMA_CMAKE" ]; then
    CMAKE_ARGS="-$LLAMA_CMAKE" pip install llama-cpp-python || {
        echo "⚠️  CUDA build failed — falling back to CPU build of llama-cpp-python"
        pip install llama-cpp-python
    }
else
    pip install llama-cpp-python
fi

# ── Step 4: Flash Attention 2 (optional) ─────────────────────────────────────
echo ""
echo "━━━ Step 4/5: Flash Attention 2 (optional) ━━━"
if [ -n "$CUDA_MAJOR" ]; then
    pip install wheel ninja
    MAX_JOBS=2 pip install flash-attn --no-build-isolation || \
        echo "⚠️  flash-attn failed to build — skipping. xformers will be used instead."
else
    echo "   Skipping (no CUDA)."
fi

# ── Step 5: Everything else ───────────────────────────────────────────────────
echo ""
echo "━━━ Step 5/5: Installing remaining dependencies ━━━"
pip install -r requirements.txt

# ── Config setup ─────────────────────────────────────────────────────────────
if [ ! -f "config.json" ]; then
    cp config.example.json config.json
    echo ""
    echo "📝 Created config.json from config.example.json"
    echo "   Edit it to set your data directory and preferences."
fi

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  ✅  Installation complete!                       ║"
echo "║                                                  ║"
echo "║  Activate venv:  source venv/bin/activate        ║"
echo "║  Run the app:    python3 main.py                 ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""