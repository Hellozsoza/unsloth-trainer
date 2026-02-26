# 🦥 Unsloth Fine-Tuning Lab

A local web UI for fine-tuning, distilling, and managing LLMs — powered by [Unsloth](https://github.com/unslothai/unsloth) and Flask. No cloud required, runs entirely on your own GPU.

---

## Requirements

- Python 3.10+
- NVIDIA GPU (8 GB VRAM minimum recommended)
- CUDA 12.x or 13.x

---

## Installation

### 1. Create a virtual environment

```bash
python3 -m venv ~/unsloth/unsloth
source ~/unsloth/unsloth/bin/activate
```

### 2. Install PyTorch (match your CUDA version)

```bash
# CUDA 13.x (use cu130 index — highest available)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# CUDA 12.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 3. Install Unsloth

```bash
# Auto-detect versions (simplest)
pip install unsloth

# Or with explicit CUDA/torch extras
pip install "unsloth[cu130-torch2100] @ git+https://github.com/unslothai/unsloth.git"
```

### 4. Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 5. (Optional) Flash Attention 2

Significantly reduces VRAM usage during training. Install **after** PyTorch:

```bash
pip install flash-attn --no-build-isolation
```

If this fails to compile (common on limited-RAM machines), xformers is already included in `requirements.txt` and works as a drop-in alternative. Enable it with:

```bash
# Add to ~/.zshrc or ~/.bashrc
export UNSLOTH_USE_XFORMERS=1
```

---

## Running

```bash
python3 main.py
```

Then open **http://localhost:5000** in your browser.

> Data (models, datasets, outputs) is stored under `/mnt/f/unsloth/` by default. Edit `BASE_DATA_DIR` in `main.py` to change this.

---

## Features

### 🎯 Fine-Tune
Train any compatible safetensors model on a local or HuggingFace dataset using LoRA. Configurable LoRA rank, learning rate, batch size, sequence length, gradient accumulation, and more. Supports low-memory mode and speed mode.

### ⚡ AutoTrain
One-click fine-tuning using curated preset configurations. Available presets:

| Preset | Description |
|---|---|
| Tool Usage | Function/API calling in structured JSON |
| Reasoning / Chain-of-Thought | Step-by-step logical and mathematical reasoning |
| Image Recognition / VQA | Vision-language fine-tuning (requires a model with a vision encoder) |
| Multimodal Upgrade | Adds vision capability to any text-only model via a CLIP projection layer (LLaVA-style) |

### 🧬 Distillation
Generate a synthetic training dataset from a teacher model and use it to fine-tune a student model.

- **Local teacher** — uses a GGUF model (llama-cpp-python) running on your machine
- **Cloud teacher** — uses an API (OpenAI, Google Gemini, Anthropic, Perplexity, OpenRouter, or any OpenAI-compatible endpoint) to generate responses

### 🤖 AutoDistill
Distillation using preconfigured expert teacher models that are auto-downloaded from HuggingFace. Available presets:

| Preset | Teacher | Specialty |
|---|---|---|
| Coding | Qwen2.5-Coder-3B | Python, JS, SQL, algorithms |
| Math & Reasoning | Qwen2.5-Math-1.5B | Step-by-step problem solving |
| Instruction Following | Phi-3.5-mini | Structured task completion |
| Creative Writing | Llama-3.2-3B | Storytelling, prose |
| Chat & Conversation | Gemma-2-2B | Natural dialogue |
| Science & Knowledge | Qwen2.5-3B | Factual Q&A across domains |
| SQL & Data | Qwen2.5-Coder-1.5B | SQL queries and data analysis |
| Translation | TranslateGemma-4B | 55-language translation |
| Tool Calling | FunctionGemma-4B | JSON function call output |

### 📊 Dataset Generator
Generate a synthetic dataset from scratch using a local model or cloud API, given a topic and style.

### 🌐 Web Dataset Generator
Build a training dataset grounded in real web content. Searches the web (DuckDuckGo free, Brave, or SerpAPI), fetches pages, and uses a local or cloud AI to generate QA pairs from the scraped content.

### ✂️ Model Pruning
Reduce model size through three methods:

- **Magnitude pruning** — zeros out the smallest weights across linear layers
- **Attention head pruning** — prunes entire attention head projection rows
- **Concept erasure** — identifies and removes weights associated with specific topics/concepts, guided by a local or cloud AI advisor

### 🚀 Export
- **GGUF Export** — convert a trained model to GGUF format with multiple quantization types (Q4_K_M, Q5_K_M, Q8_0, F16, etc.) for use with llama.cpp, Ollama, LM Studio, etc.
- **Mobile Export** — export for on-device inference (MLC-LLM / ExecuTorch)

### 🔧 Optimize
Post-training optimization pass over a saved model.

### 📏 RoPE Scaling
Extend a model's context window without retraining by patching `config.json` with a RoPE scaling factor. Supports `linear`, `dynamic`, `yarn`, and `llama3` scaling types.

### 🧱 Create Blank Model
Initialize a randomly-weighted model from scratch with a chosen architecture (LLaMA, Mistral, Gemma, Phi, Qwen2, GPT-2, GPT-NeoX, Falcon), configurable size, and vocabulary.

### 💬 Chat
Load any local model and chat with it directly in the browser. Includes VRAM estimation before loading.

### 📈 Evaluate
Run a test prompt set against a loaded model and score responses.

### 📦 Models & Datasets
- Browse and manage locally saved models and datasets
- Download models and datasets directly from HuggingFace
- Import GGUF files
- Delete models/datasets from disk

---

## Cloud Providers

Distillation, dataset generation, and concept-erasure advisor support the following providers:

| Provider | Notes |
|---|---|
| OpenAI | Standard API |
| Google (Gemini) | Via `/v1beta/openai` compatible endpoint |
| Anthropic | Standard API |
| Perplexity | Agent API with built-in web search |
| OpenRouter | Access 100s of models via a single key |
| Custom | Any OpenAI-compatible endpoint |

API keys can be saved per-provider via the 💾 Save button next to each key field — they are stored in `.cloud_keys` in the data directory and auto-loaded on startup.

---

## HuggingFace Login

Log in via the Settings panel to access gated models and datasets and to push trained models to the Hub. Your token is optionally saved to `.hf_token` in the data directory for auto-login on restart.

---

## Performance Tips

| Problem | Fix |
|---|---|
| CUDA OOM during training | Enable **Low-memory mode**, reduce sequence length or batch size |
| Slow training | Enable **Speed mode** (tf32 + 8-bit optimizer + packing) |
| FlexAttention OOM | Set `UNSLOTH_USE_XFORMERS=1` or install flash-attn |
| flash-attn compile OOM | Use `MAX_JOBS=1 pip install flash-attn --no-build-isolation`, or skip it and use xformers |
| CUDA version mismatch | Reinstall PyTorch with the correct `--index-url` for your CUDA version |

---

## Directory Structure

```
/mnt/f/unsloth/          ← BASE_DATA_DIR (configurable in main.py)
├── models/              ← downloaded / trained models
├── datasets/            ← local datasets
├── outputs/             ← fine-tuned models, GGUF exports
├── generated_datasets/  ← synthetic datasets from generator jobs
├── cloud_logs/          ← per-run JSONL logs of cloud API calls
├── .hf_token            ← saved HuggingFace token
├── .cloud_keys          ← saved cloud provider API keys
└── tuning_logs.json     ← history of completed training runs
```