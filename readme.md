# 🦥 Unsloth Lab

A local web UI for fine-tuning, distilling, merging, and managing LLMs — powered by [Unsloth](https://github.com/unslothai/unsloth) and Flask. No cloud required, runs entirely on your own GPU.

---

## Requirements

- Python 3.10+
- NVIDIA GPU (8 GB VRAM minimum recommended)
- CUDA 12.x or 13.x

---

## Installation

### Quick install (recommended)

**Linux / macOS**
```bash
bash install.sh
```

**Windows**
```bat
install.bat
```

### Manual install

#### 1. Create a virtual environment
```bash
python3 -m venv ~/unsloth/unsloth
source ~/unsloth/unsloth/bin/activate
```

#### 2. Install PyTorch (match your CUDA version)
```bash
# CUDA 13.x
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# CUDA 12.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

#### 3. Install Unsloth
```bash
pip install unsloth
# Or with explicit CUDA/torch extras
pip install "unsloth[cu130-torch2100] @ git+https://github.com/unslothai/unsloth.git"
```

#### 4. Install remaining dependencies
```bash
pip install -r requirements.txt
```

#### 5. (Optional) Flash Attention 2
Significantly reduces VRAM usage during training. Install **after** PyTorch:
```bash
pip install flash-attn --no-build-isolation
```
If this fails to compile (common on limited-RAM machines), xformers is already included in `requirements.txt` as a drop-in alternative. Enable it with:
```bash
# Add to ~/.bashrc or ~/.zshrc
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

### Training

#### ⚡ Fine-Tune
Train any compatible safetensors model on a local or HuggingFace dataset using LoRA. Configurable LoRA rank, learning rate, batch size, sequence length, gradient accumulation, and more. Supports low-memory mode and speed mode.  
**Now supports MoE models** (Mixtral, Qwen2-MoE, DeepSeek-V2/V3, OLMoE, Jamba, PhiMoE, etc.) — expert projection layers are automatically detected and included in LoRA target modules.

#### 🤖 AutoTrain
One-click fine-tuning using curated preset configurations.

| Preset | Description |
|---|---|
| Tool Usage | Function/API calling in structured JSON |
| Reasoning / Chain-of-Thought | Step-by-step logical and mathematical reasoning |
| Image Recognition / VQA | Vision-language fine-tuning (requires a model with a vision encoder) |
| Multimodal Upgrade | Adds vision capability to any text-only model via a CLIP projection layer (LLaVA-style) |

#### 🎓 Distillation
Generate a synthetic training dataset from a teacher model and use it to fine-tune a student model.
- **Local teacher** — uses a GGUF model (llama-cpp-python) running on your machine
- **Cloud teacher** — uses an API (OpenAI, Google Gemini, Anthropic, Perplexity, OpenRouter, or any OpenAI-compatible endpoint)

#### 🎯 AutoDistill
Distillation using preconfigured expert teacher models auto-downloaded from HuggingFace.

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

#### 🔀 Merge
Merge two safetensors models using SLERP, TIES, DARE, or linear blending.

#### 🔓 Abliterate *(Decensor)*
Remove the refusal behaviour from any model without fine-tuning.

**How it works:**
1. Runs calibration prompts (harmful vs harmless) through the model and records the last-token hidden states
2. Computes the *refusal direction* — the unit vector pointing from harmless → harmful activations
3. Projects that direction out of every weight matrix: `W ← W − threshold × (W·d)·dᵀ`
4. Saves the result to `.outputs/` — the original model is unchanged

| Setting | Description |
|---|---|
| Threshold | How aggressively to project (5% = surgical, 50% = aggressive) |
| Calibration Prompts | Number of prompts per class (default 64) |
| Layer Filter | Optional — restrict to specific layer name substrings |

> ⚠️ Abliteration is irreversible on the saved copy. Always keep a backup. Thresholds above 30% may degrade model coherence.

---

### Data

#### ✍️ AI Dataset Gen
Generate a synthetic dataset from scratch using a local model or cloud API, given a topic and style.

#### 🌐 Web Dataset Gen
Build a training dataset grounded in real web content. Searches the web (DuckDuckGo, Brave, or SerpAPI), fetches pages, and uses a local or cloud AI to generate QA pairs from scraped content.

#### 📦 Datasets
Browse and manage locally saved datasets, download from HuggingFace, and delete from disk.

---

### Assets

#### 💬 Chat
Load any local model and chat with it directly in the browser. Includes VRAM estimation before loading.

#### 🧱 Create Blank Model
Initialise a randomly-weighted model from scratch with a chosen architecture (LLaMA, Mistral, Gemma, Phi, Qwen2, GPT-2, GPT-NeoX, Falcon — including MoE variants), configurable size, and vocabulary.

#### 📏 RoPE Scale
Extend a model's context window without retraining by patching `config.json` with a RoPE scaling factor. Supports `linear`, `dynamic`, `yarn`, and `llama3` types.

#### 📦 Models
Browse and manage locally saved models, download from HuggingFace, import GGUF files, and delete from disk.

#### 📦 Outputs
Browse trained model outputs and exports.

#### 📈 Test / Evaluate
Run a test prompt set against a loaded model and score responses.

#### 📤 GGUF Export
Convert a trained model to GGUF format with multiple quantization types (Q4_K_M, Q5_K_M, Q8_0, F16, etc.) for use with llama.cpp, Ollama, LM Studio, etc.

#### 📱 Mobile Export
Export for on-device inference (MLC-LLM / ExecuTorch).

#### 🔧 Optimize
Post-training optimization pass over a saved model.

#### ✂️ Prune
Reduce model size through three methods:
- **Magnitude pruning** — zeros out the smallest weights across linear layers
- **Attention head pruning** — prunes entire attention head projection rows
- **Concept erasure** — removes weights associated with specific topics, guided by a local or cloud AI advisor

---

### System

#### ⚙️ Performance
Configure system-wide performance settings (tf32, 8-bit optimizer, xformers, etc.).

---

## MoE Model Support

The following Mixture-of-Experts architectures are supported for loading, fine-tuning (LoRA), and abliteration:

| Architecture | `model_type` |
|---|---|
| MixtralForCausalLM | `mixtral` |
| Qwen2MoeForCausalLM | `qwen2_moe` |
| DeepseekV2ForCausalLM | `deepseek_v2` |
| DeepseekV3ForCausalLM | `deepseek_v3` |
| OlmoeForCausalLM | `olmoe` |
| JambaForCausalLM | `jamba` |
| ArcticForCausalLM | `arctic` |
| PhiMoEForCausalLM | `phimoe` |
| Granitemoe10bForCausalLM | `granitemoe` |

Expert projection weights (`w1`, `w2`, `w3`, `shared_expert_gate`) are automatically detected at load time and included in LoRA target modules — no manual configuration needed.

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

API keys can be saved per-provider via the 💾 Save button — stored in `.cloud_keys` in the data directory and auto-loaded on startup.

---

## HuggingFace Login

Log in via the HF button in the top bar to access gated models and datasets and to push trained models to the Hub. Your token is optionally saved to `.hf_token` in the data directory for auto-login on restart.

---

## Performance Tips

| Problem | Fix |
|---|---|
| CUDA OOM during training | Enable **Low-memory mode**, reduce sequence length or batch size |
| Slow training | Enable **Speed mode** (tf32 + 8-bit optimizer + packing) |
| FlexAttention OOM | Set `UNSLOTH_USE_XFORMERS=1` or install flash-attn |
| flash-attn compile OOM | Use `MAX_JOBS=1 pip install flash-attn --no-build-isolation`, or skip it and use xformers |
| CUDA version mismatch | Reinstall PyTorch with the correct `--index-url` for your CUDA version |
| MoE model won't load | Ensure config.json has a recognised `architectures` field; the app patches `model_type` automatically |

---

## Directory Structure

```
/mnt/f/unsloth/          ← BASE_DATA_DIR (configurable in main.py)
├── models/              ← downloaded / trained models
├── datasets/            ← local datasets
├── outputs/             ← fine-tuned models, GGUF exports, abliterated models
├── generated_datasets/  ← synthetic datasets from generator jobs
├── cloud_logs/          ← per-run JSONL logs of cloud API calls
├── .hf_token            ← saved HuggingFace token
├── .cloud_keys          ← saved cloud provider API keys
└── tuning_logs.json     ← history of completed training runs
```
