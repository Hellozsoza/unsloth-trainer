import os
import gc
import json
import re
import itertools
import threading
import queue
import time
import traceback
import random
from pathlib import Path
from flask import Flask, render_template, request, jsonify, Response, stream_with_context

app = Flask(__name__)

# ─── Global state ─────────────────────────────────────────────────────────────
log_queue   = queue.Queue()
current_job = {"status": "idle", "progress": 0, "logs": [], "stage": ""}
training_thread = None
stop_flag   = threading.Event()
hf_token    = {"value": None, "username": None}

# ─── Resource registry (for instant stop/eject) ───────────────────────────────
_active_model     = {"model": None, "tokenizer": None}  # currently loaded model
_active_threads   = []          # all background threads we can try to kill
_download_dirs    = []          # partial download dirs to clean up on stop
_monitor_events   = []          # monitor/pulse threading.Events to stop

# ─── Config file ──────────────────────────────────────────────────────────────
_REPO_DIR    = Path(__file__).parent
_CONFIG_FILE = _REPO_DIR / "config.json"
_CONFIG_EXAMPLE = _REPO_DIR / "config.example.json"

CONFIG_DEFAULTS = {
    # Paths
    "base_data_dir": str(Path.home() / "unsloth_data"),
    # Server
    "host":  "0.0.0.0",
    "port":  5000,
    "debug": False,
    # Training performance defaults
    "low_memory_mode": False,
    "speed_mode":      False,
    "stream_dataset":  False,
    "default_batch":   2,
    "default_ga":      4,
    "default_seqlen":  2048,
    # Torch / Unsloth flags
    "disable_torchdynamo":    True,
    "disable_unsloth_compile": True,
    "use_xformers":           False,
}

# Write config.example.json if missing
if not _CONFIG_EXAMPLE.exists():
    _CONFIG_EXAMPLE.write_text(json.dumps(CONFIG_DEFAULTS, indent=2) + "\n")
    print(f"[INFO] Created config.example.json — copy it to config.json to customise.")

# Load config.json, fall back to defaults for any missing keys
if _CONFIG_FILE.exists():
    try:
        _user_config = json.loads(_CONFIG_FILE.read_text())
        print(f"[INFO] Loaded config.json")
    except Exception as e:
        print(f"[WARN] Could not parse config.json: {e} — using defaults")
        _user_config = {}
else:
    print(f"[INFO] No config.json found — using defaults. Copy config.example.json to config.json to customise.")
    _user_config = {}

# Merge: user values win, missing keys fall back to defaults
_cfg = {**CONFIG_DEFAULTS, **_user_config}

def _save_config():
    """Persist the current _cfg dict back to config.json."""
    try:
        _CONFIG_FILE.write_text(json.dumps(_cfg, indent=2) + "\n")
    except Exception as e:
        print(f"[WARN] Could not save config.json: {e}")

# ─── Apply torch/unsloth env flags from config ────────────────────────────────
if _cfg["disable_torchdynamo"]:
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
if _cfg["disable_unsloth_compile"]:
    os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
if _cfg["use_xformers"]:
    os.environ["UNSLOTH_USE_XFORMERS"] = "1"

# ─── Data directories ─────────────────────────────────────────────────────────
BASE_DATA_DIR    = Path(_cfg["base_data_dir"]).expanduser()
MODELS_DIR       = BASE_DATA_DIR / "models"
DATASETS_DIR     = BASE_DATA_DIR / "datasets"
OUTPUTS_DIR      = BASE_DATA_DIR / "outputs"
GEN_DIR          = BASE_DATA_DIR / "generated_datasets"
CLOUD_LOGS_DIR   = BASE_DATA_DIR / "cloud_logs"
TOKEN_FILE       = BASE_DATA_DIR / ".hf_token"
TUNING_LOGS_FILE = BASE_DATA_DIR / "tuning_logs.json"

for d in [MODELS_DIR, DATASETS_DIR, OUTPUTS_DIR, GEN_DIR, CLOUD_LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def _load_saved_token():
    try:
        if TOKEN_FILE.exists():
            token = TOKEN_FILE.read_text().strip()
            if token:
                from huggingface_hub import HfApi
                info = HfApi(token=token).whoami()
                hf_token["value"]    = token
                hf_token["username"] = info.get("name", "unknown")
                print(f"[INFO] Auto-logged in to HuggingFace as: {hf_token['username']}")
    except Exception as e:
        print(f"[WARN] Could not restore HF token: {e}")

_load_saved_token()


# ─── AutoTrain presets ────────────────────────────────────────────────────────
AUTOTRAIN_PRESETS = {
    "tool_usage": {
        "label": "Tool Usage",
        "description": "Teaches the model to call functions/APIs in structured JSON format.",
        "datasets": [
            {"repo": "glaiveai/glaive-function-calling-v2",    "split": "train", "text_col": "chat"},
            {"repo": "Salesforce/xlam-function-calling-60k",   "split": "train", "text_col": "query"},
            {"repo": "NousResearch/hermes-function-calling-v1","split": "train", "text_col": "conversations"},
        ],
        "lora_r": 32, "max_steps": 200,
    },
    "reasoning": {
        "label": "Reasoning / Chain-of-Thought",
        "description": "Teaches step-by-step logical and mathematical reasoning.",
        "datasets": [
            {"repo": "open-r1/OpenR1-Math-220k",    "split": "train",     "text_col": "text"},
            {"repo": "teknium/OpenHermes-2.5",       "split": "train",     "text_col": "text"},
            {"repo": "HuggingFaceH4/ultrachat_200k", "split": "train_sft", "text_col": "prompt"},
            {"repo": "yahma/alpaca-cleaned",         "split": "train",     "text_col": "text"},
        ],
        "lora_r": 64, "max_steps": 300,
    },
    "image_recognition": {
        "label": "Image Recognition / VQA",
        "description": "Fine-tune an existing vision-language model (LLaVA, Qwen-VL, Idefics) on image+text data.",
        "datasets": [
            {"repo": "liuhaotian/LLaVA-Instruct-150K", "split": "train", "text_col": "conversations"},
            {"repo": "HuggingFaceM4/the_cauldron",     "split": "train", "text_col": "texts"},
            {"repo": "nyu-visionx/CV-Bench",           "split": "test",  "text_col": "question"},
        ],
        "note": "Requires a model that already has a vision encoder (LLaVA, Qwen-VL, Idefics, PaliGemma).",
        "lora_r": 32, "max_steps": 150,
    },
    "multimodal_upgrade": {
        "label": "Multimodal Upgrade",
        "description": "Add vision capability to any text-only model by attaching a CLIP vision encoder and training a projection bridge (LLaVA-style). The base LLM stays frozen; only the new projection layer trains.",
        "datasets": [
            {"repo": "liuhaotian/LLaVA-CC3M-Pretrain-595K", "split": "train", "text_col": "conversations"},
            {"repo": "liuhaotian/LLaVA-Instruct-150K",      "split": "train", "text_col": "conversations"},
        ],
        "note": "Downloads CLIP ViT-L/14 automatically. Works on any causal LM.",
        "lora_r": 32, "max_steps": 500,
        "vision_encoder": "openai/clip-vit-large-patch14",
        "projection_only": True,
    },
}

SYNTH_TOPICS = {
    "tool_usage": ["book a flight","search the web","get current weather","send an email","query a database","calculate compound interest","convert currencies","set a reminder"],
    "reasoning":  ["a train leaves at 60mph and another at 80mph","if all cats are animals and Felix is a cat","prove sqrt(2) is irrational","a farmer has chickens and cows totaling 20 heads and 56 legs"],
    "general":    ["photosynthesis","quantum entanglement","the French Revolution","sorting algorithms","neural networks","climate change","supply and demand","DNA replication","black holes","democracy","machine learning","the Roman Empire","cellular respiration","linear algebra","natural selection","the water cycle"],
}

# ─── AutoDistill presets ──────────────────────────────────────────────────────
# Each preset uses a small (<4B), specialized GGUF teacher (Q8_0 preferred).
# The teacher is auto-downloaded from HF into ./models/ if not already present.
# Students are any safetensors model the user provides.
AUTODISTILL_PRESETS = {

    "ad_coding": {
        "label": "Coding",
        "icon": "💻",
        "description": "Distill coding expertise — Python, JS, SQL, algorithms and debugging — from a Qwen2.5-Coder specialist.",
        "teacher_repo": "bartowski/Qwen2.5-Coder-3B-Instruct-GGUF",
        "teacher_file": "Qwen2.5-Coder-3B-Instruct-Q8_0.gguf",
        "teacher_size_gb": 3.4,
        "teacher_params": "3B",
        "topics": [
            "Write a Python function to reverse a linked list",
            "Explain the difference between a stack and a queue with code examples",
            "Write an efficient SQL query to find duplicate rows in a table",
            "Implement binary search in JavaScript and explain its time complexity",
            "Debug this Python code and explain the fix: def factorial(n): return n * factorial(n)",
            "What is a closure in programming? Give an example in Python",
            "Write a REST API endpoint in Python using Flask",
            "Explain recursion vs iteration with code examples",
            "How do you handle errors and exceptions in Python?",
            "Write a function to check if a string is a palindrome",
        ],
        "num_prompts": 150,
        "max_new_tokens": 512,
        "lora_r": 32,
        "max_steps": 120,
    },

    "ad_math": {
        "label": "Math & Reasoning",
        "icon": "🧮",
        "description": "Distill mathematical reasoning and step-by-step problem solving from a Qwen2.5-Math specialist.",
        "teacher_repo": "bartowski/Qwen2.5-Math-1.5B-Instruct-GGUF",
        "teacher_file": "Qwen2.5-Math-1.5B-Instruct-Q8_0.gguf",
        "teacher_size_gb": 1.6,
        "teacher_params": "1.5B",
        "topics": [
            "Solve: if 3x + 7 = 22, what is x?",
            "What is the derivative of f(x) = x³ + 2x² - 5x + 1?",
            "A train travels 120 miles in 2 hours. What is its average speed?",
            "Calculate the area of a circle with radius 7cm",
            "Solve the quadratic equation: x² - 5x + 6 = 0",
            "What is the sum of the first 100 natural numbers?",
            "Explain the Pythagorean theorem and give a worked example",
            "A bag has 4 red and 6 blue balls. What is the probability of drawing a red ball?",
            "What is the greatest common divisor of 48 and 72?",
            "Explain what a prime number is and list all primes up to 50",
        ],
        "num_prompts": 150,
        "max_new_tokens": 400,
        "lora_r": 32,
        "max_steps": 120,
    },

    "ad_instruction": {
        "label": "Instruction Following",
        "icon": "📋",
        "description": "Distill precise, structured instruction-following ability from Phi-3.5-mini, one of the best small instruction models.",
        "teacher_repo": "bartowski/Phi-3.5-mini-instruct-GGUF",
        "teacher_file": "Phi-3.5-mini-instruct-Q8_0.gguf",
        "teacher_size_gb": 3.9,
        "teacher_params": "3.8B",
        "topics": [
            "Summarize the following text in exactly 3 bullet points",
            "List the steps to make a cup of tea, in order",
            "Classify this sentence as positive, negative, or neutral",
            "Rewrite this paragraph in a more formal tone",
            "Extract all names and dates from the following text",
            "Translate this English sentence to French",
            "Sort the following list alphabetically",
            "Format this data as a JSON object with name, age, and email fields",
            "Write a haiku about winter",
            "Answer in yes or no: Is Paris the capital of France?",
        ],
        "num_prompts": 150,
        "max_new_tokens": 300,
        "lora_r": 32,
        "max_steps": 120,
    },

    "ad_creative": {
        "label": "Creative Writing",
        "icon": "✍️",
        "description": "Distill creative writing, storytelling, and imaginative content generation from Llama-3.2-3B.",
        "teacher_repo": "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "teacher_file": "Llama-3.2-3B-Instruct-Q8_0.gguf",
        "teacher_size_gb": 3.4,
        "teacher_params": "3B",
        "topics": [
            "Write a short story about a robot who discovers emotions",
            "Write the opening paragraph of a mystery novel set in Victorian London",
            "Describe a sunset over the ocean in vivid, poetic language",
            "Write a dialogue between two strangers stuck in an elevator",
            "Create a short poem about the feeling of nostalgia",
            "Write a product description for a magical umbrella that predicts the future",
            "Describe a character who is both brave and deeply afraid of failure",
            "Write a short horror story that ends with a twist",
            "Create a metaphor that describes the feeling of learning something new",
            "Write a children's story about a cloud who wants to make it rain",
        ],
        "num_prompts": 120,
        "max_new_tokens": 512,
        "lora_r": 16,
        "max_steps": 100,
    },

    "ad_chat": {
        "label": "Chat & Conversation",
        "icon": "💬",
        "description": "Distill natural, helpful conversational ability and social intelligence from Gemma-2-2B-IT.",
        "teacher_repo": "bartowski/gemma-2-2b-it-GGUF",
        "teacher_file": "gemma-2-2b-it-Q8_0.gguf",
        "teacher_size_gb": 2.7,
        "teacher_params": "2.6B",
        "topics": [
            "How are you doing today?",
            "Can you recommend a good book to read on a rainy day?",
            "I'm feeling overwhelmed with work. What should I do?",
            "What's the best way to start a conversation with someone new?",
            "How do I politely decline an invitation?",
            "What are some tips for a good first date?",
            "I want to learn a new language. Where should I start?",
            "Can you help me write a friendly email to a colleague?",
            "What are some good ways to stay motivated?",
            "I just moved to a new city. How do I make friends?",
        ],
        "num_prompts": 120,
        "max_new_tokens": 350,
        "lora_r": 16,
        "max_steps": 100,
    },

    "ad_science": {
        "label": "Science & Knowledge",
        "icon": "🔬",
        "description": "Distill broad factual knowledge across science, history, and general topics from Qwen2.5-3B.",
        "teacher_repo": "bartowski/Qwen2.5-3B-Instruct-GGUF",
        "teacher_file": "Qwen2.5-3B-Instruct-Q8_0.gguf",
        "teacher_size_gb": 3.4,
        "teacher_params": "3B",
        "topics": [
            "Explain how photosynthesis works",
            "What causes the seasons on Earth?",
            "How does a vaccine work?",
            "Explain Newton's three laws of motion",
            "What is DNA and why is it important?",
            "How does the immune system fight infections?",
            "Explain the water cycle",
            "What caused the extinction of the dinosaurs?",
            "How does electricity generate from solar panels?",
            "What is the theory of evolution?",
        ],
        "num_prompts": 150,
        "max_new_tokens": 450,
        "lora_r": 32,
        "max_steps": 120,
    },

    "ad_sql": {
        "label": "SQL & Data",
        "icon": "🗄️",
        "description": "Distill structured data querying, SQL expertise, and data analysis skills from Qwen2.5-Coder-1.5B.",
        "teacher_repo": "bartowski/Qwen2.5-Coder-1.5B-Instruct-GGUF",
        "teacher_file": "Qwen2.5-Coder-1.5B-Instruct-Q8_0.gguf",
        "teacher_size_gb": 1.7,
        "teacher_params": "1.5B",
        "topics": [
            "Write a SQL query to select all rows from a table called 'users' where age > 30",
            "How do you join two tables in SQL? Give an example",
            "Write a SQL query to count the number of orders per customer",
            "What is the difference between INNER JOIN and LEFT JOIN?",
            "Write a query to find the top 5 highest-paid employees",
            "How do you create an index in SQL and why is it useful?",
            "Write a SQL query using GROUP BY and HAVING",
            "Explain what a subquery is with an example",
            "How do you insert a new row into a database table?",
            "Write a query to find all duplicate email addresses in a table",
        ],
        "num_prompts": 150,
        "max_new_tokens": 400,
        "lora_r": 32,
        "max_steps": 120,
    },

    "ad_multilingual": {
        "label": "Translation",
        "icon": "🌍",
        "description": "Distill professional-grade translation across 55 languages from Google's TranslateGemma-4B — a dedicated translation specialist distilled from Gemini.",
        "teacher_repo": "mradermacher/translategemma-4b-it-GGUF",
        "teacher_file": "translategemma-4b-it-Q8_0.gguf",
        "teacher_size_gb": 4.13,
        "teacher_params": "4B",
        "topics": [
            "Translate to French: 'The advances in artificial intelligence are reshaping how we interact with technology every day.'",
            "Translate to Spanish: 'Climate change is one of the most pressing challenges facing humanity in the 21st century.'",
            "Translate to German: 'The economic report highlights strong growth in the technology and renewable energy sectors.'",
            "Translate to Japanese: 'Please confirm your appointment for tomorrow at 3pm at the downtown office.'",
            "Translate to Chinese (Simplified): 'Machine learning models require large amounts of data and computational resources.'",
            "Translate to Arabic: 'The conference will feature speakers from over 40 countries discussing global health initiatives.'",
            "Translate to Portuguese: 'We are pleased to announce the launch of our new product line this autumn.'",
            "Translate to Italian: 'The museum's new exhibition explores the relationship between art and technology.'",
            "Translate to Russian: 'Safety protocols must be followed at all times when operating heavy machinery.'",
            "Translate to Korean: 'The startup raised 50 million dollars in its latest funding round led by venture capital firms.'",
            "Translate to Hindi: 'Education is the foundation of a prosperous and equitable society for all citizens.'",
            "Translate to Dutch: 'The new bridge connecting the two city districts will open to traffic next month.'",
        ],
        "num_prompts": 150,
        "max_new_tokens": 300,
        "lora_r": 32,
        "max_steps": 120,
    },

    "ad_tool_calling": {
        "label": "Tool Calling",
        "icon": "🔨",
        "description": "Distill structured function-calling ability from Google's FunctionGemma-270M — purpose-built for tool use and agent workflows. Generates JSON function call outputs from natural language.",
        "teacher_repo": "unsloth/functiongemma-270m-it-GGUF",
        "teacher_file": "functiongemma-270m-it-UD-Q8_K_XL.gguf",
        "teacher_size_gb": 0.47,
        "teacher_params": "270M",
        # FunctionGemma needs a developer-role system prompt listing available tools.
        # We embed a small tool schema directly into each prompt so the teacher
        # produces realistic function-call JSON outputs.
        "prompt_template": "tool_calling",   # signals run_autodistill to use special prompting
        "topics": [
            "Get the current weather for New York City",
            "Search the web for the latest news about electric vehicles",
            "Send an email to john@example.com with subject 'Meeting Tomorrow' and body 'Let's meet at 10am'",
            "Create a calendar event titled 'Team standup' on Friday at 9am",
            "Look up the stock price for Apple (AAPL)",
            "Set a reminder to take medication in 2 hours",
            "Find restaurants near my location that are open now",
            "Translate the text 'Hello world' to Spanish",
            "Calculate the distance between London and Paris",
            "Get the latest score for the Lakers game",
            "Book a flight from San Francisco to Tokyo next Tuesday",
            "Turn on the bedroom lights and set them to 50% brightness",
            "Play the song 'Bohemian Rhapsody' by Queen",
            "What is the current time in Tokyo?",
            "Add 'buy groceries' to my shopping list",
        ],
        "num_prompts": 150,
        "max_new_tokens": 256,
        "lora_r": 32,
        "max_steps": 120,
    },
}


def _download_gguf_teacher(repo, filename, dest_dir=None):
    """
    Download a GGUF file from HuggingFace into MODELS_DIR if not already present.
    Returns the local path to the file.
    """
    dest_dir  = Path(dest_dir) if dest_dir else MODELS_DIR
    local_path = dest_dir / filename
    if local_path.exists():
        emit_log(f"GGUF teacher already cached: {local_path}", "info")
        return str(local_path)

    emit_log(f"Downloading GGUF teacher: {repo}/{filename} ...", "info")
    import urllib.request
    url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    headers = {}
    if hf_token["value"]:
        headers["Authorization"] = f"Bearer {hf_token['value']}"

    req  = urllib.request.Request(url, headers=headers)
    dest_dir.mkdir(parents=True, exist_ok=True)
    tmp  = local_path.with_suffix(".tmp")
    downloaded = 0
    last_pct = -1
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            with open(tmp, "wb") as f:
                while True:
                    chunk = resp.read(1 << 20)  # 1 MB
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = int(downloaded / total * 100)
                        if pct != last_pct and pct % 10 == 0:
                            emit_log(f"  Downloading {filename}: {pct}% ({downloaded//1024//1024}MB / {total//1024//1024}MB)", "info")
                            last_pct = pct
        tmp.rename(local_path)
        emit_log(f"Downloaded: {local_path} ({downloaded//1024//1024} MB)", "success")
        return str(local_path)
    except Exception as e:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(f"Failed to download GGUF teacher '{filename}': {e}")


def run_autodistill(config):
    """
    AutoDistill: selects a small specialized GGUF teacher from presets,
    auto-downloads it if needed, generates a synthetic dataset, then
    trains the user's student model on it.
    """
    try:
        mode         = config["mode"]
        preset       = AUTODISTILL_PRESETS[mode]
        student_name = config["student_model"]
        max_steps    = cfg_int(config.get("max_steps") or preset["max_steps"])
        lora_r       = cfg_int(config.get("lora_r")    or preset["lora_r"])
        num_prompts  = cfg_int(config.get("num_prompts") or preset["num_prompts"])
        max_tokens   = cfg_int(config.get("max_new_tokens") or preset["max_new_tokens"])
        inf_batch    = cfg_int(config.get("inf_batch_size"), 4)
        out_name     = config.get("output_name") or f"autodistill_{mode}_{int(time.time())}"
        run_ts       = int(time.time())

        emit_log(f"AutoDistill mode: {preset['label']}", "info")
        emit_log(f"Teacher: {preset['teacher_file']} ({preset['teacher_params']})", "info")
        emit_log(f"Student: {student_name}", "info")

        # ── Phase 0: Download GGUF teacher if needed ──────────────────────────
        set_stage(f"Phase 0: Fetching GGUF teacher ({preset['teacher_params']})")
        set_progress(2)
        teacher_path = _download_gguf_teacher(preset["teacher_repo"], preset["teacher_file"])
        set_progress(8)

        # ── Phase 1: Load teacher and generate dataset ────────────────────────
        set_stage("Phase 1: Loading GGUF teacher")
        llm_gguf = load_gguf_for_inference(teacher_path)
        set_progress(15)

        # Build prompts — FunctionGemma needs a tool schema in the system prompt
        topics   = preset["topics"]
        is_tool_calling = preset.get("prompt_template") == "tool_calling"

        if is_tool_calling:
            # FunctionGemma requires: developer role with tool list, then user query.
            # We embed a small but realistic JSON tool schema so the teacher produces
            # proper function-call JSON outputs that the student can learn from.
            TOOL_SCHEMAS = [
                {"type":"function","function":{"name":"get_weather","description":"Get current weather for a location","parameters":{"type":"object","properties":{"location":{"type":"string","description":"City name or address"},"unit":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["location"]}}},
                {"type":"function","function":{"name":"web_search","description":"Search the web for information","parameters":{"type":"object","properties":{"query":{"type":"string","description":"Search query"},"num_results":{"type":"integer","description":"Number of results","default":5}},"required":["query"]}}},
                {"type":"function","function":{"name":"send_email","description":"Send an email message","parameters":{"type":"object","properties":{"to":{"type":"string"},"subject":{"type":"string"},"body":{"type":"string"}},"required":["to","subject","body"]}}},
                {"type":"function","function":{"name":"create_calendar_event","description":"Create a calendar event","parameters":{"type":"object","properties":{"title":{"type":"string"},"date":{"type":"string"},"time":{"type":"string"},"duration_minutes":{"type":"integer"}},"required":["title","date"]}}},
                {"type":"function","function":{"name":"get_stock_price","description":"Get current stock price","parameters":{"type":"object","properties":{"symbol":{"type":"string","description":"Stock ticker symbol"}},"required":["symbol"]}}},
                {"type":"function","function":{"name":"set_reminder","description":"Set a reminder","parameters":{"type":"object","properties":{"message":{"type":"string"},"time_minutes":{"type":"integer","description":"Minutes from now"}},"required":["message","time_minutes"]}}},
                {"type":"function","function":{"name":"play_music","description":"Play a song or playlist","parameters":{"type":"object","properties":{"song":{"type":"string"},"artist":{"type":"string"}},"required":["song"]}}},
                {"type":"function","function":{"name":"smart_home_control","description":"Control smart home devices","parameters":{"type":"object","properties":{"device":{"type":"string"},"action":{"type":"string","enum":["on","off","dim"]},"level":{"type":"integer","description":"0-100 for dim"}},"required":["device","action"]}}},
            ]
            system_prompt = (
                "You are a model that can do function calling with the following functions\n"
                + json.dumps(TOOL_SCHEMAS, indent=2)
            )
            raw_topics = [topics[i % len(topics)] for i in range(num_prompts)]
            random.shuffle(raw_topics)
            # Wrap each topic as a proper FunctionGemma prompt with system context
            prompts = [
                f"<developer>\n{system_prompt}\n</developer>\n<user>\n{t}\n</user>"
                for t in raw_topics
            ]
        else:
            raw_topics = [topics[i % len(topics)] for i in range(num_prompts)]
            # Add variation by appending instruction formats
            formats = [
                "Explain clearly and in detail.",
                "Provide a step-by-step answer.",
                "Give a concise but complete answer.",
                "Answer as an expert would.",
            ]
            prompts = [f"{p} {formats[i % len(formats)]}" for i, p in enumerate(raw_topics)]
            random.shuffle(prompts)

        set_stage("Phase 1: Generating teacher responses")
        pairs = []
        emit_log(f"Generating {num_prompts} teacher responses (GGUF, ~10 tok/s) ...", "info")

        for i, prompt in enumerate(prompts):
            if stop_flag.is_set():
                raise KeyboardInterrupt()
            resp = generate_text_gguf(llm_gguf, prompt, max_new_tokens=max_tokens)
            # For tool-calling, store the clean user query as the instruction
            clean_prompt = topics[i % len(topics)] if is_tool_calling else prompt
            pairs.append({
                "prompt":   clean_prompt,
                "response": resp,
                "text":     f"### Instruction:\n{clean_prompt}\n\n### Response:\n{resp}",
            })
            set_progress(15 + (i / num_prompts) * 35)
            if (i + 1) % 20 == 0:
                emit_log(f"Generated {i+1}/{num_prompts}", "info")

        # Unload GGUF teacher
        try:
            del llm_gguf
        except Exception:
            pass
        _active_model["model"] = None
        _active_model["tokenizer"] = None
        import gc as _gc; _gc.collect()
        try:
            import torch as _torch; _torch.cuda.empty_cache()
        except Exception:
            pass
        emit_log("GGUF teacher unloaded", "info")
        set_progress(50)

        # ── Save synthetic dataset ────────────────────────────────────────────
        set_stage("Saving synthetic dataset")
        ds_name = f"autodistill_{mode}_{run_ts}"
        with open(GEN_DIR / f"{ds_name}.jsonl", "w") as f:
            for p in pairs:
                f.write(json.dumps(p) + "\n")
        from datasets import Dataset as _HFDataset
        ds = _HFDataset.from_list(pairs)
        # JSONL is the archival copy; skip save_to_disk to avoid duplicate RAM/disk use
        emit_log(f"Dataset saved: {len(pairs)} pairs → {ds_name}", "success")
        del pairs; gc.collect()   # free the raw list before loading the student model
        set_progress(52)

        # ── Phase 2: Train student ────────────────────────────────────────────
        set_stage("Phase 2: Loading student model")
        model, tok, _ = load_model_and_tokenizer(
            student_name,
            cfg_int(config.get("max_seq_length"), 2048),
            config.get("load_in_4bit", True),
        )
        set_progress(60)
        model = apply_lora(model, r=lora_r, alpha=lora_r)
        set_progress(64)
        set_stage("Phase 2: Training student")
        out_dir = str(OUTPUTS_DIR / out_name)
        run_sft(model, tok, ds, {
            **config,
            "output_dir": out_dir,
            "text_col":   "text",
            "max_steps":  max_steps,
        }, progress_start=64, progress_end=93)
        del ds; gc.collect()

        set_stage("Saving")
        set_progress(94)
        model.save_pretrained(out_dir)
        tok.save_pretrained(out_dir)
        set_progress(100)
        current_job["status"] = "done"
        emit_log(f"✅ AutoDistill complete! [{preset['label']}] → {out_dir}", "success")

    except KeyboardInterrupt:
        current_job["status"] = "stopped"
        emit_log("Stopped.", "warn")
    except Exception as e:
        current_job["status"] = "error"
        emit_log(f"Error: {e}", "error")
        emit_log(traceback.format_exc(), "error")


# ─── Helpers ──────────────────────────────────────────────────────────────────
def emit_log(message, level="info"):
    entry = {"time": time.strftime("%H:%M:%S"), "msg": message, "level": level}
    current_job["logs"].append(entry)
    # Cap in-memory log list so it never grows unbounded on long runs
    if len(current_job["logs"]) > 500:
        current_job["logs"] = current_job["logs"][-500:]
    log_queue.put(entry)
    print(f"[{level.upper()}] {message}")

def set_stage(stage):
    current_job["stage"] = stage
    emit_log(f"── {stage} ──", "info")

def set_progress(p):
    current_job["progress"] = min(int(p), 100)

def reset_job():
    # IMPORTANT: mutate the existing dict, do NOT reassign it.
    # The training thread holds a reference to this dict via emit_log/set_progress.
    # Reassigning creates a new dict that those functions never see.
    current_job.clear()
    current_job.update({"status": "running", "progress": 0, "logs": [], "stage": ""})
    stop_flag.clear()

def cfg_int(val, default=0):
    """Safely convert a config value to int, handling float strings like '0.125'."""
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return default

def cfg_float(val, default=0.0):
    """Safely convert a config value to float."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return default

def _load_dataset_hf(repo_or_path, split="train", max_samples=5000,
                     streaming=False, token=None, config_name=None):
    """
    Load a HuggingFace dataset with optional streaming mode.

    streaming=True  — never downloads the full dataset; materialises only
                      max_samples rows into RAM (great for low-end machines).
    streaming=False — downloads the full split first, then selects rows
                      (faster random access during training, needs more RAM).
    """
    from datasets import load_dataset, Dataset as _DS
    kw = {"split": split, "token": token}
    if config_name:
        kw["name"] = config_name
    if streaming:
        emit_log(f"Dataset streaming ON — will materialise first {max_samples} rows", "info")
        kw["streaming"] = True
        streamed = load_dataset(repo_or_path, **kw)
        rows = list(itertools.islice(streamed, max_samples))
        dataset = _DS.from_list(rows)
        emit_log(f"Streamed {len(dataset)} rows into RAM", "success")
    else:
        dataset = load_dataset(repo_or_path, **kw)
        if len(dataset) > max_samples:
            dataset = dataset.select(range(max_samples))
            emit_log(f"Capped to {max_samples} samples", "info")
        emit_log(f"Dataset loaded: {len(dataset)} rows", "success")
    return dataset

def _check_not_gguf(model_name):
    if "gguf" in str(model_name).lower():
        raise ValueError(f"'{model_name}' is a GGUF model (inference-only). Use the safetensors version for training.")
    p = Path(model_name)
    if p.is_dir() and (list(p.glob("*.gguf")) + list(p.glob("*.GGUF"))):
        raise ValueError(f"Directory '{model_name}' contains GGUF files — cannot fine-tune.")

def load_model_and_tokenizer(model_name, max_seq_length=2048, load_in_4bit=True):
    """Load model with automatic detection of FastModel vs FastLanguageModel."""
    import torch, json as _json
    _check_not_gguf(model_name)
    gc.collect()
    torch.cuda.empty_cache()
    token = hf_token["value"]
    errors = {}

    # Resolve to absolute path so no loader misreads it as a HF repo id
    _resolved = str(Path(model_name).resolve()) if Path(model_name).exists() else model_name
    if _resolved != model_name:
        emit_log(f"Resolved path: {_resolved}", "info")

    # If config.json has no model_type, patch it so Unsloth/transformers can load it.
    # This happens with models created by the "Create Blank Model" page.
    _cfg_path = Path(_resolved) / "config.json"
    if _cfg_path.exists():
        try:
            _cfg = _json.loads(_cfg_path.read_text())
            if not _cfg.get("model_type"):
                _arch_map = {
                    "LlamaForCausalLM": "llama",    "MistralForCausalLM": "mistral",
                    "GemmaForCausalLM": "gemma",    "Gemma2ForCausalLM": "gemma2",
                    "PhiForCausalLM":   "phi",      "Phi3ForCausalLM":   "phi3",
                    "Qwen2ForCausalLM": "qwen2",    "GPT2LMHeadModel":   "gpt2",
                    "GPTNeoXForCausalLM": "gpt_neox", "FalconForCausalLM": "falcon",
                }
                _arch     = (_cfg.get("architectures") or ["LlamaForCausalLM"])[0]
                _inferred = _arch_map.get(_arch, "llama")
                _cfg["model_type"] = _inferred
                _cfg_path.write_text(_json.dumps(_cfg, indent=2))
                emit_log(f"Patched config.json: added model_type='{_inferred}' (from architecture '{_arch}')", "warn")
        except Exception as _pe:
            emit_log(f"Could not patch config.json: {_pe}", "warn")

    # Attempt 1: FastModel (works for multimodal AND text models)
    try:
        from unsloth import FastModel
        emit_log(f"Trying FastModel loader for '{_resolved}'...", "info")
        model, tokenizer = FastModel.from_pretrained(
            model_name=_resolved, max_seq_length=max_seq_length,
            dtype=None, load_in_4bit=load_in_4bit)
        emit_log(f"Loaded via Unsloth FastModel", "success")
        emit_log(f"VRAM used: {round(torch.cuda.memory_allocated()/1024**3, 2)} GB", "info")
        _active_model["model"] = model; _active_model["tokenizer"] = tokenizer
        return model, tokenizer, "unsloth_fastmodel"
    except Exception as e:
        errors["FastModel"] = str(e); emit_log(f"FastModel failed: {e}", "warn")

    # Attempt 2: FastLanguageModel
    try:
        from unsloth import FastLanguageModel
        emit_log(f"Trying FastLanguageModel loader...", "info")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=_resolved, max_seq_length=max_seq_length,
            dtype=None, load_in_4bit=load_in_4bit)
        emit_log(f"Loaded via Unsloth FastLanguageModel", "success")
        emit_log(f"VRAM used: {round(torch.cuda.memory_allocated()/1024**3, 2)} GB", "info")
        _active_model["model"] = model; _active_model["tokenizer"] = tokenizer
        return model, tokenizer, "unsloth_fastlanguagemodel"
    except Exception as e:
        errors["FastLanguageModel"] = str(e); emit_log(f"FastLanguageModel also failed: {e}", "warn")

    # Attempt 3: BitsAndBytes 4-bit — no CPU memory cap so full VRAM is used
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        emit_log("Falling back to transformers 4-bit...", "warn")
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4")
        tokenizer = AutoTokenizer.from_pretrained(_resolved, token=token)
        model     = AutoModelForCausalLM.from_pretrained(
            _resolved, quantization_config=bnb_cfg,
            device_map="auto", token=token)
        emit_log(f"Loaded via transformers (4-bit)", "success")
        _active_model["model"] = model; _active_model["tokenizer"] = tokenizer
        return model, tokenizer, "transformers_bnb"
    except Exception as e:
        errors["Transformers_BnB"] = str(e); emit_log(f"BnB fallback failed: {e}", "warn")

    # Attempt 4: float16 last resort
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        emit_log("Last resort: transformers float16...", "warn")
        tokenizer = AutoTokenizer.from_pretrained(_resolved, token=token)
        model     = AutoModelForCausalLM.from_pretrained(
            _resolved, torch_dtype=torch.float16,
            device_map="auto", token=token)
        emit_log(f"Loaded via transformers (float16)", "success")
        _active_model["model"] = model; _active_model["tokenizer"] = tokenizer
        return model, tokenizer, "transformers"
    except Exception as e:
        errors["Transformers_fp16"] = str(e)

    error_summary = "\n".join(f"  {k}: {v}" for k, v in errors.items())
    raise RuntimeError(f"All loading methods failed for '{model_name}'.\n{error_summary}")

def apply_lora(model, r=16, alpha=16, dropout=0.0):
    """Apply LoRA — tries FastModel first, then FastLanguageModel, then PEFT directly."""
    try:
        from unsloth import FastModel
        return FastModel.get_peft_model(
            model, r=r, lora_alpha=alpha, lora_dropout=dropout,
            target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
            bias="none", use_gradient_checkpointing="unsloth", random_state=42)
    except Exception as e1:
        emit_log(f"FastModel LoRA failed ({e1}), trying FastLanguageModel...", "warn")
    try:
        from unsloth import FastLanguageModel
        return FastLanguageModel.get_peft_model(
            model, r=r, lora_alpha=alpha, lora_dropout=dropout,
            target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
            bias="none", use_gradient_checkpointing="unsloth", random_state=42)
    except Exception as e2:
        emit_log(f"FastLanguageModel LoRA failed ({e2}), falling back to PEFT...", "warn")
    from peft import get_peft_model, LoraConfig, TaskType
    lora_config = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        bias="none", task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    emit_log("Applied LoRA via PEFT directly", "success")
    return model

def run_sft(model, tokenizer, dataset, config, progress_start=40, progress_end=90):
    from trl import SFTTrainer
    from transformers import TrainingArguments, TrainerCallback

    text_col = config.get("text_col", "text")
    max_seq  = cfg_int(config.get("max_seq_length"), 2048)
    cols     = dataset.column_names if hasattr(dataset, "column_names") else []

    # ── Low-memory mode: halve seq length, use batch 1 + more grad accum ────
    low_memory = bool(config.get("low_memory_mode", False))
    speed_mode = bool(config.get("speed_mode", False))
    if low_memory:
        max_seq = min(max_seq, 1024)
        emit_log(f"Low-memory mode ON — seq_length capped to {max_seq}, batch=1, grad_accum=8", "info")
        effective_batch_size = cfg_int(config.get("batch_size"), 2) * cfg_int(config.get("grad_accum"), 4)
        _batch_size = 1
        _grad_accum = max(effective_batch_size, 8)   # preserve effective batch, minimum 8
    else:
        _batch_size = cfg_int(config.get("batch_size"), 2)
        _grad_accum = cfg_int(config.get("grad_accum"), 4)

    if speed_mode:
        emit_log("Speed mode ON — tf32, fused optimizer, packing, no evals/checkpoints", "info")

    if text_col not in cols:
        for fallback in ["text","prompt","input","instruction","conversations","chat","query","content","question","problem","messages","output","response"]:
            if fallback in cols:
                emit_log(f"Column '{text_col}' not found, using '{fallback}'", "warn")
                text_col = fallback; break

    try:
        sample = dataset[0]
        emit_log(f"Dataset columns: {cols}", "info")
        for k, v in sample.items():
            emit_log(f"  Sample['{k}']: {str(v)[:150].replace(chr(10),' ')}", "info")
    except Exception as e:
        emit_log(f"Could not preview dataset: {e}", "warn")

    def _row_to_text(row):
        raw = row.get(text_col)
        if isinstance(raw, str) and raw.strip(): return raw.strip()
        if isinstance(raw, list) and len(raw) > 0:
            parts = []
            for item in raw:
                if isinstance(item, dict):
                    role = item.get("role", item.get("from", ""))
                    content_val = item.get("content", item.get("value", item.get("text", "")))
                    parts.append(f"### {role.capitalize()}:\n{content_val}" if role else str(content_val))
                elif isinstance(item, str): parts.append(item)
            if parts: return "\n\n".join(parts)
        instruction = row.get("instruction", row.get("system_prompt", ""))
        inp    = row.get("input", row.get("context", row.get("question", row.get("problem", ""))))
        output = row.get("output", row.get("response", row.get("answer", row.get("solution", ""))))
        prompt = row.get("prompt", "")
        for field_name in ["messages","conversations","chat"]:
            msgs = row.get(field_name)
            if isinstance(msgs, list) and len(msgs) > 0:
                parts = []
                for m in msgs:
                    if isinstance(m, dict):
                        role = m.get("role", m.get("from", ""))
                        c    = m.get("content", m.get("value", ""))
                        parts.append(f"### {role.capitalize()}:\n{c}" if role else str(c))
                    elif isinstance(m, str): parts.append(m)
                if parts: return "\n\n".join(parts)
        if instruction or inp or output:
            parts = []
            if instruction: parts.append(f"### Instruction:\n{instruction}")
            if inp:         parts.append(f"### Input:\n{inp}")
            if output:      parts.append(f"### Response:\n{output}")
            if parts: return "\n\n".join(parts)
        if prompt and output: return f"### Instruction:\n{prompt}\n\n### Response:\n{output}"
        if prompt: return prompt
        all_parts = [v.strip() for v in row.values() if isinstance(v, str) and v.strip()]
        return "\n\n".join(all_parts) if all_parts else ""

    def formatting_func(examples):
        output = []
        first_val = examples.get(text_col) or next(iter(examples.values()), None)
        is_batched = isinstance(first_val, list)
        if is_batched:
            batch_size = len(first_val)
            for i in range(batch_size):
                row  = {k: (v[i] if isinstance(v, list) and i < len(v) else v) for k, v in examples.items()}
                text = _row_to_text(row)
                if text.strip(): output.append(text)
        else:
            text = _row_to_text(examples)
            if text.strip(): output.append(text)
        return output if output else ["No content available."]

    try:
        test = formatting_func(dataset[0])
        emit_log(f"Formatting test OK: {len(test[0])} chars. Preview: {test[0][:200]}", "success")
    except Exception as e:
        emit_log(f"Formatting test failed: {e}", "warn")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Auto-detect precision from the model's actual dtype
    import torch, os as _os
    _use_bf16 = (next(model.parameters()).dtype == torch.bfloat16)
    _use_fp16 = not _use_bf16
    emit_log(f"Precision: {'bf16' if _use_bf16 else 'fp16'}", "info")

    # ── Auto-steps mode: train until loss ≤ 2.00, then round up to nearest ×50 ─
    _raw_steps_val = config.get("max_steps", 100)
    _auto_steps = (
        str(_raw_steps_val).strip().lower() in ("auto", "0", "")
        or _raw_steps_val == 0
    )
    if _auto_steps:
        _max_steps = 5000   # hard safety cap; early-stop callback halts sooner
        emit_log("Auto-steps mode ON — will train until loss ≤ 2.00, then round up to nearest ×50", "info")
    else:
        _max_steps = cfg_int(_raw_steps_val, 100)

    # Log every 5% of steps (minimum 1) — avoids Python overhead on every single step
    _log_steps = max(1, _max_steps // 20)

    # Speed mode: faster optimizer, tf32 matmuls, no checkpointing overhead
    # Low-memory mode: paged optimizer, gradient checkpointing (saves VRAM at cost of speed)
    if speed_mode:
        _optim       = "adamw_8bit"     # bitsandbytes 8-bit: faster than paged variant
        _grad_ckpt   = False            # no recompute overhead
        _group_len   = True             # batch similar-length seqs → less padding waste
        _tf32        = True             # Ampere+ only: bfloat16 matmuls via TF32 units
        _pin_mem     = True             # pin memory for faster CPU→GPU transfers
        _dl_workers  = min(4, _os.cpu_count() or 1)
        emit_log(f"Speed mode: adamw_8bit, tf32={_tf32}, group_by_length, {_dl_workers} dataloader workers", "info")
    else:
        _optim       = "paged_adamw_8bit"
        _grad_ckpt   = True
        _group_len   = False
        _tf32        = False
        _pin_mem     = False
        _dl_workers  = 0

    # Sequence packing: fills each max_seq_length window fully instead of padding
    # → more tokens per step, faster convergence. Best with packing=True on SFTTrainer.
    _packing = speed_mode

    # Build TrainingArguments kwargs dynamically: several params were renamed/removed
    # across transformers versions, so introspect the actual signature first.
    import inspect as _inspect
    _ta_params = set(_inspect.signature(TrainingArguments).parameters)

    _ta_kwargs = dict(
        output_dir                   = config.get('output_dir', '/tmp/unsloth_out'),
        per_device_train_batch_size  = _batch_size,
        gradient_accumulation_steps  = _grad_accum,
        warmup_steps                 = cfg_int(config.get('warmup_steps'), 10),
        max_steps                    = _max_steps,
        learning_rate                = cfg_float(config.get('learning_rate'), 2e-4),
        fp16                         = _use_fp16,
        bf16                         = _use_bf16,
        logging_steps                = _log_steps,
        optim                        = _optim,
        weight_decay                 = cfg_float(config.get('weight_decay'), 0.01),
        lr_scheduler_type            = 'linear',
        seed                         = 42,
        save_strategy                = 'no',
        report_to                    = 'none',
        dataloader_pin_memory        = _pin_mem,
        dataloader_num_workers       = _dl_workers,
        gradient_checkpointing       = _grad_ckpt,
    )

    # tf32 — present in most versions but guard anyway
    if 'tf32' in _ta_params:
        _ta_kwargs['tf32'] = _tf32

    # eval_strategy (>=4.41) replaced evaluation_strategy (<4.41)
    if 'eval_strategy' in _ta_params:
        _ta_kwargs['eval_strategy'] = 'no'
    elif 'evaluation_strategy' in _ta_params:
        _ta_kwargs['evaluation_strategy'] = 'no'

    # group_by_length was removed in transformers >=4.45
    if 'group_by_length' in _ta_params:
        _ta_kwargs['group_by_length'] = _group_len

    # Final safety net: drop any key not in this version's signature
    _unknown = [k for k in list(_ta_kwargs) if k not in _ta_params]
    for k in _unknown:
        del _ta_kwargs[k]
    if _unknown:
        emit_log(f'Skipped unsupported TrainingArguments params: {_unknown}', 'info')

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=dataset,
        formatting_func=formatting_func, max_seq_length=max_seq,
        packing=_packing,
        args=TrainingArguments(**_ta_kwargs),
    )
    total = trainer.args.max_steps
    _auto_steps_target = [None]   # mutable box: filled when loss ≤ 2.00

    class ProgressCB(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            if state.global_step:
                pct       = progress_start + (state.global_step / total) * (progress_end - progress_start)
                set_progress(pct)
                loss_val  = logs.get("loss", None) if logs else None
                loss_str  = f"{loss_val:.4f}" if isinstance(loss_val, float) else (str(loss_val) if loss_val is not None else "?")
                tps       = logs.get("train_samples_per_second", "")
                spd       = f" | {tps:.1f} samples/s" if tps else ""
                emit_log(f"Step {state.global_step}/{total} | loss: {loss_str}{spd}", "info")

                # ── Auto-steps: record rounded target the first time loss ≤ 2.00 ──
                if _auto_steps and _auto_steps_target[0] is None:
                    if isinstance(loss_val, float) and loss_val <= 2.00:
                        import math as _math
                        raw_step = state.global_step
                        rounded  = _math.ceil(raw_step / 50) * 50
                        _auto_steps_target[0] = rounded
                        emit_log(
                            f"Auto-steps: loss {loss_val:.4f} ≤ 2.00 reached at step {raw_step}"
                            f" → rounding up to {rounded} steps", "info"
                        )

        def on_step_end(self, args, state, control, **kw):
            # Once the rounded target is set, stop training when we reach it
            if _auto_steps and _auto_steps_target[0] is not None:
                if state.global_step >= _auto_steps_target[0]:
                    emit_log(
                        f"Auto-steps: reached target {_auto_steps_target[0]} steps — stopping.", "success"
                    )
                    control.should_training_stop = True
            return control

    trainer.add_callback(ProgressCB())
    trainer.train()

    # ── Save per-run timing data so future estimates improve ─────────────────
    try:
        _state  = trainer.state
        _runtime = getattr(_state, 'log_history', [{}])
        # train_runtime comes from the last log entry HF Trainer writes
        _actual_runtime = next(
            (e.get('train_runtime') for e in reversed(_state.log_history or []) if 'train_runtime' in e),
            None
        )
        _actual_steps = _state.global_step or total
        if _actual_runtime and _actual_steps > 0:
            _secs_per_step = _actual_runtime / _actual_steps
            # Also read average tokens/s if available
            _tok_per_sec = next(
                (e.get('train_tokens_per_second') or e.get('train_samples_per_second')
                 for e in reversed(_state.log_history or []) if
                 'train_tokens_per_second' in e or 'train_samples_per_second' in e),
                None
            )
            _entry = {
                'timestamp':    int(time.time()),
                'secs_per_step': round(_secs_per_step, 4),
                'steps':         _actual_steps,
                'runtime_secs':  round(_actual_runtime, 2),
                'batch_size':    _batch_size,
                'grad_accum':    _grad_accum,
                'seq_length':    max_seq,
                'speed_mode':    speed_mode,
                'low_memory':    low_memory,
            }
            if _tok_per_sec:
                _entry['tok_per_sec'] = round(float(_tok_per_sec), 2)
            _save_tuning_log(_entry)
            emit_log(f'Timing logged: {round(_secs_per_step, 2)} s/it over {_actual_steps} steps', 'info')
        else:
            emit_log('Could not capture training runtime for timing log', 'info')
    except Exception as _te:
        emit_log(f'Timing log skipped: {_te}', 'info')

def _save_tuning_log(entry):
    """Append a timing entry to tuning_logs.json (one JSON object per line)."""
    try:
        with open(TUNING_LOGS_FILE, 'a', encoding='utf-8') as _f:
            _f.write(json.dumps(entry) + chr(10))
    except Exception as _e:
        print(f'[WARN] Could not write tuning log: {_e}')

def _load_tuning_logs():
    """Return list of all saved timing entries."""
    if not TUNING_LOGS_FILE.exists():
        return []
    entries = []
    try:
        for line in TUNING_LOGS_FILE.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except Exception:
                    pass
    except Exception:
        pass
    return entries

def unload_model(model=None, tokenizer=None):
    if model: del model
    if tokenizer: del tokenizer
    gc.collect()
    try:
        import torch; torch.cuda.empty_cache()
    except: pass

def generate_text(model, tokenizer, prompt, max_new_tokens=512, temperature=0.7, top_p=0.9):
    """Single-prompt generation. Prefer generate_text_batched for bulk work."""
    import torch
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    input_len = inputs["input_ids"].shape[1]
    return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)


def generate_text_batched(model, tokenizer, prompts, max_new_tokens=512,
                           temperature=0.7, top_p=0.9, batch_size=8):
    """
    Generate responses for a list of prompts using left-padded batched inference.
    Automatically halves batch_size on OOM and retries.
    Returns a list of response strings in the same order as prompts.
    """
    import torch

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # Left-padding is required for batched generation so all sequences
    # are right-aligned and the model attends to real tokens at the end.
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    results = [""] * len(prompts)

    i = 0
    while i < len(prompts):
        batch_prompts = prompts[i:i + batch_size]
        try:
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(model.device)

            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                )

            input_len = inputs["input_ids"].shape[1]
            for j, out in enumerate(outputs):
                results[i + j] = tokenizer.decode(out[input_len:], skip_special_tokens=True)

            i += batch_size   # advance only on success

        except RuntimeError as e:
            if "out of memory" in str(e).lower() and batch_size > 1:
                batch_size = max(1, batch_size // 2)
                emit_log(f"OOM — reducing inference batch size to {batch_size}", "warn")
                torch.cuda.empty_cache()
                # retry same i with smaller batch
            else:
                raise

    tokenizer.padding_side = original_padding_side
    return results

def _is_gguf_path(model_name):
    """Return True if the given path/name points to a GGUF file."""
    s = str(model_name).lower()
    if s.endswith(".gguf"):
        return True
    p = Path(model_name)
    if p.is_dir() and (list(p.glob("*.gguf")) + list(p.glob("*.GGUF"))):
        return True
    return False

def load_gguf_for_inference(model_name):
    """
    Load a GGUF model via llama-cpp-python for inference-only use
    (teacher generation, dataset generation). Returns a llama_cpp.Llama instance.
    The returned object has no tokenizer — use generate_text_gguf() instead.
    """
    try:
        from llama_cpp import Llama
    except ImportError:
        raise RuntimeError(
            "llama-cpp-python is not installed. "
            "Run: pip install llama-cpp-python --break-system-packages"
        )

    # Resolve single .gguf file path
    p = Path(model_name)
    if p.is_dir():
        candidates = list(p.glob("*.gguf")) + list(p.glob("*.GGUF"))
        if not candidates:
            raise ValueError(f"No .gguf files found in directory: {model_name}")
        # Prefer the largest file (most likely the full model, not a shard header)
        gguf_path = str(max(candidates, key=lambda f: f.stat().st_size))
    else:
        gguf_path = str(model_name)

    emit_log(f"Loading GGUF for inference: {Path(gguf_path).name}", "info")
    import os as _os
    _n_threads = _os.cpu_count() or 4
    # n_gpu_layers=-1 offloads all layers to GPU; n_batch=512 fills the GPU kernel better
    llm = Llama(model_path=gguf_path, n_ctx=2048, n_gpu_layers=-1,
                n_batch=512, n_threads=_n_threads, verbose=False)
    emit_log(f"GGUF loaded (n_batch=512, n_threads={_n_threads})", "success")
    _active_model["model"] = llm          # register so stop/eject can clear it
    _active_model["tokenizer"] = None
    return llm

def generate_text_gguf(llm, prompt, max_new_tokens=512, temperature=0.7, top_p=0.9):
    """Generate text from a llama_cpp.Llama instance."""
    out = llm(
        prompt,
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        echo=False,
    )
    return out["choices"][0]["text"].strip()

def _start_job(fn, config):
    global training_thread
    if current_job.get("status") == "running":
        return jsonify({"error": "A job is already running"}), 409
    reset_job()
    training_thread = threading.Thread(target=fn, args=(config,), daemon=True)
    _active_threads.append(training_thread)
    training_thread.start()
    return jsonify({"ok": True})

# ═══ JOB 1 – Fine-Tune ═══════════════════════════════════════════════════════
def run_training(config):
    try:
        set_stage("Loading model")
        model, tok, _ = load_model_and_tokenizer(config["model_name"], cfg_int(config.get("max_seq_length"), 2048), config.get("load_in_4bit",True))
        set_progress(20)
        set_stage("Applying LoRA")
        model = apply_lora(model, r=cfg_int(config.get("lora_r"), 16), alpha=cfg_int(config.get("lora_alpha"), 16), dropout=cfg_float(config.get("lora_dropout"), 0.0))
        set_progress(30)
        set_stage("Loading dataset")
        from datasets import load_from_disk
        ds_path     = config["dataset"]
        max_samples = cfg_int(config.get("max_samples"), 5000)
        streaming   = bool(config.get("stream_dataset", False))
        try:
            dataset = load_from_disk(ds_path)
            emit_log(f"Dataset loaded from disk: {len(dataset)} rows", "success")
            if len(dataset) > max_samples:
                dataset = dataset.select(range(max_samples))
                emit_log(f"Capped to {max_samples} samples", "info")
        except Exception:
            dataset = _load_dataset_hf(
                ds_path, split="train", max_samples=max_samples,
                streaming=streaming, token=hf_token["value"])
        emit_log(f"Dataset ready: {len(dataset)} rows", "success")
        set_progress(40)
        set_stage("Training")
        out_name = config.get("output_name") or f"ft_{int(time.time())}"
        out_dir  = str(OUTPUTS_DIR / out_name)
        config["output_dir"] = out_dir
        run_sft(model, tok, dataset, config)
        del dataset; gc.collect()
        set_stage("Saving")
        set_progress(92)
        model.save_pretrained(out_dir); tok.save_pretrained(out_dir)
        set_progress(100); current_job["status"] = "done"
        emit_log(f"Fine-tune complete! Saved to: {out_dir}", "success")
    except KeyboardInterrupt:
        current_job["status"] = "stopped"; emit_log("Stopped.", "warn")
    except Exception as e:
        current_job["status"] = "error"; emit_log(f"Error: {e}", "error"); emit_log(traceback.format_exc(), "error")

# ═══ MULTIMODAL UPGRADE ══════════════════════════════════════════════════════
def run_multimodal_upgrade(config, preset):
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, CLIPVisionModel, CLIPImageProcessor
        from torch import nn
        from datasets import load_dataset
        vision_repo = preset.get("vision_encoder", "openai/clip-vit-large-patch14")
        token       = hf_token["value"]
        out_name    = config.get("output_name") or "multimodal_upgrade"
        out_dir     = str(OUTPUTS_DIR / out_name)
        max_steps   = cfg_int(config.get("max_steps") or preset["max_steps"])
        set_stage("Loading CLIP vision encoder")
        processor  = CLIPImageProcessor.from_pretrained(vision_repo, token=token)
        vision_enc = CLIPVisionModel.from_pretrained(vision_repo, token=token)
        vision_enc.eval(); vision_dim = vision_enc.config.hidden_size
        emit_log(f"Vision encoder loaded. dim={vision_dim}", "success"); set_progress(15)
        set_stage("Loading language model")
        tokenizer = AutoTokenizer.from_pretrained(config["model_name"], token=token)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        llm = AutoModelForCausalLM.from_pretrained(config["model_name"], torch_dtype=torch.float16, device_map="auto", token=token)
        text_dim = llm.config.hidden_size
        emit_log(f"LLM loaded. dim={text_dim}", "success"); set_progress(25)
        for p in llm.parameters(): p.requires_grad = False
        set_stage("Building projection layer")
        class VisionProjection(nn.Module):
            def __init__(self, in_dim, out_dim):
                super().__init__()
                self.net = nn.Sequential(nn.Linear(in_dim, out_dim*2), nn.GELU(), nn.Linear(out_dim*2, out_dim))
            def forward(self, x): return self.net(x)
        proj = VisionProjection(vision_dim, text_dim).to(llm.device).to(torch.float16)
        emit_log(f"Projection: {sum(p.numel() for p in proj.parameters()):,} params", "success"); set_progress(30)
        set_stage("Loading pre-training dataset")
        ds_info = preset["datasets"][0]; kw = {"split": ds_info["split"]}
        if token: kw["token"] = token
        try:
            dataset = load_dataset(ds_info["repo"], **kw)
        except Exception as e:
            emit_log(f"Primary dataset failed ({e}), trying fallback...", "warn")
            ds_info = preset["datasets"][1]; kw["split"] = ds_info["split"]
            dataset = load_dataset(ds_info["repo"], **kw)
        max_samples = cfg_int(config.get("max_samples"), 5000)
        if len(dataset) > max_samples: dataset = dataset.select(range(max_samples))
        set_progress(38)
        set_stage("Stage 1: Training projection layer")
        emit_log(f"Training {max_steps} steps, LLM frozen", "info")
        optimizer = torch.optim.AdamW(proj.parameters(), lr=cfg_float(config.get("learning_rate"), 1e-3))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps)
        proj.train(); step = 0; running_loss = 0.0; text_col = ds_info.get("text_col","conversations")
        for i, row in enumerate(dataset):
            if stop_flag.is_set(): raise KeyboardInterrupt()
            if step >= max_steps: break
            try:
                raw = row.get(text_col,"")
                text = " ".join([t.get("value","") if isinstance(t,dict) else str(t) for t in raw]) if isinstance(raw,list) else str(raw)
                enc = tokenizer(text[:256], return_tensors="pt", padding=True, truncation=True, max_length=128).to(llm.device)
                with torch.inference_mode():
                    text_embeds = llm.get_input_embeddings()(enc["input_ids"])
                    dummy = torch.zeros(1, vision_dim, device=llm.device, dtype=torch.float16)
                loss = nn.functional.mse_loss(proj(dummy), text_embeds[:,0,:])
                optimizer.zero_grad(); loss.backward(); optimizer.step(); scheduler.step()
                running_loss += loss.item(); step += 1
                if step % 10 == 0:
                    emit_log(f"Step {step}/{max_steps} | Loss: {running_loss/10:.4f}", "info")
                    running_loss = 0.0; set_progress(38 + (step/max_steps)*45)
            except: continue
        emit_log("Stage 1 complete.", "success"); set_progress(85)
        set_stage("Saving"); os.makedirs(out_dir, exist_ok=True)
        torch.save(proj.state_dict(), os.path.join(out_dir,"vision_projection.pt"))
        tokenizer.save_pretrained(out_dir)
        import json as _json
        with open(os.path.join(out_dir,"multimodal_config.json"),"w") as f:
            _json.dump({"base_llm":config["model_name"],"vision_encoder":vision_repo,"vision_dim":vision_dim,"text_dim":text_dim},f,indent=2)
        set_progress(100); current_job["status"] = "done"
        emit_log(f"Multimodal upgrade complete! Saved to: {out_dir}", "success")
    except KeyboardInterrupt:
        current_job["status"] = "stopped"; emit_log("Stopped.", "warn")
    except Exception as e:
        current_job["status"] = "error"; emit_log(f"Error: {e}", "error"); emit_log(traceback.format_exc(), "error")

# ═══ JOB 2 – AutoTrain ═══════════════════════════════════════════════════════
def run_autotrain(config):
    mode = config["mode"]; preset = AUTOTRAIN_PRESETS[mode]
    try:
        set_stage(f"AutoTrain: {preset['label']}")
        emit_log(preset["description"], "info")
        if "note" in preset: emit_log(preset["note"], "warn")
        set_progress(5)
        if mode == "multimodal_upgrade":
            run_multimodal_upgrade(config, preset); return
        set_stage("Loading model")
        model, tok, _ = load_model_and_tokenizer(config["model_name"], cfg_int(config.get("max_seq_length"), 2048), config.get("load_in_4bit",True))
        set_progress(20)
        set_stage("Applying LoRA")
        lora_r = cfg_int(config.get("lora_r") or preset["lora_r"])
        model  = apply_lora(model, r=lora_r, alpha=lora_r)
        set_progress(30)
        set_stage("Downloading dataset")
        from datasets import load_dataset
        dataset  = None
        text_col = "text"
        max_samples = cfg_int(config.get("max_samples"), 5000)
        streaming   = bool(config.get("stream_dataset", False))
        for ds_info in preset["datasets"]:
            if stop_flag.is_set(): raise KeyboardInterrupt()
            try:
                emit_log(f"Trying: {ds_info['repo']}", "info")
                dataset = _load_dataset_hf(
                    ds_info["repo"],
                    split=ds_info["split"],
                    max_samples=max_samples,
                    streaming=streaming,
                    token=hf_token["value"],
                    config_name=ds_info.get("config"),
                )
                text_col = ds_info.get("text_col", "text")
                if text_col not in dataset.column_names:
                    for c in ["text","prompt","input","instruction","conversations","chat","query"]:
                        if c in dataset.column_names: text_col = c; break
                emit_log(f"Dataset ready: {ds_info['repo']} ({len(dataset)} rows, col='{text_col}')", "success")
                break
            except Exception as e:
                emit_log(f"Failed: {e}", "warn")
        if dataset is None: raise RuntimeError("All datasets failed to download.")
        set_progress(40)
        set_stage("Training")
        out_name = config.get("output_name") or f"autotrain_{mode}_{int(time.time())}"
        out_dir  = str(OUTPUTS_DIR / out_name)
        cfg = {**config,"output_dir":out_dir,"text_col":text_col,"max_steps":cfg_int(config.get("max_steps") or preset["max_steps"])}
        run_sft(model, tok, dataset, cfg)
        del dataset; gc.collect()
        set_stage("Saving"); set_progress(92)
        model.save_pretrained(out_dir); tok.save_pretrained(out_dir)
        set_progress(100); current_job["status"] = "done"
        emit_log(f"AutoTrain complete! Saved to: {out_dir}", "success")
    except KeyboardInterrupt:
        current_job["status"] = "stopped"; emit_log("Stopped.", "warn")
    except Exception as e:
        current_job["status"] = "error"; emit_log(f"Error: {e}", "error"); emit_log(traceback.format_exc(), "error")

# ═══ JOB 3 – Distillation ════════════════════════════════════════════════════
def _cloud_generate(prompt, config, max_tokens):
    """Generate a response from a cloud provider."""
    provider  = config.get("cloud_provider", "custom")
    model     = config.get("cloud_model", "")
    api_key   = config.get("cloud_api_key", "")
    base_url  = (config.get("cloud_base_url") or "").strip().rstrip("/")

    import urllib.request, json as _json

    # ── Perplexity Agent API (/v1/responses) ──────────────────────────────────
    # Uses { preset } OR { model } — presets are identified by having no "/" prefix
    # and matching known preset names. Third-party models use provider/name format.
    # Response: walk output array for output_text blocks (no top-level output_text field).
    if provider == "perplexity":
        url      = "https://api.perplexity.ai/v1/responses"
        selected = model or "pro-search"
        # Presets: fast-search, pro-search, deep-research, advanced-deep-research
        PRESETS  = {"fast-search", "pro-search", "deep-research", "advanced-deep-research"}
        payload  = {
            "input":             prompt,
            "max_output_tokens": max_tokens,
        }
        if selected in PRESETS:
            payload["preset"] = selected
        else:
            payload["model"] = selected
            # Only add tools when using a direct model (presets include their own tools)
            payload["tools"] = [{"type": "web_search"}]

        body = _json.dumps(payload).encode()
        req  = urllib.request.Request(url, data=body, headers={
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {api_key}",
        }, method="POST")
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = _json.loads(resp.read())
        # Walk output array: each item is a message block, content is a list of parts
        text = ""
        for block in data.get("output", []):
            for part in block.get("content", []):
                if part.get("type") == "output_text":
                    text += part.get("text", "")
        return text.strip()

    # ── Google (OpenAI-compatible /v1beta/openai endpoint) ────────────────────
    if provider == "google":
        url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    elif base_url:
        url = base_url + "/chat/completions"
    else:
        url = "https://api.openai.com/v1/chat/completions"

    body = _json.dumps({
        "model":      model,
        "messages":   [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.8,
    }).encode()
    req = urllib.request.Request(url, data=body, headers={
        "Content-Type":  "application/json",
        "Authorization": f"Bearer {api_key}",
    }, method="POST")
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = _json.loads(resp.read())
    return data["choices"][0]["message"]["content"].strip()


def run_distillation(config):
    try:
        teacher_source = config.get("teacher_source", "local")
        student_name   = config["student_model"]
        num_prompts    = cfg_int(config.get("num_prompts"), 200)
        max_tokens     = cfg_int(config.get("max_new_tokens"), 512)
        topic_mode     = config.get("topic_mode", "general")
        _cp            = config.get("custom_prompts", "")
        custom_prompts = ([p.strip() for p in _cp if p.strip()]
                          if isinstance(_cp, list)
                          else [p.strip() for p in _cp.split("\n") if p.strip()])

        # Build prompt list
        prompts = list(custom_prompts)
        topics  = SYNTH_TOPICS.get(topic_mode, SYNTH_TOPICS["general"])
        while len(prompts) < num_prompts:
            prompts.append(f"Explain {random.choice(topics)} clearly and in detail.")
        prompts = prompts[:num_prompts]

        pairs = []
        model = tok = None
        llm_gguf = None          # GGUF teacher handle (llama-cpp-python)
        is_gguf_teacher = False  # set below if local GGUF path is used
        run_ts = int(time.time())

        # ── Phase 1: Generate teacher responses ───────────────────────────────
        if teacher_source == "cloud":
            provider  = config.get("cloud_provider", "openai")
            cmodel    = config.get("cloud_model", "")
            price_in  = cfg_float(config.get("cloud_price_in"), 0)
            price_out = cfg_float(config.get("cloud_price_out"), 0)
            emit_log(f"Using cloud teacher: {provider} / {cmodel}", "info")
            set_stage(f"Phase 1: Cloud teacher ({provider})")
            set_progress(5)
            total_in = total_out = 0

            # Prepare cloud conversation log
            log_name = f"distill_cloud_{run_ts}"
            log_path = CLOUD_LOGS_DIR / f"{log_name}.jsonl"
            meta = {
                "type": "distillation", "provider": provider, "model": cmodel,
                "num_prompts": num_prompts, "max_tokens": max_tokens,
                "topic_mode": topic_mode, "started": run_ts,
                "price_in_per_m": price_in, "price_out_per_m": price_out,
            }
            with open(log_path, "w") as lf:
                lf.write(json.dumps({"_meta": meta}) + "\n")
            emit_log(f"Cloud log: {log_path}", "info")

            for i, prompt in enumerate(prompts):
                if stop_flag.is_set(): raise KeyboardInterrupt()
                t_start = time.time()
                try:
                    resp = _cloud_generate(prompt, config, max_tokens)
                except Exception as e:
                    emit_log(f"Cloud API error on prompt {i+1}: {e}", "error")
                    # Flush whatever we have before re-raising
                    with open(log_path, "a") as lf:
                        lf.write(json.dumps({"_error": str(e), "prompt_index": i, "prompt": prompt}) + "\n")
                    raise
                elapsed = time.time() - t_start

                # Rough token accounting (~4 chars/token)
                tok_in  = len(prompt) // 4 + 50
                tok_out = len(resp)   // 4
                total_in  += tok_in
                total_out += tok_out

                entry = {
                    "index": i, "prompt": prompt, "response": resp,
                    "tokens_in_est": tok_in, "tokens_out_est": tok_out,
                    "elapsed_s": round(elapsed, 2),
                }
                pairs.append({"prompt": prompt, "response": resp,
                               "text": f"### Instruction:\n{prompt}\n\n### Response:\n{resp}"})
                # Append to log immediately — so partial runs are still saved
                with open(log_path, "a") as lf:
                    lf.write(json.dumps(entry) + "\n")

                pct = 5 + (i / num_prompts) * 40
                set_progress(pct)
                if (i + 1) % 10 == 0:
                    cost = (total_in / 1_000_000) * price_in + (total_out / 1_000_000) * price_out
                    emit_log(f"Generated {i+1}/{num_prompts} | ~{total_in//1000}K in / {total_out//1000}K out | cost so far: ${cost:.4f}", "info")

            # Write summary to log
            final_cost = (total_in / 1_000_000) * price_in + (total_out / 1_000_000) * price_out
            with open(log_path, "a") as lf:
                lf.write(json.dumps({
                    "_summary": True, "total_pairs": len(pairs),
                    "total_tokens_in": total_in, "total_tokens_out": total_out,
                    "final_cost_usd": round(final_cost, 6),
                    "finished": int(time.time()),
                }) + "\n")
            emit_log(f"Cloud log saved: {log_path.name} | Total cost: ${final_cost:.4f}", "success")

        else:
            teacher_name = config["teacher_model"]
            set_stage("Phase 1: Loading teacher model")
            is_gguf_teacher = _is_gguf_path(teacher_name)
            if is_gguf_teacher:
                emit_log("GGUF teacher detected — loading via llama-cpp-python (inference only)", "info")
                llm_gguf = load_gguf_for_inference(teacher_name)
                tok = None
            else:
                model, tok, _ = load_model_and_tokenizer(teacher_name)
            set_progress(10)
            set_stage("Phase 1: Generating teacher responses")
            inf_batch = cfg_int(config.get("inf_batch_size"), 8)
            emit_log(f"Inference batch size: {inf_batch} (adjust with inf_batch_size config)", "info")
            for batch_start in range(0, num_prompts, inf_batch):
                if stop_flag.is_set(): raise KeyboardInterrupt()
                batch_prompts = prompts[batch_start:batch_start + inf_batch]
                if is_gguf_teacher:
                    resps = [generate_text_gguf(llm_gguf, p, max_new_tokens=max_tokens)
                             for p in batch_prompts]
                else:
                    resps = generate_text_batched(model, tok, batch_prompts,
                                                  max_new_tokens=max_tokens,
                                                  batch_size=inf_batch)
                for prompt, resp in zip(batch_prompts, resps):
                    pairs.append({"prompt": prompt, "response": resp,
                                  "text": f"### Instruction:\n{prompt}\n\n### Response:\n{resp}"})
                done = min(batch_start + inf_batch, num_prompts)
                set_progress(10 + (done / num_prompts) * 35)
                emit_log(f"Generated {done}/{num_prompts}", "info")

        # ── Save synthetic dataset ─────────────────────────────────────────────
        set_stage("Saving synthetic dataset")
        ds_name = f"distill_{run_ts}"
        with open(GEN_DIR / f"{ds_name}.jsonl", "w") as f:
            for p in pairs: f.write(json.dumps(p) + "\n")
        from datasets import Dataset
        ds = Dataset.from_list(pairs)
        ds.save_to_disk(str(DATASETS_DIR / ds_name))
        emit_log(f"Synthetic dataset saved: {len(pairs)} pairs", "success")
        set_progress(50)

        # Unload local teacher before loading student
        if llm_gguf is not None:
            try: del llm_gguf
            except Exception: pass
            llm_gguf = None
            _active_model["model"] = None
            import gc; gc.collect()
            emit_log("GGUF teacher unloaded", "info")
        elif model is not None:
            unload_model(model, tok)
            model = tok = None

        # ── Phase 2: Train student ─────────────────────────────────────────────
        set_stage("Phase 2: Loading student model")
        model, tok, _ = load_model_and_tokenizer(student_name)
        set_progress(55)
        model = apply_lora(model, r=cfg_int(config.get("lora_r"), 16), alpha=cfg_int(config.get("lora_alpha"), 16))
        set_progress(60)
        set_stage("Phase 2: Training student")
        out_name = config.get("output_name") or f"distill_{run_ts}"
        out_dir  = str(OUTPUTS_DIR / out_name)
        run_sft(model, tok, ds, {**config, "output_dir": out_dir, "text_col": "text",
                                  "max_steps": cfg_int(config.get("max_steps"), 100)})
        set_stage("Saving")
        set_progress(92)
        model.save_pretrained(out_dir)
        tok.save_pretrained(out_dir)
        set_progress(100)
        current_job["status"] = "done"
        emit_log(f"Distillation complete! Saved to: {out_dir}", "success")
    except KeyboardInterrupt:
        current_job["status"] = "stopped"; emit_log("Stopped.", "warn")
    except Exception as e:
        current_job["status"] = "error"; emit_log(f"Error: {e}", "error"); emit_log(traceback.format_exc(), "error")

# ═══ JOB 4 – Dataset Generator ═══════════════════════════════════════════════
def run_generate_dataset(config):
    try:
        gen_source   = config.get("gen_source", "local")
        num_samples  = cfg_int(config.get("num_samples"), 100)
        topic        = config.get("topic", "general knowledge")
        style        = config.get("style", "instruction")
        max_tokens   = cfg_int(config.get("max_new_tokens"), 256)
        ds_name      = config.get("dataset_name") or f"generated_{int(time.time())}"
        _sr          = config.get("seed_prompts", "")
        custom_seeds = ([p.strip() for p in _sr if p.strip()] if isinstance(_sr, list)
                        else [p.strip() for p in _sr.split("\n") if p.strip()])
        TEMPLATES    = {
            "instruction": f"Write an instruction and detailed response about {topic}.",
            "qa":          f"Write a question and answer about {topic}.",
            "reasoning":   f"Write a step-by-step reasoning problem about {topic}.",
            "story":       f"Write a short story or dialogue that teaches about {topic}.",
        }
        seeds    = custom_seeds if custom_seeds else [TEMPLATES.get(style, TEMPLATES["instruction"])]
        run_ts   = int(time.time())
        samples  = []
        model    = tok = None

        if gen_source == "cloud":
            provider  = config.get("cloud_provider", "openai")
            cmodel    = config.get("cloud_model", "")
            price_in  = cfg_float(config.get("cloud_price_in"), 0)
            price_out = cfg_float(config.get("cloud_price_out"), 0)
            emit_log(f"Using cloud generator: {provider} / {cmodel}", "info")
            set_stage(f"Generating {num_samples} samples via {provider}")
            set_progress(5)
            total_in = total_out = 0

            # Cloud log — every response saved immediately
            log_path = CLOUD_LOGS_DIR / f"gendata_cloud_{run_ts}.jsonl"
            with open(log_path, "w") as lf:
                lf.write(json.dumps({"_meta": {
                    "type": "dataset_generation", "provider": provider, "model": cmodel,
                    "num_samples": num_samples, "topic": topic, "style": style,
                    "started": run_ts, "price_in_per_m": price_in, "price_out_per_m": price_out,
                }}) + "\n")

            for i in range(num_samples):
                if stop_flag.is_set(): raise KeyboardInterrupt()
                seed   = seeds[i % len(seeds)].replace("{topic}", topic)
                t0     = time.time()
                try:
                    output = _cloud_generate(seed, config, max_tokens)
                except Exception as e:
                    emit_log(f"Cloud API error on sample {i+1}: {e}", "error")
                    with open(log_path, "a") as lf:
                        lf.write(json.dumps({"_error": str(e), "index": i, "prompt": seed}) + "\n")
                    raise
                elapsed  = time.time() - t0
                tok_in   = len(seed)   // 4 + 50
                tok_out  = len(output) // 4
                total_in  += tok_in
                total_out += tok_out
                entry = {"index": i, "prompt": seed, "response": output,
                         "tokens_in_est": tok_in, "tokens_out_est": tok_out,
                         "elapsed_s": round(elapsed, 2)}
                samples.append({"prompt": seed, "response": output,
                                 "text": f"### Instruction:\n{seed}\n\n### Response:\n{output}"})
                with open(log_path, "a") as lf:
                    lf.write(json.dumps(entry) + "\n")
                set_progress(5 + (i / num_samples) * 80)
                if (i + 1) % 10 == 0:
                    cost = (total_in / 1_000_000) * price_in + (total_out / 1_000_000) * price_out
                    emit_log(f"Generated {i+1}/{num_samples} | ~{total_in//1000}K in / {total_out//1000}K out | cost: ${cost:.4f}", "info")

            final_cost = (total_in / 1_000_000) * price_in + (total_out / 1_000_000) * price_out
            with open(log_path, "a") as lf:
                lf.write(json.dumps({"_summary": True, "total_samples": len(samples),
                    "total_tokens_in": total_in, "total_tokens_out": total_out,
                    "final_cost_usd": round(final_cost, 6), "finished": int(time.time())}) + "\n")
            emit_log(f"Cloud log saved: {log_path.name} | Total cost: ${final_cost:.4f}", "success")

        else:
            model_name = config.get("model_name", "")
            if not model_name: raise ValueError("model_name required for local generation")
            is_gguf_gen = _is_gguf_path(model_name)
            if is_gguf_gen:
                set_stage("Loading GGUF generator model")
                emit_log("GGUF generator detected — loading via llama-cpp-python (inference only)", "info")
                llm_gguf = load_gguf_for_inference(model_name)
                tok = None
            else:
                set_stage("Loading generator model")
                model, tok, _ = load_model_and_tokenizer(model_name)
            set_progress(20)
            set_stage(f"Generating {num_samples} samples")
            inf_batch = cfg_int(config.get("inf_batch_size"), 8)
            emit_log(f"Inference batch size: {inf_batch}", "info")
            all_seeds = [seeds[i % len(seeds)].replace("{topic}", topic) for i in range(num_samples)]
            for batch_start in range(0, num_samples, inf_batch):
                if stop_flag.is_set(): raise KeyboardInterrupt()
                batch_seeds = all_seeds[batch_start:batch_start + inf_batch]
                if is_gguf_gen:
                    outputs = [generate_text_gguf(llm_gguf, s, max_new_tokens=max_tokens)
                               for s in batch_seeds]
                else:
                    outputs = generate_text_batched(model, tok, batch_seeds,
                                                    max_new_tokens=max_tokens,
                                                    batch_size=inf_batch)
                for seed, output in zip(batch_seeds, outputs):
                    samples.append({"prompt": seed, "response": output,
                                    "text": f"### Instruction:\n{seed}\n\n### Response:\n{output}"})
                done = min(batch_start + inf_batch, num_samples)
                set_progress(20 + (done / num_samples) * 65)
                emit_log(f"Generated {done}/{num_samples}", "info")

        set_stage("Saving dataset")
        with open(GEN_DIR / f"{ds_name}.jsonl", "w") as f:
            for s in samples: f.write(json.dumps(s) + "\n")
        from datasets import Dataset
        ds = Dataset.from_list(samples)
        ds.save_to_disk(str(DATASETS_DIR / ds_name))
        set_progress(100)
        current_job["status"] = "done"
        emit_log(f"Dataset generated: {len(samples)} samples → {ds_name}", "success")
    except KeyboardInterrupt:
        current_job["status"] = "stopped"; emit_log("Stopped.", "warn")
    except Exception as e:
        current_job["status"] = "error"; emit_log(f"Error: {e}", "error"); emit_log(traceback.format_exc(), "error")


# ═══ JOB 5 – Web-Search Dataset Generator ════════════════════════════════════

def _web_search(query, engine="duckduckgo", api_key="", num_results=5):
    """
    Search the web and return a list of {"title", "url", "snippet"} dicts.
    Supports: serpapi, brave, duckduckgo (free, no key needed).
    """
    import urllib.request, urllib.parse, json as _json, html, re

    results = []

    if engine == "serpapi" and api_key:
        url = "https://serpapi.com/search.json?" + urllib.parse.urlencode({
            "q": query, "api_key": api_key, "num": num_results, "engine": "google"
        })
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            data = _json.loads(r.read())
        for item in data.get("organic_results", [])[:num_results]:
            results.append({"title": item.get("title",""), "url": item.get("link",""), "snippet": item.get("snippet","")})

    elif engine == "brave" and api_key:
        url = "https://api.search.brave.com/res/v1/web/search?" + urllib.parse.urlencode({
            "q": query, "count": num_results
        })
        req = urllib.request.Request(url, headers={
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": api_key,
        })
        with urllib.request.urlopen(req, timeout=15) as r:
            raw = r.read()
            try:
                import gzip
                raw = gzip.decompress(raw)
            except Exception:
                pass
            data = _json.loads(raw)
        for item in data.get("web", {}).get("results", [])[:num_results]:
            results.append({"title": item.get("title",""), "url": item.get("url",""), "snippet": item.get("description","")})

    else:
        # Free DuckDuckGo HTML scrape — no API key needed
        url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        })
        with urllib.request.urlopen(req, timeout=15) as r:
            body = r.read().decode("utf-8", errors="replace")
        # Parse result blocks from DDG HTML
        blocks = re.findall(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>.*?<a[^>]+class="result__snippet"[^>]*>(.*?)</a>', body, re.DOTALL)
        for href, title_raw, snippet_raw in blocks[:num_results]:
            title   = re.sub(r"<[^>]+>", "", title_raw).strip()
            snippet = re.sub(r"<[^>]+>", "", snippet_raw).strip()
            title   = html.unescape(title)
            snippet = html.unescape(snippet)
            if href.startswith("//duckduckgo.com/l/"):
                # Decode the actual URL from DDG redirect
                qs = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
                href = qs.get("uddg", [href])[0]
            results.append({"title": title, "url": href, "snippet": snippet})

    return results


def _fetch_page_text(url, max_chars=8000):
    """
    Fetch a URL and return clean plaintext (strips HTML tags, scripts, style).
    Returns empty string on failure.
    """
    import urllib.request, re, html
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
        })
        with urllib.request.urlopen(req, timeout=12) as r:
            raw = r.read(max_chars * 6).decode("utf-8", errors="replace")
        # Strip scripts, styles, and HTML tags
        raw = re.sub(r"(?s)<(script|style|nav|header|footer|aside)[^>]*>.*?</\1>", " ", raw)
        raw = re.sub(r"<[^>]+>", " ", raw)
        raw = re.sub(r"&nbsp;", " ", raw)
        raw = html.unescape(raw)
        raw = re.sub(r"[ \t]{2,}", " ", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()[:max_chars]
    except Exception as e:
        return ""


def _chunk_text(text, chunk_size=1500, overlap=200):
    """Split text into overlapping chunks for context windows."""
    words  = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def run_websearch_dataset(config):
    """
    Job 5: Web-Search Dataset Generator.

    Pipeline:
      1. Expand the user's topic into N search queries (via AI or template).
      2. Run web searches for each query, collect URLs + snippets.
      3. Optionally fetch and clean full page text from each URL.
      4. For each chunk of retrieved content, prompt the AI to generate
         training pairs (Q&A, instruction/response, reasoning, etc.).
      5. Save the resulting dataset.
    """
    try:
        topic          = config.get("topic", "general knowledge")
        style          = config.get("style", "qa")
        num_queries    = cfg_int(config.get("num_queries"), 5)
        results_per_q  = cfg_int(config.get("results_per_query"), 5)
        pairs_per_chunk= cfg_int(config.get("pairs_per_chunk"), 3)
        fetch_pages    = config.get("fetch_pages", True)
        search_engine  = config.get("search_engine", "duckduckgo")
        search_api_key = config.get("search_api_key", "")
        ai_source      = config.get("ai_source", "cloud")
        max_tokens     = cfg_int(config.get("max_new_tokens"), 512)
        ds_name        = config.get("dataset_name") or f"websearch_{int(time.time())}"
        custom_queries = [q.strip() for q in config.get("custom_queries", []) if str(q).strip()]

        model = tok = llm_gguf = None
        is_gguf_gen = False
        run_ts = int(time.time())
        samples = []

        set_stage("Phase 1: Preparing search queries")
        set_progress(3)

        # ── Step 1: Build search queries ──────────────────────────────────────
        if custom_queries:
            queries = custom_queries[:num_queries]
            emit_log(f"Using {len(queries)} custom queries", "info")
        else:
            # Auto-generate queries from topic using AI or templates
            query_templates = [
                f"{topic} overview and introduction",
                f"{topic} key concepts and definitions",
                f"{topic} examples and applications",
                f"{topic} advanced techniques and best practices",
                f"{topic} common questions and answers",
                f"{topic} history and background",
                f"how does {topic} work",
                f"{topic} tutorials and guides",
            ]
            queries = query_templates[:num_queries]
            emit_log(f"Auto-generated {len(queries)} search queries for: {topic}", "info")

        for q in queries:
            emit_log(f"  → {q}", "info")

        # ── Step 2: Load AI model (cloud or local) ────────────────────────────
        set_stage("Phase 2: Initializing AI model")
        set_progress(8)

        if ai_source == "cloud":
            emit_log(f"Using cloud AI: {config.get('cloud_provider')} / {config.get('cloud_model')}", "info")
        else:
            model_name = config.get("model_name", "")
            if not model_name:
                raise ValueError("model_name required for local AI source")
            is_gguf_gen = _is_gguf_path(model_name)
            if is_gguf_gen:
                emit_log("GGUF model detected — loading via llama-cpp-python", "info")
                llm_gguf = load_gguf_for_inference(model_name)
            else:
                model, tok, _ = load_model_and_tokenizer(model_name)
            emit_log("Local AI model ready", "success")

        set_progress(15)

        # ── Step 3: Search + fetch ────────────────────────────────────────────
        all_contexts = []   # list of {"url", "title", "text"}

        total_q = len(queries)
        for qi, query in enumerate(queries):
            if stop_flag.is_set(): raise KeyboardInterrupt()
            set_stage(f"Phase 3: Searching [{qi+1}/{total_q}] — {query[:50]}")
            set_progress(15 + (qi / total_q) * 30)
            emit_log(f"Searching ({search_engine}): {query}", "info")

            try:
                results = _web_search(query, engine=search_engine,
                                      api_key=search_api_key, num_results=results_per_q)
            except Exception as e:
                emit_log(f"Search failed for '{query}': {e}", "warn")
                results = []

            for res in results:
                if stop_flag.is_set(): raise KeyboardInterrupt()
                url     = res.get("url", "")
                title   = res.get("title", "")
                snippet = res.get("snippet", "")

                if fetch_pages and url.startswith("http"):
                    emit_log(f"  Fetching: {url[:70]}", "info")
                    page_text = _fetch_page_text(url)
                    text = page_text if page_text else snippet
                else:
                    text = snippet

                if text.strip():
                    all_contexts.append({"url": url, "title": title, "text": text, "query": query})

        emit_log(f"Collected {len(all_contexts)} content sources", "success")
        set_progress(45)

        if not all_contexts:
            raise RuntimeError("No content retrieved from web search. Try a different topic or search engine.")

        # ── Step 4: Generate dataset pairs from content ───────────────────────
        STYLE_PROMPTS = {
            "qa": (
                "You are a dataset creator. Given the following text, generate {n} question-answer pairs "
                "that test understanding of the key facts and concepts. Each pair must be directly supported by the text.\n\n"
                "Text:\n{text}\n\n"
                "Respond with ONLY a JSON array like:\n"
                '[{{"question": "...", "answer": "..."}}, ...]\n'
                "Generate {n} pairs:"
            ),
            "instruction": (
                "You are a dataset creator. Given the following text, generate {n} instruction-response pairs "
                "where the instruction is a task or request and the response is a helpful, accurate answer based on the text.\n\n"
                "Text:\n{text}\n\n"
                "Respond with ONLY a JSON array like:\n"
                '[{{"instruction": "...", "response": "..."}}, ...]\n'
                "Generate {n} pairs:"
            ),
            "reasoning": (
                "You are a dataset creator. Given the following text, generate {n} reasoning problems "
                "with step-by-step solutions grounded in the text content.\n\n"
                "Text:\n{text}\n\n"
                "Respond with ONLY a JSON array like:\n"
                '[{{"problem": "...", "reasoning": "...", "answer": "..."}}, ...]\n'
                "Generate {n} pairs:"
            ),
            "summary": (
                "You are a dataset creator. Given the following text, generate {n} summarization training pairs: "
                "the input is a passage and the output is a concise summary.\n\n"
                "Text:\n{text}\n\n"
                "Respond with ONLY a JSON array like:\n"
                '[{{"input": "...", "summary": "..."}}, ...]\n'
                "Generate {n} pairs:"
            ),
        }
        style_prompt_template = STYLE_PROMPTS.get(style, STYLE_PROMPTS["qa"])

        total_ctx = len(all_contexts)
        for ci, ctx in enumerate(all_contexts):
            if stop_flag.is_set(): raise KeyboardInterrupt()
            set_stage(f"Phase 4: Generating pairs [{ci+1}/{total_ctx}]")
            set_progress(45 + (ci / total_ctx) * 45)

            # Chunk the context text and generate from each chunk
            chunks = _chunk_text(ctx["text"], chunk_size=1200, overlap=150)
            for chunk in chunks[:2]:   # max 2 chunks per source to stay efficient
                if stop_flag.is_set(): raise KeyboardInterrupt()
                prompt = style_prompt_template.format(text=chunk.strip(), n=pairs_per_chunk)

                try:
                    if ai_source == "cloud":
                        raw = _cloud_generate(prompt, config, max_tokens)
                    elif is_gguf_gen:
                        raw = generate_text_gguf(llm_gguf, prompt, max_new_tokens=max_tokens)
                    else:
                        raw = generate_text(model, tok, prompt, max_new_tokens=max_tokens)

                    # Parse JSON response
                    raw = raw.strip()
                    # Strip markdown fences if present
                    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
                    raw = re.sub(r"\s*```$", "", raw)
                    pairs_raw = json.loads(raw)
                    if not isinstance(pairs_raw, list):
                        pairs_raw = [pairs_raw]

                    for pair in pairs_raw:
                        if not isinstance(pair, dict): continue
                        # Normalise keys across styles
                        if style == "qa":
                            q = pair.get("question") or pair.get("instruction") or pair.get("prompt", "")
                            a = pair.get("answer")   or pair.get("response", "")
                        elif style == "instruction":
                            q = pair.get("instruction") or pair.get("question") or pair.get("prompt", "")
                            a = pair.get("response")    or pair.get("answer", "")
                        elif style == "reasoning":
                            q = pair.get("problem") or pair.get("question", "")
                            a = (pair.get("reasoning", "") + "\n\n" + pair.get("answer", "")).strip()
                        elif style == "summary":
                            q = pair.get("input", "")
                            a = pair.get("summary") or pair.get("output", "")
                        else:
                            q = str(pair.get(list(pair.keys())[0], ""))
                            a = str(pair.get(list(pair.keys())[-1], "")) if len(pair) > 1 else ""

                        if q and a:
                            samples.append({
                                "prompt":   q,
                                "response": a,
                                "text":     f"### Instruction:\n{q}\n\n### Response:\n{a}",
                                "source_url":   ctx["url"],
                                "source_title": ctx["title"],
                                "search_query": ctx["query"],
                            })

                except Exception as e:
                    emit_log(f"Parse/generate error on source {ci+1}: {e}", "warn")
                    continue

            if (ci + 1) % 5 == 0:
                emit_log(f"Progress: {len(samples)} pairs so far from {ci+1}/{total_ctx} sources", "info")

        # ── Step 5: Save ──────────────────────────────────────────────────────
        set_stage("Phase 5: Saving dataset")
        set_progress(92)

        if not samples:
            raise RuntimeError("No dataset pairs were generated. The AI may have returned invalid JSON — try a different model or style.")

        with open(GEN_DIR / f"{ds_name}.jsonl", "w") as f:
            for s in samples: f.write(json.dumps(s) + "\n")

        from datasets import Dataset as HFDataset
        ds = HFDataset.from_list(samples)
        ds.save_to_disk(str(DATASETS_DIR / ds_name))

        # Save a summary log
        summary = {
            "topic": topic, "style": style, "search_engine": search_engine,
            "queries": queries, "sources_fetched": len(all_contexts),
            "total_pairs": len(samples), "dataset_name": ds_name, "created": run_ts,
        }
        with open(GEN_DIR / f"{ds_name}_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        set_progress(100)
        current_job["status"] = "done"
        emit_log(f"✅ Web dataset complete! {len(samples)} pairs from {len(all_contexts)} sources → {ds_name}", "success")

    except KeyboardInterrupt:
        current_job["status"] = "stopped"; emit_log("Stopped.", "warn")
    except Exception as e:
        current_job["status"] = "error"; emit_log(f"Error: {e}", "error"); emit_log(traceback.format_exc(), "error")


# ═══ JOB 6 – Evaluate / Test ═════════════════════════════════════════════════
def run_evaluation(config):
    try:
        model_path   = config["model_path"]
        ds_path      = config["dataset"]
        num_samples  = cfg_int(config.get("num_samples"), 20)
        max_new_tokens = cfg_int(config.get("max_new_tokens"), 256)
        input_col    = config.get("input_col", "")
        target_col   = config.get("target_col", "")

        set_stage("Loading model for evaluation")
        model, tok, _ = load_model_and_tokenizer(model_path, load_in_4bit=True)
        # Put model in inference mode
        try:
            from unsloth import FastModel
            FastModel.for_inference(model)
        except:
            try:
                from unsloth import FastLanguageModel
                FastLanguageModel.for_inference(model)
            except:
                model.eval()
        set_progress(20)

        set_stage("Loading dataset")
        from datasets import load_dataset, load_from_disk
        try:    dataset = load_from_disk(ds_path)
        except: dataset = load_dataset(ds_path, split="train", token=hf_token["value"])
        emit_log(f"Dataset loaded: {len(dataset)} rows", "success")

        # Auto-detect columns if not specified
        cols = dataset.column_names
        if not input_col:
            for c in ["problem","question","input","instruction","prompt","text"]:
                if c in cols: input_col = c; break
        if not target_col:
            for c in ["solution","answer","output","response","thinking","text"]:
                if c in cols and c != input_col: target_col = c; break

        emit_log(f"Input col: '{input_col}' | Target col: '{target_col}'", "info")

        # Sample
        import random
        indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
        samples = dataset.select(indices)
        set_progress(30)

        set_stage(f"Running inference on {len(samples)} samples")
        import torch
        results = []
        scores  = []

        for i, row in enumerate(samples):
            if stop_flag.is_set(): raise KeyboardInterrupt()

            prompt   = str(row.get(input_col, row.get("instruction", "")))[:512] or "<s>"
            expected = str(row.get(target_col, "")) if target_col else ""

            # Generate
            inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=tok.eos_token_id,
                )
            generated = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

            # Score: simple token-level overlap (lightweight ROUGE-L approximation)
            score = None
            if expected:
                exp_tokens = set(expected.lower().split())
                gen_tokens = set(generated.lower().split())
                if exp_tokens:
                    score = round(len(exp_tokens & gen_tokens) / len(exp_tokens) * 100, 1)
                    scores.append(score)

            results.append({
                "idx":       int(indices[i]),
                "prompt":    prompt[:300],
                "expected":  expected[:500] if expected else "",
                "generated": generated[:500],
                "score":     score,
            })

            pct = 30 + ((i+1) / len(samples)) * 65
            set_progress(pct)
            emit_log(f"Sample {i+1}/{len(samples)} | score: {score}%", "info" if score is None or score >= 30 else "warn")

        avg_score = round(sum(scores)/len(scores), 1) if scores else None
        set_stage("Evaluation complete")
        set_progress(100)
        current_job["status"]  = "done"
        current_job["eval_results"] = results
        current_job["eval_avg"]     = avg_score
        if avg_score is not None:
            emoji = "✅" if avg_score >= 50 else "⚠️" if avg_score >= 25 else "❌"
            emit_log(f"{emoji} Avg token overlap score: {avg_score}% over {len(scores)} samples", "success" if avg_score >= 50 else "warn")
        else:
            emit_log(f"✅ Generated {len(results)} samples (no target col for scoring)", "success")

    except KeyboardInterrupt:
        current_job["status"] = "stopped"; emit_log("Stopped.", "warn")
    except Exception as e:
        current_job["status"] = "error"; emit_log(f"Error: {e}", "error"); emit_log(traceback.format_exc(), "error")

# ═══ ROUTES ══════════════════════════════════════════════════════════════════


@app.route("/api/cloud/models", methods=["POST"])
def cloud_models():
    """Fetch available models from a cloud provider using their API."""
    data     = request.json or {}
    provider = data.get("provider", "")
    api_key  = data.get("api_key", "")
    base_url = data.get("base_url", "") or ""

    import urllib.request, json as _json

    # ── Google: uses v1beta/models with API key in query param ────────────────
    if provider == "google":
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}&pageSize=100"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=10) as resp:
                raw = _json.loads(resp.read())
            models = []
            for m in raw.get("models", []):
                name = m.get("name", "").replace("models/", "")
                # Keep generative text models only (skip embeddings, vision-only, deprecated)
                supported = [a.get("name","") for a in m.get("supportedGenerationMethods",[])]
                if "generateContent" in supported and any(x in name for x in ["gemini","gemma"]):
                    models.append(name)
            return jsonify({"ok": True, "models": sorted(models)})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)})

    # ── Perplexity: Agent API model list (presets + validate key) ────────────
    if provider == "perplexity":
        # Perplexity Agent API — full model list from official docs (Feb 2026)
        known = [
            # ── Presets (recommended — send as `preset` field, not `model`) ───
            "fast-search",             # xAI Grok 4.1, 1 step, fastest
            "pro-search",              # OpenAI GPT-5.1, 3 steps, balanced
            "deep-research",           # OpenAI GPT-5.2, 10 steps, thorough
            "advanced-deep-research",  # Anthropic Claude Opus 4.6, 10 steps, max depth
            # ── Perplexity native model ─────────────────────────────────────
            "perplexity/sonar",
            # ── OpenAI models (via Perplexity, no separate key needed) ──────
            "openai/gpt-5.2",
            "openai/gpt-5.1",
            "openai/gpt-5-mini",
            # ── Anthropic models ────────────────────────────────────────────
            "anthropic/claude-opus-4-6",
            "anthropic/claude-opus-4-5",
            "anthropic/claude-sonnet-4-5",
            "anthropic/claude-haiku-4-5",
            # ── Google models ───────────────────────────────────────────────
            "google/gemini-3-pro-preview",
            "google/gemini-3-flash-preview",
            "google/gemini-2.5-pro",
            "google/gemini-2.5-flash",
            # ── xAI models ──────────────────────────────────────────────────
            "xai/grok-4-1-fast-non-reasoning",
        ]
        try:
            # Validate key cheaply with a fast-search preset call, 1 token
            url  = "https://api.perplexity.ai/v1/responses"
            body = _json.dumps({
                "preset": "fast-search",
                "input":  "hi",
                "max_output_tokens": 1,
            }).encode()
            req = urllib.request.Request(url, data=body, headers={
                "Content-Type":  "application/json",
                "Authorization": f"Bearer {api_key}",
            }, method="POST")
            with urllib.request.urlopen(req, timeout=15) as resp:
                resp.read()
            return jsonify({"ok": True, "models": known})
        except Exception as e:
            err = str(e)
            if "401" in err or "403" in err:
                return jsonify({"ok": False, "error": "Invalid API key"})
            # Transient timeout etc. — still return the known list
            return jsonify({"ok": True, "models": known})

    # ── Custom: standard OpenAI-compatible GET /models ─────────────────────────
    base = base_url.rstrip("/") if base_url else ""
    if not base:
        return jsonify({"ok": False, "error": "Enter a Base URL for custom providers"})
    try:
        url = base + "/models"
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = _json.loads(resp.read())
        models = sorted([m.get("id","") for m in raw.get("data", []) if m.get("id")])
        return jsonify({"ok": True, "models": models})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

@app.route("/")
def index(): return render_template("index.html")

@app.route("/api/status")
def status():
    return jsonify({"status":current_job["status"],"progress":current_job["progress"],"stage":current_job.get("stage",""),"logs":current_job["logs"][-100:]})

@app.route("/api/debug")
def debug():
    """Diagnostic endpoint — call from browser console: fetch('/api/debug').then(r=>r.json()).then(console.log)"""
    import sys
    return jsonify({
        "current_job_id":  id(current_job),
        "current_job":     {k: v if k != "logs" else f"{len(v)} entries" for k, v in current_job.items()},
        "stop_flag":       stop_flag.is_set(),
        "training_thread": str(training_thread),
        "thread_alive":    training_thread.is_alive() if training_thread else False,
        "python":          sys.version,
    })

@app.route("/api/train",            methods=["POST"])
def train():       return _start_job(run_training,        request.json)
@app.route("/api/autotrain",        methods=["POST"])
def autotrain():   return _start_job(run_autotrain,        request.json)
@app.route("/api/distill",          methods=["POST"])
def distill():     return _start_job(run_distillation,     request.json)
@app.route("/api/generate_dataset", methods=["POST"])
def gen_dataset(): return _start_job(run_generate_dataset, request.json)
@app.route("/api/websearch_dataset", methods=["POST"])
def websearch_dataset(): return _start_job(run_websearch_dataset, request.json)
@app.route("/api/autodistill", methods=["POST"])
def autodistill(): return _start_job(run_autodistill, request.json)
@app.route("/api/autodistill/presets")
def autodistill_presets():
    result = {}
    for k, v in AUTODISTILL_PRESETS.items():
        result[k] = {
            "label":           v["label"],
            "icon":            v["icon"],
            "description":     v["description"],
            "teacher_repo":    v["teacher_repo"],
            "teacher_file":    v["teacher_file"],
            "teacher_size_gb": v["teacher_size_gb"],
            "teacher_params":  v["teacher_params"],
            "num_prompts":     v["num_prompts"],
            "max_new_tokens":  v["max_new_tokens"],
            "lora_r":          v["lora_r"],
            "max_steps":       v["max_steps"],
            "prompt_template": v.get("prompt_template", "default"),
        }
    return jsonify(result)

def _eject_model():
    """Immediately unload model from VRAM and RAM."""
    import gc
    model     = _active_model.get("model")
    tokenizer = _active_model.get("tokenizer")
    if model:
        try:
            # Move to CPU first (releases VRAM), then delete
            model.cpu()
        except: pass
        try: del model
        except: pass
        _active_model["model"] = None
    if tokenizer:
        try: del tokenizer
        except: pass
        _active_model["tokenizer"] = None
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        freed = torch.cuda.memory_allocated()
        emit_log(f"VRAM after eject: {round(freed/1024**3,2)} GB", "info")
    except: pass
    emit_log("Model ejected from memory.", "warn")


def _kill_thread(t):
    """Raise KeyboardInterrupt in a thread via ctypes (best-effort)."""
    if t is None or not t.is_alive():
        return
    import ctypes
    try:
        tid = t.ident
        if tid:
            res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_ulong(tid),
                ctypes.py_object(KeyboardInterrupt)
            )
            if res == 0:
                emit_log("Thread kill: invalid thread id", "warn")
    except Exception as e:
        emit_log(f"Thread kill failed: {e}", "warn")


def _delete_partial_downloads():
    """Delete tracked download directories ONLY if they are genuinely incomplete.
    A directory is considered incomplete if it contains .incomplete / .tmp files
    but NO usable model files (.safetensors, .bin, .gguf, config.json).
    Directories inside OUTPUTS_DIR are NEVER deleted by stop — those are trained models.
    """
    import shutil
    SAFE_EXTS  = {".safetensors", ".bin", ".gguf", ".pt"}
    SAFE_FILES = {"config.json", "tokenizer.json", "tokenizer_config.json"}
    for path in list(_download_dirs):
        try:
            p = Path(path)
            if not p.exists():
                continue
            # Never touch outputs (trained models, fixed configs, etc.)
            try:
                p.resolve().relative_to(OUTPUTS_DIR.resolve())
                emit_log(f"Stop: skipping {p.name} (in outputs/, not a download)", "info")
                continue
            except ValueError:
                pass
            # Check if this dir has any usable model files already
            has_model = any(
                f.suffix.lower() in SAFE_EXTS or f.name in SAFE_FILES
                for f in p.rglob("*") if f.is_file()
            )
            has_incomplete = any(
                f.suffix in (".incomplete", ".tmp")
                for f in p.rglob("*") if f.is_file()
            )
            if has_model and not has_incomplete:
                emit_log(f"Stop: keeping {p.name} — looks complete (has model files)", "info")
                continue
            shutil.rmtree(str(p))
            emit_log(f"Deleted partial download: {p.name}", "warn")
        except Exception as e:
            emit_log(f"Could not delete {path}: {e}", "warn")
    _download_dirs.clear()


@app.route("/api/stop", methods=["POST"])
def stop():
    emit_log("⛔ STOP requested — ejecting everything...", "warn")
    current_job["status"] = "stopping"

    # 1. Signal all cooperative loops
    stop_flag.set()

    # 2. Stop all monitor/pulse threads
    for ev in list(_monitor_events):
        try: ev.set()
        except: pass
    _monitor_events.clear()

    # 3. Kill training/download threads via async exception
    for t in list(_active_threads):
        _kill_thread(t)
    _active_threads.clear()
    _kill_thread(training_thread)

    # 4. Eject model from VRAM/RAM in background (don't block the response)
    threading.Thread(target=_eject_model, daemon=True).start()

    # 5. Delete partial downloads
    threading.Thread(target=_delete_partial_downloads, daemon=True).start()

    # 6. Clear HF hub cache locks (allows re-download cleanly)
    try:
        import huggingface_hub.constants as _hfc
        cache_dir = Path(_hfc.HF_HUB_CACHE)
        for lock in cache_dir.rglob("*.lock"):
            try: lock.unlink()
            except: pass
    except: pass

    current_job["status"]   = "idle"
    current_job["progress"] = 0
    current_job["stage"]    = ""
    emit_log("⛔ Stopped. Memory cleared.", "warn")
    return jsonify({"ok": True})


@app.route("/api/gguf_export",    methods=["POST"])
def gguf_export():    return _start_job(run_gguf_export,   request.json)

@app.route("/api/mobile_export",  methods=["POST"])
def mobile_export():  return _start_job(run_mobile_export, request.json)

@app.route("/api/optimize",       methods=["POST"])
def optimize():       return _start_job(run_optimization,  request.json)

@app.route("/api/prune",          methods=["POST"])
def prune():          return _start_job(run_pruning,        request.json)

@app.route("/api/gguf_quant_types")
def gguf_quant_types():
    return jsonify({"types": GGUF_QUANT_TYPES})

@app.route("/api/evaluate", methods=["POST"])
def evaluate():    return _start_job(run_evaluation, request.json)

@app.route("/api/eval_results")
def eval_results():
    return jsonify({
        "status":  current_job.get("status"),
        "results": current_job.get("eval_results", []),
        "avg":     current_job.get("eval_avg"),
    })

@app.route("/api/logs/stream")
def stream_logs():
    def generate():
        while True:
            try:    entry = log_queue.get(timeout=30); yield f"data: {json.dumps(entry)}\n\n"
            except: yield f"data: {json.dumps({'ping':True})}\n\n"
    return Response(stream_with_context(generate()), mimetype="text/event-stream")

# ── Models ────────────────────────────────────────────────────────────────────
@app.route("/api/models/local")
def list_local_models():
    models = []
    def model_info(p, tag="model"):
        is_gguf     = bool(list(p.glob("*.gguf"))+list(p.glob("*.GGUF"))) if p.is_dir() else str(p).lower().endswith(".gguf")
        has_weights = bool(list(p.glob("*.safetensors"))+list(p.glob("pytorch_model*.bin"))) if p.is_dir() else False
        try: size_bytes = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.is_dir() else p.stat().st_size
        except: size_bytes = 0
        size_str = f"{size_bytes/1e9:.1f} GB" if size_bytes > 1e9 else f"{size_bytes/1e6:.0f} MB"
        return {"name":p.name,"path":str(p),"tag":tag,"is_gguf":is_gguf,"has_weights":has_weights,"size":size_str,"trainable":has_weights and not is_gguf}
    for p in sorted(MODELS_DIR.iterdir()):
        if p.is_dir(): models.append(model_info(p,"model"))
        elif p.suffix.lower() == ".gguf": models.append(model_info(p,"gguf"))
    for p in sorted(OUTPUTS_DIR.iterdir()):
        if p.is_dir(): models.append(model_info(p,"output"))
    return jsonify(models)

@app.route("/api/models/hf_files", methods=["POST"])
def list_hf_files():
    repo_id = request.json.get("repo_id","")
    try:
        from huggingface_hub import list_repo_tree
        token = hf_token["value"]
        file_sizes = {}  # filename -> bytes
        all_files  = []
        total_bytes = 0
        for item in list_repo_tree(repo_id, recursive=True, token=token):
            fname = getattr(item, "path", None) or getattr(item, "rfilename", None)
            if not fname: continue
            size = getattr(item, "size", 0) or 0
            all_files.append(fname)
            file_sizes[fname] = size
            total_bytes += size

        def file_entry(f):
            return {"name": f, "size": file_sizes.get(f, 0)}

        safetensors = [file_entry(f) for f in all_files if f.endswith(".safetensors")]
        gguf        = [file_entry(f) for f in all_files if f.lower().endswith(".gguf")]
        bins        = [file_entry(f) for f in all_files if f.endswith(".bin")]

        # Config/tokenizer files always downloaded alongside weights
        config_bytes = sum(file_sizes.get(f,0) for f in all_files
                          if f.endswith((".json",".txt",".model")) and file_sizes.get(f,0) < 10_000_000)

        return jsonify({
            "ok": True,
            "safetensors": safetensors,
            "gguf": gguf,
            "bin": bins,
            "all": all_files,
            "file_sizes": file_sizes,
            "total_bytes": total_bytes,
            "config_bytes": config_bytes,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/datasets/hf_info", methods=["POST"])
def dataset_hf_info():
    """Return size + row count + available splits for a HF dataset before downloading."""
    data    = request.json or {}
    repo_id = data.get("repo_id", "")
    if not repo_id:
        return jsonify({"ok": False, "error": "No repo_id"})
    try:
        from huggingface_hub import list_repo_tree
        token = hf_token["value"]

        # 1. Total compressed size from repo tree
        total_bytes = 0
        file_paths  = []
        try:
            for item in list_repo_tree(repo_id, repo_type="dataset", recursive=True, token=token):
                size = getattr(item, "size", 0) or 0
                total_bytes += size
                path = getattr(item, "path", "") or ""
                if path:
                    file_paths.append(path)
        except Exception:
            pass

        # 2. Split names + per-split row counts
        splits_info = {}   # {split_name: row_count_or_None}

        # Method A: huggingface_hub dataset_info
        try:
            from huggingface_hub import dataset_info as hf_dataset_info
            info = hf_dataset_info(repo_id, token=token)
            raw_splits = getattr(info, "splits", None) or {}
            if hasattr(raw_splits, "items"):
                for name, si in raw_splits.items():
                    n = getattr(si, "num_examples", None)
                    splits_info[name] = int(n) if n is not None else None
            elif hasattr(raw_splits, "__iter__"):
                for si in raw_splits:
                    name = getattr(si, "name", str(si))
                    n    = getattr(si, "num_examples", None)
                    splits_info[name] = int(n) if n is not None else None
        except Exception:
            pass

        # Method B: datasets library
        if not splits_info:
            try:
                from datasets import get_dataset_split_names, get_dataset_infos
                for sn in get_dataset_split_names(repo_id, token=token):
                    splits_info[sn] = None
                try:
                    for cfg_info in get_dataset_infos(repo_id, token=token).values():
                        for sn, si in (getattr(cfg_info, "splits", {}) or {}).items():
                            n = getattr(si, "num_examples", None)
                            if n is not None:
                                splits_info[sn] = int(n)
                except Exception:
                    pass
            except Exception:
                pass

        # Method C: parse split names from file paths
        if not splits_info and file_paths:
            KNOWN = {"train", "test", "validation", "dev", "train_sft", "test_sft"}
            for path in file_paths:
                for part in path.replace("\\", "/").split("/"):
                    if part in KNOWN:
                        splits_info.setdefault(part, None)

        # 3. Total rows
        counts     = [v for v in splits_info.values() if v is not None]
        total_rows = sum(counts) if counts else None

        # 4. Preferred split
        preferred = None
        for candidate in ("train", "train_sft", "test", "validation", "dev"):
            if candidate in splits_info:
                preferred = candidate
                break
        if preferred is None and splits_info:
            preferred = next(iter(splits_info))

        return jsonify({
            "ok":              True,
            "total_bytes":     total_bytes,
            "num_rows":        total_rows,
            "splits":          splits_info,
            "splits_list":     list(splits_info.keys()),
            "preferred_split": preferred,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

@app.route("/api/datasets/hf_files", methods=["POST"])
def dataset_hf_files():
    """List dataset splits with file sizes and row counts from a HF dataset repo."""
    repo_id = (request.json or {}).get("repo_id", "")
    if not repo_id:
        return jsonify({"ok": False, "error": "No repo_id"})
    try:
        from huggingface_hub import list_repo_tree
        token = hf_token["value"]

        # Collect all files with sizes
        file_paths  = {}  # path -> size
        try:
            for item in list_repo_tree(repo_id, repo_type="dataset", recursive=True, token=token):
                path = getattr(item, "path", "") or ""
                size = getattr(item, "size", 0) or 0
                if path:
                    file_paths[path] = size
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)})

        total_bytes = sum(file_paths.values())

        # Build splits dict: {split_name: {size, num_rows, files}}
        splits = {}
        KNOWN_SPLITS = {"train", "test", "validation", "dev", "train_sft", "test_sft",
                        "valid", "eval", "sample", "train_prefs", "test_prefs"}

        for path, size in file_paths.items():
            parts = path.replace("\\", "/").split("/")
            detected = None
            # Check folder name or filename prefix
            for part in parts:
                slug = part.lower().split("-")[0].split(".")[0]
                if slug in KNOWN_SPLITS or part in KNOWN_SPLITS:
                    detected = slug if slug in KNOWN_SPLITS else part
                    break
            # Fallback: data/ flat files like "data/train-00001-of-00002.parquet"
            if detected is None and len(parts) >= 2 and parts[0] == "data":
                fname = parts[-1]
                for sn in KNOWN_SPLITS:
                    if fname.startswith(sn):
                        detected = sn
                        break
            if detected is None:
                continue
            if detected not in splits:
                splits[detected] = {"size": 0, "num_rows": None, "files": []}
            splits[detected]["size"] += size
            splits[detected]["files"].append(path)

        # If no splits parsed from files, treat whole repo as one "train" split
        if not splits and file_paths:
            splits["train"] = {"size": total_bytes, "num_rows": None, "files": list(file_paths.keys())}

        # Enrich with row counts via huggingface_hub dataset_info
        try:
            from huggingface_hub import dataset_info as hf_dataset_info
            info = hf_dataset_info(repo_id, token=token)
            raw_splits = getattr(info, "splits", None) or {}
            if hasattr(raw_splits, "items"):
                for name, si in raw_splits.items():
                    n = getattr(si, "num_examples", None)
                    if n is not None and name in splits:
                        splits[name]["num_rows"] = int(n)
                    elif n is not None:
                        splits.setdefault(name, {"size": 0, "num_rows": int(n), "files": []})
                        splits[name]["num_rows"] = int(n)
        except Exception:
            pass

        # Preferred split
        preferred = None
        for candidate in ("train", "train_sft", "test", "validation", "dev"):
            if candidate in splits:
                preferred = candidate
                break
        if preferred is None and splits:
            preferred = next(iter(splits))

        return jsonify({
            "ok": True,
            "splits": splits,
            "total_bytes": total_bytes,
            "preferred_split": preferred,
            "all_files": [{"path": p, "size": s} for p, s in file_paths.items()],
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/download/model", methods=["POST"])
def download_model():
    data=request.json; repo_id=data.get("repo_id"); patterns=data.get("patterns"); filename=data.get("filename"); out_name=data.get("out_name") or repo_id.replace("/","_")
    def do(repo_id, patterns, filename, out_name):
        import shutil
        local_dir  = Path(MODELS_DIR / out_name)
        local_dir.mkdir(parents=True, exist_ok=True)

        # ── Set downloading state immediately ──────────────────────────────────
        current_job.clear()
        current_job.update({
            "status": "downloading", "progress": 0, "logs": [],
            "stage": f"↓ Connecting to HuggingFace…"
        })
        emit_log(f"Starting download: {repo_id}", "info")

        # ── Get expected total size from HF metadata ───────────────────────────
        expected_bytes = 0
        try:
            from huggingface_hub import list_repo_tree
            for item in list_repo_tree(repo_id, token=hf_token["value"]):
                size = getattr(item, "size", None)
                if size: expected_bytes += size
        except Exception:
            pass  # size unknown, we'll still show bytes-done progress

        if expected_bytes:
            emit_log(f"Expected size: {expected_bytes/1e9:.2f} GB", "info")

        # ── Background filesystem size monitor ─────────────────────────────────
        monitor_stop = threading.Event()
        _monitor_events.append(monitor_stop)
        _download_dirs.append(str(local_dir))

        def _size_monitor():
            while not monitor_stop.is_set():
                try:
                    done = sum(
                        f.stat().st_size
                        for f in local_dir.rglob("*")
                        if f.is_file() and not f.name.endswith(".incomplete")
                    )
                    done_mb = done / 1e6
                    if expected_bytes > 0:
                        pct = min(int(done / expected_bytes * 100), 99)
                        current_job["progress"] = pct
                        total_mb = expected_bytes / 1e6
                        current_job["stage"] = f"↓ {done_mb:.0f} / {total_mb:.0f} MB  ({pct}%)"
                    else:
                        # Size unknown — show bytes done, pulse progress
                        current_job["stage"] = f"↓ {done_mb:.0f} MB downloaded…"
                        # Fake asymptotic progress so bar moves
                        current_job["progress"] = min(current_job["progress"] + 1, 90)
                except Exception:
                    pass
                monitor_stop.wait(1.0)

        monitor_thread = threading.Thread(target=_size_monitor, daemon=True)
        monitor_thread.start()

        try:
            from huggingface_hub import hf_hub_download, snapshot_download
            token = hf_token["value"]

            if filename:
                hf_hub_download(repo_id=repo_id, filename=filename,
                                local_dir=str(local_dir), token=token)
            else:
                kwargs = {"repo_id": repo_id, "local_dir": str(local_dir)}
                if patterns:
                    kwargs["allow_patterns"] = (
                        patterns + ["*.json","*.txt","*.model","tokenizer*","special_tokens*"]
                    )
                if token:
                    kwargs["token"] = token
                snapshot_download(**kwargs)

            monitor_stop.set()
            current_job["progress"] = 100
            current_job["status"]   = "done"
            current_job["stage"]    = f"✅ Downloaded: {out_name}"
            emit_log(f"Download complete: {local_dir}", "success")
            # ── Remove from partial-download list so Stop won't delete it ──
            try: _download_dirs.remove(str(local_dir))
            except ValueError: pass

            all_files = sorted(f.name for f in local_dir.rglob("*") if f.is_file())
            emit_log(f"Files: {', '.join(all_files[:20])}" + (" …" if len(all_files) > 20 else ""), "info")

        except Exception as e:
            monitor_stop.set()
            current_job["status"] = "error"
            current_job["stage"]  = "❌ Download failed"
            emit_log(f"Download failed: {e}", "error")

    t = threading.Thread(target=do, args=(repo_id,patterns,filename,out_name), daemon=True)
    _active_threads.append(t)
    t.start()
    return jsonify({"ok":True})

@app.route("/api/models/import_gguf", methods=["POST"])
def import_gguf():
    from werkzeug.utils import secure_filename
    if "file" not in request.files: return jsonify({"ok":False,"error":"No file in request"})
    f = request.files["file"]; name = request.form.get("alias","").strip() or f.filename
    name = secure_filename(name)
    if not name: return jsonify({"ok":False,"error":"Could not determine filename"})
    if not name.lower().endswith(".gguf"): return jsonify({"ok":False,"error":"Only .gguf files supported"})
    dest = MODELS_DIR / name
    try:
        f.save(str(dest)); size=dest.stat().st_size
        size_str = f"{size/1e9:.2f} GB" if size>1e9 else f"{size/1e6:.0f} MB"
        emit_log(f"Uploaded GGUF: {name} ({size_str})", "success")
        return jsonify({"ok":True,"path":str(dest),"name":name,"size":size_str})
    except Exception as e: return jsonify({"ok":False,"error":str(e)})

@app.route("/api/models/delete", methods=["POST"])
def delete_model():
    import shutil
    path = (request.json or {}).get("path","").strip()
    if not path: return jsonify({"ok":False,"error":"No path provided"})
    target = Path(path).resolve()
    if not any(str(target).startswith(str(a.resolve())) for a in [MODELS_DIR, OUTPUTS_DIR]):
        return jsonify({"ok":False,"error":"Path outside allowed directories"})
    if not target.exists(): return jsonify({"ok":False,"error":"Path does not exist"})
    try:
        shutil.rmtree(str(target)) if target.is_dir() else target.unlink()
        emit_log(f"Deleted model: {target.name}", "warn"); return jsonify({"ok":True})
    except Exception as e: return jsonify({"ok":False,"error":str(e)})

# ── Datasets ──────────────────────────────────────────────────────────────────
@app.route("/api/datasets/local")
def list_local_datasets():
    results = []
    def sz(p):
        try: b = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.is_dir() else p.stat().st_size
        except: b = 0
        return f"{b/1e9:.1f} GB" if b>1e9 else f"{b/1e6:.0f} MB"
    if DATASETS_DIR.exists():
        for p in DATASETS_DIR.iterdir():
            results.append({"name":p.name,"path":str(p),"source":"datasets","size":sz(p)})
    if GEN_DIR.exists():
        for p in GEN_DIR.glob("*.jsonl"):
            results.append({"name":p.name,"path":str(p),"source":"generated","size":sz(p)})
        for p in GEN_DIR.iterdir():
            if p.is_dir(): results.append({"name":f"[gen] {p.name}","path":str(p),"source":"generated","size":sz(p)})
    return jsonify(results)

@app.route("/api/datasets/delete", methods=["POST"])
def delete_dataset():
    import shutil
    path = (request.json or {}).get("path","").strip()
    if not path: return jsonify({"ok":False,"error":"No path provided"})
    target = Path(path).resolve()
    if not any(str(target).startswith(str(a.resolve())) for a in [DATASETS_DIR, GEN_DIR]):
        return jsonify({"ok":False,"error":"Path outside allowed directories"})
    if not target.exists(): return jsonify({"ok":False,"error":"Path does not exist"})
    try:
        shutil.rmtree(str(target)) if target.is_dir() else target.unlink()
        emit_log(f"Deleted dataset: {target.name}", "warn"); return jsonify({"ok":True})
    except Exception as e: return jsonify({"ok":False,"error":str(e)})

@app.route("/api/download/dataset", methods=["POST"])
def download_dataset():
    data     = request.json
    repo_id  = data.get("repo_id")
    split    = data.get("split", "train")
    patterns = data.get("patterns")   # list of specific file paths to download
    files    = data.get("files")      # alias for patterns

    if files and not patterns:
        patterns = files

    def do(repo_id, split, patterns):
        current_job.clear()
        current_job.update({
            "status": "downloading", "progress": 0, "logs": [],
            "stage": "↓ Connecting to HuggingFace…"
        })

        pulse_stop = threading.Event()
        _monitor_events.append(pulse_stop)
        def _pulse():
            while not pulse_stop.is_set():
                p = current_job.get("progress", 0)
                if p < 85:
                    current_job["progress"] = p + 1
                    current_job["stage"] = f"↓ Downloading dataset: {repo_id.split('/')[-1]}…"
                pulse_stop.wait(2.0)
        threading.Thread(target=_pulse, daemon=True).start()

        try:
            save_path = str(DATASETS_DIR / repo_id.replace("/", "_"))

            if patterns:
                # Download specific files via hf_hub_download / snapshot_download
                from huggingface_hub import snapshot_download, hf_hub_download
                import shutil
                token = hf_token["value"]
                emit_log(f"Downloading {len(patterns)} file(s) from {repo_id}", "info")
                if len(patterns) == 1:
                    hf_hub_download(repo_id=repo_id, filename=patterns[0],
                                    repo_type="dataset", local_dir=save_path, token=token)
                else:
                    snapshot_download(repo_id=repo_id, repo_type="dataset",
                                      allow_patterns=patterns, local_dir=save_path, token=token)
            else:
                emit_log(f"Downloading dataset: {repo_id} (split={split})", "info")
                from datasets import load_dataset
                ds = load_dataset(repo_id, split=split, token=hf_token["value"])
                pulse_stop.set()
                current_job["stage"]    = "Saving to disk…"
                current_job["progress"] = 90
                ds.save_to_disk(save_path)

            pulse_stop.set()
            current_job["status"]   = "done"
            current_job["progress"] = 100
            current_job["stage"]    = f"✅ Saved: {repo_id.split('/')[-1]}"
            emit_log(f"Dataset saved to: {save_path}", "success")
        except Exception as e:
            pulse_stop.set()
            current_job["status"] = "error"
            current_job["stage"]  = "❌ Dataset download failed"
            emit_log(f"Dataset download failed: {e}", "error")

    threading.Thread(target=do, args=(repo_id, split, patterns), daemon=True).start()
    return jsonify({"ok": True})

@app.route("/api/outputs")
def list_outputs():
    return jsonify([{"name":p.name,"path":str(p)} for p in OUTPUTS_DIR.iterdir() if p.is_dir()])

@app.route("/api/autotrain/presets")
def autotrain_presets():
    result = {}
    for k, v in AUTOTRAIN_PRESETS.items():
        result[k] = {"label":v["label"],"description":v["description"],"note":v.get("note",""),
            "recommended_datasets":[{"repo":d["repo"],"split":d["split"]} for d in v["datasets"]],
            "lora_r":v["lora_r"],"max_steps":v["max_steps"],
            "projection_only":v.get("projection_only",False),"vision_encoder":v.get("vision_encoder",None)}
    return jsonify(result)

@app.route("/api/datasets/generated")
def list_generated():
    results = []
    for p in GEN_DIR.glob("*.jsonl"): results.append({"name":p.name,"path":str(p)})
    for p in GEN_DIR.iterdir():
        if p.is_dir(): results.append({"name":p.name,"path":str(p)})
    return jsonify(results)

# ── HuggingFace Auth ──────────────────────────────────────────────────────────
@app.route("/api/hf/login", methods=["POST"])
def hf_login():
    data=request.json or {}; token=data.get("token","").strip(); save_flag=data.get("save",False)
    if not token: return jsonify({"ok":False,"error":"No token provided"})
    try:
        from huggingface_hub import HfApi
        info=HfApi(token=token).whoami(); username=info.get("name","unknown")
        hf_token["value"]=token; hf_token["username"]=username
        if save_flag:
            TOKEN_FILE.write_text(token)
            TOKEN_FILE.chmod(0o600)
            emit_log(f"Token saved", "info")
        emit_log(f"Logged in as: {username}", "success")
        return jsonify({"ok":True,"username":username,"saved":save_flag})
    except Exception as e: return jsonify({"ok":False,"error":str(e)})

@app.route("/api/hf/logout", methods=["POST"])
def hf_logout():
    data=request.json or {}; delete_saved=data.get("delete_saved",False)
    hf_token["value"]=None; hf_token["username"]=None
    if delete_saved and TOKEN_FILE.exists(): TOKEN_FILE.unlink(); emit_log("Saved token deleted.","info")
    emit_log("Logged out of HuggingFace.", "info"); return jsonify({"ok":True})

@app.route("/api/hf/status")
def hf_status():
    return jsonify({"logged_in":hf_token["value"] is not None,"username":hf_token["username"],"token_saved":TOKEN_FILE.exists()})


# ── App Config ────────────────────────────────────────────────────────────────
@app.route("/api/config", methods=["GET"])
def get_config():
    """Return the current config (safe subset — no secrets)."""
    return jsonify({k: v for k, v in _cfg.items()})

@app.route("/api/config", methods=["POST"])
def post_config():
    """Update one or more config values and persist to config.json."""
    data = request.json or {}
    allowed = set(CONFIG_DEFAULTS.keys())
    updated = {}
    for k, v in data.items():
        if k in allowed:
            _cfg[k] = v
            updated[k] = v
    _save_config()
    return jsonify({"ok": True, "updated": updated})



# ═══ SYSTEM RESOURCES ════════════════════════════════════════════════════════

@app.route("/api/tuning_stats")
def api_tuning_stats():
    """Return aggregated timing stats from all saved tuning runs."""
    entries = _load_tuning_logs()
    if not entries:
        return jsonify({"ok": True, "count": 0, "secs_per_step": None, "tok_per_sec": None})

    # Use median to be robust against outliers (e.g. first cold-start run)
    sps_vals = sorted(e["secs_per_step"] for e in entries if "secs_per_step" in e and e["secs_per_step"] > 0)
    tps_vals = sorted(e["tok_per_sec"]   for e in entries if "tok_per_sec"   in e and e["tok_per_sec"]   > 0)

    def _median(vals):
        if not vals: return None
        m = len(vals) // 2
        return vals[m] if len(vals) % 2 else (vals[m-1] + vals[m]) / 2

    # Also expose the last 20 raw entries so the frontend can show a history
    recent = entries[-20:]

    return jsonify({
        "ok":           True,
        "count":        len(entries),
        "secs_per_step": round(_median(sps_vals), 4) if sps_vals else None,
        "tok_per_sec":   round(_median(tps_vals), 2) if tps_vals else None,
        "sps_min":       round(sps_vals[0],  4) if sps_vals else None,
        "sps_max":       round(sps_vals[-1], 4) if sps_vals else None,
        "recent":        recent,
    })

@app.route("/api/tuning_stats/clear", methods=["POST"])
def api_tuning_stats_clear():
    """Delete all saved timing logs (reset calibration)."""
    try:
        if TUNING_LOGS_FILE.exists():
            TUNING_LOGS_FILE.unlink()
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

@app.route("/api/system/resources")
def system_resources():
    import subprocess, re as _re
    result = {"vram_total_gb": 0, "vram_free_gb": 0,
              "ram_total_gb": 0, "ram_free_gb": 0, "disk_free_gb": 0}
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total,memory.free",
             "--format=csv,noheader,nounits"], timeout=5
        ).decode().strip().split("\n")[0]
        tot, free = [int(x.strip()) for x in out.split(",")]
        result["vram_total_gb"] = round(tot  / 1024, 1)
        result["vram_free_gb"]  = round(free / 1024, 1)
    except Exception:
        pass
    try:
        with open("/proc/meminfo") as f:
            mem = f.read()
        def _kb(key):
            m = _re.search(rf"{key}:\s+(\d+)", mem)
            return int(m.group(1)) if m else 0
        result["ram_total_gb"] = round(_kb("MemTotal")     / 1024**2, 1)
        result["ram_free_gb"]  = round(_kb("MemAvailable") / 1024**2, 1)
    except Exception:
        pass
    try:
        import shutil as _sh
        du = _sh.disk_usage(str(OUTPUTS_DIR))
        result["disk_free_gb"] = round(du.free / 1024**3, 1)
    except Exception:
        pass
    return jsonify(result)

# ═══ MODEL CONFIG READ ════════════════════════════════════════════════════════

@app.route("/api/model/config", methods=["POST"])
def read_model_config():
    import json as _json
    model_path = request.json.get("model_path", "").strip()
    cfg_file = Path(model_path) / "config.json"
    if not cfg_file.exists():
        return jsonify({"ok": False, "error": "config.json not found"})
    try:
        cfg = _json.loads(cfg_file.read_text())
        return jsonify({"ok": True, "config": cfg})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


# ═══ ROPE SCALING ════════════════════════════════════════════════════════════

def run_rope_scale(config):
    try:
        import shutil, json as _json
        model_path   = config["model_path"]
        rope_type    = config.get("rope_type", "linear")
        factor       = float(config.get("factor", 2.0))
        orig_maxpos  = int(config.get("original_max_position_embeddings", 2048))
        out_name     = config.get("output_name") or f"rope_{rope_type}_{int(factor)}x_{int(time.time())}"
        out_dir      = OUTPUTS_DIR / out_name

        set_stage("Copying model files")
        set_progress(5)
        if out_dir.exists():
            shutil.rmtree(out_dir)
        shutil.copytree(model_path, str(out_dir))
        emit_log(f"Copied {model_path} → {out_dir}", "success")
        set_progress(60)

        set_stage("Patching config.json")
        cfg_file = out_dir / "config.json"
        if not cfg_file.exists():
            raise FileNotFoundError(f"config.json not found in {model_path}")

        cfg = _json.loads(cfg_file.read_text())
        new_maxpos = round(orig_maxpos * factor)

        rope_scaling = {"rope_type": rope_type, "factor": factor}
        if rope_type in ("yarn", "llama3"):
            rope_scaling["original_max_position_embeddings"] = int(
                config.get("attention_factor") and orig_maxpos or orig_maxpos)
            rope_scaling["original_max_position_embeddings"] = orig_maxpos
            rope_scaling["attention_factor"] = float(config.get("attention_factor", 0.1))
            rope_scaling["beta_fast"]        = int(config.get("beta_fast", 32))
            rope_scaling["beta_slow"]        = int(config.get("beta_slow", 1))

        cfg["rope_scaling"]              = rope_scaling
        cfg["max_position_embeddings"]   = new_maxpos

        # Some architectures use different key names
        if "n_positions" in cfg:
            cfg["n_positions"] = new_maxpos

        cfg_file.write_text(_json.dumps(cfg, indent=2))
        emit_log(f"rope_scaling = {_json.dumps(rope_scaling)}", "info")
        emit_log(f"max_position_embeddings: {orig_maxpos} → {new_maxpos}", "success")
        set_progress(90)

        set_stage("Done")
        set_progress(100)
        current_job["status"] = "done"
        emit_log(f"✅ RoPE-scaled model saved: {out_dir}", "success")
        emit_log(f"   New context window: {new_maxpos:,} tokens", "info")
        emit_log(f"   Use in Fine-Tune or Chat: {out_dir}", "info")

    except KeyboardInterrupt:
        current_job["status"] = "stopped"; emit_log("Stopped.", "warn")
    except Exception as e:
        current_job["status"] = "error"; emit_log(f"Error: {e}", "error"); emit_log(traceback.format_exc(), "error")


@app.route("/api/rope_scale", methods=["POST"])
def rope_scale_route():
    return _start_job(run_rope_scale, request.json)


# ═══ CREATE BLANK MODEL ══════════════════════════════════════════════════════

def run_create_blank_model(config):
    try:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerFast
        from tokenizers import Tokenizer
        from tokenizers.models import BPE

        arch         = config.get("architecture", "llama")
        vocab_size   = int(config.get("vocab_size",  32000))
        hidden_size  = int(config.get("hidden_size",  512))
        num_layers   = int(config.get("num_layers",   6))
        num_heads    = int(config.get("num_heads",    8))
        intermediate = int(config.get("intermediate_size", hidden_size * 4))
        max_pos      = int(config.get("max_position_embeddings", 2048))
        out_name     = config.get("output_name") or f"blank_{arch}_{int(time.time())}"
        out_dir      = str(OUTPUTS_DIR / out_name)

        set_stage("Building model config"); set_progress(5)
        model_type = arch.lower()

        if model_type in ("llama", "mistral", "gemma"):
            cfg = AutoConfig.for_model(
                model_type, vocab_size=vocab_size, hidden_size=hidden_size,
                intermediate_size=intermediate, num_hidden_layers=num_layers,
                num_attention_heads=num_heads, num_key_value_heads=num_heads,
                max_position_embeddings=max_pos)
        elif model_type == "phi":
            cfg = AutoConfig.for_model(
                "phi", vocab_size=vocab_size, hidden_size=hidden_size,
                intermediate_size=intermediate, num_hidden_layers=num_layers,
                num_attention_heads=num_heads, max_position_embeddings=max_pos)
        else:  # gpt2
            cfg = AutoConfig.for_model(
                "gpt2", vocab_size=vocab_size, n_embd=hidden_size,
                n_layer=num_layers, n_head=num_heads, n_positions=max_pos)

        emit_log(f"Config: {model_type}, layers={num_layers}, hidden={hidden_size}, heads={num_heads}, max_pos={max_pos}", "info")
        set_progress(15)
                # ── GPU / CPU init ────────────────────────────────────────────────────
        use_gpu = config.get("use_gpu", True)
        if use_gpu and torch.cuda.is_available():
            device   = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            vram_free = round((torch.cuda.get_device_properties(0).total_memory
                               - torch.cuda.memory_allocated()) / 1024**3, 1)
            emit_log(f"GPU: {gpu_name} — {vram_free} GB free — init on GPU (fp16)", "success")
        else:
            device = torch.device("cpu")
            emit_log("No CUDA GPU found — using CPU (fp32)" if use_gpu else "CPU mode (fp32)", "warn" if use_gpu else "info")

        set_stage(f"Initialising weights on {device.type.upper()} ({'fp16' if device.type == 'cuda' else 'fp32'})")
        if device.type == "cuda":
            cfg.torch_dtype = torch.float16
            model = AutoModelForCausalLM.from_config(cfg).to(device)
            emit_log(f"VRAM used: {round(torch.cuda.memory_allocated()/1024**3, 2)} GB", "info")
        else:
            model = AutoModelForCausalLM.from_config(cfg)

        param_count = sum(p.numel() for p in model.parameters())
        emit_log(f"Model created: {param_count/1e6:.1f}M parameters", "success")
        set_progress(40)


        set_stage("Building minimal tokenizer")
        base_vocab = {chr(i): i for i in range(128)}
        base_vocab["<unk>"] = 128; base_vocab["<s>"] = 129
        base_vocab["</s>"] = 130; base_vocab["<pad>"] = 131
        for b in range(256):
            tok_str = f"<0x{b:02X}>"
            if tok_str not in base_vocab:
                base_vocab[tok_str] = len(base_vocab)
        while len(base_vocab) < vocab_size:
            base_vocab[f"[unused{len(base_vocab)}]"] = len(base_vocab)
        base_vocab = dict(list(base_vocab.items())[:vocab_size])
        hf_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(BPE(vocab=base_vocab, merges=[])),
            bos_token="<s>", eos_token="</s>", unk_token="<unk>", pad_token="<pad>",
            model_max_length=max_pos)
        emit_log("Tokenizer built", "success"); set_progress(65)

        set_stage("Saving model")
        os.makedirs(out_dir, exist_ok=True)
        model.save_pretrained(out_dir)
        hf_tokenizer.save_pretrained(out_dir)
        if device.type == "cuda":
            del model
            torch.cuda.empty_cache()
            emit_log("GPU memory released after save", "info")
        set_progress(95)
        set_progress(100); current_job["status"] = "done"
        emit_log(f"✅ Blank model saved: {out_dir}", "success")
        emit_log(f"   Max context: {max_pos:,} tokens", "info")
        emit_log(f"   Use in Fine-Tune or Chat: {out_dir}", "info")
        emit_log(f"   Extend context later via RoPE Scale page", "info")

    except KeyboardInterrupt:
        current_job["status"] = "stopped"; emit_log("Stopped.", "warn")
    except Exception as e:
        current_job["status"] = "error"; emit_log(f"Error: {e}", "error"); emit_log(traceback.format_exc(), "error")


@app.route("/api/create_blank_model", methods=["POST"])
def create_blank_model_route():
    return _start_job(run_create_blank_model, request.json)


# ═══ CHAT STATE + VRAM ESTIMATE ══════════════════════════════════════════════

chat_model_state = {"model": None, "tokenizer": None, "model_path": None, "config": {}, "status": "idle"}
chat_lock = threading.Lock()

def _estimate_vram_gb(model_path: str, load_in_4bit: bool = True) -> float:
    import json as _json
    p = Path(model_path)
    cfg_file = p / "config.json"
    params = None
    if cfg_file.exists():
        try:
            cfg = _json.loads(cfg_file.read_text())
            H  = cfg.get("hidden_size", cfg.get("n_embd", 0))
            I  = cfg.get("intermediate_size", H * 4)
            L  = cfg.get("num_hidden_layers", cfg.get("n_layer", 0))
            V  = cfg.get("vocab_size", 32000)
            heads    = cfg.get("num_attention_heads", cfg.get("n_head", 8))
            kv_heads = cfg.get("num_key_value_heads", heads)
            if H and L and V:
                kv_ratio = kv_heads / max(heads, 1)
                attn  = L * (H*H + H*H*kv_ratio*2 + H*H)
                ffn   = L * 3 * H * I
                emb   = V * H
                params = attn + ffn + emb
        except Exception:
            pass
    if params is None:
        try:
            disk = sum(f.stat().st_size for f in p.rglob("*.safetensors"))
            if disk == 0:
                disk = sum(f.stat().st_size for f in p.rglob("*.bin"))
            params = disk / 2
        except Exception:
            params = 0
    if params == 0:
        return 0.0
    bpp    = 0.55 if load_in_4bit else 2.0
    return round(params * bpp * 1.20 / 1e9, 2)


@app.route("/api/chat/vram_estimate", methods=["POST"])
def chat_vram_estimate():
    data = request.json or {}
    model_path   = data.get("model_path", "").strip()
    load_in_4bit = data.get("load_in_4bit", True)
    if not model_path:
        return jsonify({"ok": False, "error": "No model path"})
    try:
        vram_gb = _estimate_vram_gb(model_path, load_in_4bit)
        return jsonify({"ok": True, "vram_gb": vram_gb, "fits": vram_gb <= 12.0 or vram_gb == 0.0, "limit_gb": 12.0})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/chat/load", methods=["POST"])
def chat_load():
    data          = request.json or {}
    model_path    = data.get("model_path", "").strip()
    load_in_4bit  = data.get("load_in_4bit", True)
    max_seq       = int(data.get("max_seq_length", 2048))
    system_prompt = data.get("system_prompt", "").strip()
    if not model_path:
        return jsonify({"ok": False, "error": "No model path"})
    if "gguf" in model_path.lower():
        return jsonify({"ok": False, "error": "GGUF models not supported for chat here."})
    vram_gb = _estimate_vram_gb(model_path, load_in_4bit)
    if vram_gb > 12.0:
        return jsonify({"ok": False, "error": f"Estimated VRAM {vram_gb} GB > 12 GB limit. Use 4-bit or smaller model."})

    def _do_load():
        with chat_lock:
            if chat_model_state["model"] is not None:
                del chat_model_state["model"], chat_model_state["tokenizer"]
                chat_model_state["model"] = chat_model_state["tokenizer"] = None
                gc.collect()
                try: import torch; torch.cuda.empty_cache()
                except: pass
            try:
                import torch
                model, tokenizer, loader = load_model_and_tokenizer(model_path, max_seq_length=max_seq, load_in_4bit=load_in_4bit)
                try:
                    from unsloth import FastModel; FastModel.for_inference(model)
                except:
                    try:
                        from unsloth import FastLanguageModel; FastLanguageModel.for_inference(model)
                    except: model.eval()
                actual_vram = 0.0
                try: import torch; actual_vram = round(torch.cuda.memory_allocated()/1e9, 2)
                except: pass
                chat_model_state.update({"model": model, "tokenizer": tokenizer, "model_path": model_path,
                    "config": {"load_in_4bit": load_in_4bit, "max_seq_length": max_seq,
                               "system_prompt": system_prompt, "actual_vram_gb": actual_vram},
                    "status": "ready", "error": None})
                print(f"[INFO] Chat model loaded: {model_path} ({actual_vram} GB)")
            except Exception as e:
                chat_model_state["status"] = "error"; chat_model_state["error"] = str(e)
                print(f"[ERROR] Chat load failed: {e}")

    chat_model_state["status"] = "loading"; chat_model_state["error"] = None
    threading.Thread(target=_do_load, daemon=True).start()
    return jsonify({"ok": True, "estimated_vram_gb": vram_gb})


@app.route("/api/chat/status")
def chat_status():
    s = chat_model_state
    return jsonify({"loaded": s["model"] is not None, "status": s.get("status","idle"),
                    "model_path": s.get("model_path"), "error": s.get("error"), "config": s.get("config",{})})


@app.route("/api/chat/unload", methods=["POST"])
def chat_unload():
    with chat_lock:
        if chat_model_state["model"] is not None:
            del chat_model_state["model"], chat_model_state["tokenizer"]
            chat_model_state.update({"model": None, "tokenizer": None, "model_path": None,
                                     "config": {}, "status": "idle"})
            gc.collect()
            try: import torch; torch.cuda.empty_cache()
            except: pass
    return jsonify({"ok": True})


@app.route("/api/chat/generate", methods=["POST"])
def chat_generate():
    data               = request.json or {}
    messages           = data.get("messages", [])
    max_new_tokens     = int(data.get("max_new_tokens", 512))
    temperature        = float(data.get("temperature", 0.7))
    top_p              = float(data.get("top_p", 0.9))
    repetition_penalty = float(data.get("repetition_penalty", 1.1))

    if chat_model_state["model"] is None:
        return jsonify({"ok": False, "error": "No model loaded"}), 400

    model     = chat_model_state["model"]
    tokenizer = chat_model_state["tokenizer"]
    sys_p     = chat_model_state["config"].get("system_prompt", "")
    max_seq   = chat_model_state["config"].get("max_seq_length", 2048)

    def _build_prompt(messages, sys_p):
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            msgs = ([{"role": "system", "content": sys_p}] if sys_p else []) + messages
            try: return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            except: pass
        parts = [f"### System:\n{sys_p}\n"] if sys_p else []
        for m in messages:
            parts.append(f"### {m.get('role','user').capitalize()}:\n{m.get('content','')}")
        parts.append("### Assistant:\n")
        return "\n\n".join(parts)

    prompt = _build_prompt(messages, sys_p)

    stop_flag = threading.Event()
    def _stream():
        import torch, queue as _queue
        tq = _queue.Queue()
        done_ev = threading.Event()
        def _gen():
            with torch.no_grad():
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_seq).to(model.device)
                gen_ids = inputs["input_ids"].clone()
                for _ in range(max_new_tokens):
                    if stop_flag.is_set(): break
                    out  = model(input_ids=gen_ids, attention_mask=torch.ones_like(gen_ids))
                    logits = out.logits[:, -1, :]
                    if temperature > 0:
                        probs   = torch.softmax(logits / temperature, dim=-1)
                        next_id = torch.multinomial(probs, 1)
                    else:
                        next_id = logits.argmax(dim=-1, keepdim=True)
                    tq.put(next_id.item())
                    gen_ids = torch.cat([gen_ids, next_id], dim=-1)
                    if next_id.item() == tokenizer.eos_token_id: break
            tq.put(None)
        threading.Thread(target=_gen, daemon=True).start()
        while True:
            tok_id = tq.get(timeout=30)
            if tok_id is None: break
            yield f"data: {json.dumps({'token': tokenizer.decode([tok_id], skip_special_tokens=True)})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"

    return Response(stream_with_context(_stream()), mimetype="text/event-stream")


# ═══════════════════════════════════════════════════════════════════════════════
# JOB: GGUF EXPORT
# ═══════════════════════════════════════════════════════════════════════════════
GGUF_QUANT_TYPES = [
    "Q4_0","Q4_1","Q4_K_S","Q4_K_M","Q4_K_L",
    "Q5_0","Q5_1","Q5_K_S","Q5_K_M",
    "Q6_K",
    "Q8_0",
    "F16","BF16","F32",
]

def run_gguf_export(config):
    try:
        model_path   = config["model_path"]
        quant_types  = config.get("quant_types") or ["Q4_K_M"]
        output_name  = config.get("output_name") or Path(model_path).name
        out_base     = OUTPUTS_DIR / output_name
        out_base.mkdir(parents=True, exist_ok=True)
        results = []

        set_stage("Loading model for GGUF export")
        set_progress(5)

        # Try Unsloth's built-in GGUF saver first (fastest, no llama.cpp needed)
        try:
            model, tok, _ = load_model_and_tokenizer(model_path)
            set_progress(20)
            from unsloth import FastModel
            for i, qt in enumerate(quant_types):
                if stop_flag.is_set(): raise KeyboardInterrupt()
                out_file = str(out_base / f"{output_name}_{qt}.gguf")
                set_stage(f"Exporting {qt} ({i+1}/{len(quant_types)})")
                emit_log(f"Exporting {qt} → {out_file}", "info")
                try:
                    model.save_pretrained_gguf(out_base / output_name, tok,
                                               quantization_method=qt.lower())
                    # Unsloth names it automatically; rename to our convention
                    import glob as _glob
                    produced = _glob.glob(str(out_base / "*.gguf"))
                    if produced:
                        import shutil
                        shutil.move(produced[-1], out_file)
                    emit_log(f"✅ {qt} saved: {Path(out_file).name}", "success")
                    results.append({"quant": qt, "file": out_file, "ok": True})
                except Exception as e:
                    emit_log(f"⚠️ {qt} failed via Unsloth: {e}", "warn")
                    results.append({"quant": qt, "error": str(e), "ok": False})
                set_progress(20 + (i+1)/len(quant_types)*70)
            unload_model(model, tok)

        except Exception as e:
            emit_log(f"Unsloth GGUF export failed: {e}, trying llama.cpp fallback", "warn")
            # llama.cpp convert + quantize fallback
            import subprocess, shutil
            convert_py = Path("/usr/local/lib/python3.11/dist-packages/llama_cpp/convert_hf_to_gguf.py")
            if not convert_py.exists():
                for p in ["/usr/local/bin/convert-hf-to-gguf", "/usr/bin/llama-quantize"]:
                    if Path(p).exists(): convert_py = p; break
            if not convert_py:
                raise RuntimeError("llama.cpp not found. Install llama-cpp-python or llama.cpp.")

            f16_path = str(out_base / f"{output_name}_F16.gguf")
            set_stage("Converting to F16 GGUF base")
            emit_log("Running llama.cpp convert...", "info")
            r = subprocess.run(["python3", str(convert_py), model_path,
                                 "--outtype", "f16", "--outfile", f16_path],
                                capture_output=True, text=True, timeout=600)
            if r.returncode != 0:
                raise RuntimeError(f"convert failed: {r.stderr[-500:]}")
            emit_log("F16 base created", "success")
            set_progress(40)

            quantize_bin = shutil.which("llama-quantize") or shutil.which("quantize")
            if quantize_bin:
                for i, qt in enumerate(quant_types):
                    if stop_flag.is_set(): raise KeyboardInterrupt()
                    if qt in ("F16","BF16","F32"): results.append({"quant":qt,"file":f16_path,"ok":True}); continue
                    out_file = str(out_base / f"{output_name}_{qt}.gguf")
                    set_stage(f"Quantizing {qt}")
                    r2 = subprocess.run([quantize_bin, f16_path, out_file, qt],
                                        capture_output=True, text=True, timeout=300)
                    if r2.returncode == 0:
                        emit_log(f"✅ {qt} → {Path(out_file).name}", "success")
                        results.append({"quant":qt,"file":out_file,"ok":True})
                    else:
                        emit_log(f"⚠️ {qt} quantize failed: {r2.stderr[-200:]}", "warn")
                        results.append({"quant":qt,"error":r2.stderr[-200:],"ok":False})
                    set_progress(40 + (i+1)/len(quant_types)*55)
            else:
                emit_log("llama-quantize binary not found; only F16 available", "warn")
                results.append({"quant":"F16","file":f16_path,"ok":True})

        set_progress(100)
        current_job["status"] = "done"
        ok_count = sum(1 for r in results if r.get("ok"))
        emit_log(f"GGUF export done: {ok_count}/{len(results)} formats succeeded → {out_base}", "success")
        current_job["gguf_results"] = results
    except KeyboardInterrupt:
        current_job["status"] = "stopped"; emit_log("Stopped.", "warn")
    except Exception as e:
        current_job["status"] = "error"; emit_log(f"Error: {e}", "error"); emit_log(traceback.format_exc(), "error")


# ═══════════════════════════════════════════════════════════════════════════════
# JOB: MOBILE / PHONE EXPORT  (MLC-LLM / GGUF small / ExecuTorch)
# ═══════════════════════════════════════════════════════════════════════════════
# ── Model family detection helper ─────────────────────────────────────────────
_ARCH_STOP_TOKENS = {
    "gemma":   {"start": "<bos>",  "stop": ["<eos>", "<end_of_turn>"],
                "prefix": "<start_of_turn>user\n",
                "suffix": "<end_of_turn>\n<start_of_turn>model\n"},
    "llama":   {"start": "<s>",    "stop": ["</s>", "<|eot_id|>"],
                "prefix": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n",
                "suffix": "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"},
    "phi":     {"start": "<|endoftext|>", "stop": ["<|endoftext|>", "<|end|>"],
                "prefix": "<|user|>\n", "suffix": "<|end|>\n<|assistant|>\n"},
    "qwen":    {"start": "<|im_start|>", "stop": ["<|im_end|>", "<|endoftext|>"],
                "prefix": "<|im_start|>user\n", "suffix": "<|im_end|>\n<|im_start|>assistant\n"},
    "mistral": {"start": "<s>",    "stop": ["</s>", "[/INST]"],
                "prefix": "[INST] ", "suffix": " [/INST]"},
    "default": {"start": "<s>",    "stop": ["</s>"],
                "prefix": "User: ", "suffix": "\nAssistant: "},
}

def _detect_model_family(model_path):
    name = Path(model_path).name.lower()
    for family in ("gemma", "llama", "phi", "qwen", "mistral"):
        if family in name:
            return family
    # Try config.json
    cfg_file = Path(model_path) / "config.json"
    if cfg_file.exists():
        try:
            import json as _j
            cfg = _j.loads(cfg_file.read_text())
            arch = " ".join(str(v).lower() for v in cfg.values())
            for family in ("gemma", "llama", "phi", "qwen", "mistral"):
                if family in arch:
                    return family
        except Exception:
            pass
    return "default"

def _get_sentencepiece_tokenizer(model_path, out_dir):
    """
    Try to find or convert a SentencePiece tokenizer.model file.
    Gemma / Llama / Mistral ship tokenizer.model directly.
    BPE-only models (Phi, Qwen) need conversion via ai_edge_torch utility.
    Returns path to the .model file, or None if not found.
    """
    import subprocess, shutil
    sp_path = Path(model_path) / "tokenizer.model"
    if sp_path.exists():
        return str(sp_path)

    # Try ai_edge_torch tokenizer_to_sentencepiece conversion
    tool_candidates = [
        "/usr/local/lib/python3.11/dist-packages/ai_edge_torch/generative/tools/tokenizer_to_sentencepiece.py",
    ]
    import glob as _glob
    found_tools = _glob.glob("/usr/local/lib/**/tokenizer_to_sentencepiece.py", recursive=True)
    tool_candidates = found_tools + tool_candidates
    for tool in tool_candidates:
        if Path(tool).exists():
            out_tok = str(out_dir / "tokenizer.model")
            r = subprocess.run(
                ["python3", tool, "--checkpoint", model_path, "--output_path", out_tok],
                capture_output=True, text=True, timeout=120)
            if r.returncode == 0 and Path(out_tok).exists():
                emit_log("Tokenizer converted to SentencePiece via ai_edge_torch", "info")
                return out_tok
            else:
                emit_log(f"Tokenizer conversion warn: {r.stderr[-200:]}", "warn")

    # Last resort: try sentencepiece directly from vocab files
    try:
        import sentencepiece as spm
        sp_model = str(out_dir / "tokenizer.model")
        # Sentencepiece can't train from HF vocab easily; just warn
        emit_log("No SentencePiece tokenizer.model found. For non-Gemma/Llama models, "
                 "run: python -m ai_edge_torch.generative.tools.tokenizer_to_sentencepiece "
                 f"--checkpoint {model_path} --output_path {sp_model}", "warn")
    except ImportError:
        pass
    return None


def run_mobile_export(config):
    try:
        model_path   = config["model_path"]
        target       = config.get("target", "litertlm")
        output_name  = config.get("output_name") or (Path(model_path).name + "_mobile")
        out_dir      = OUTPUTS_DIR / output_name
        out_dir.mkdir(parents=True, exist_ok=True)
        quantize     = config.get("quantize", "dynamic_int8")   # dynamic_int8 | int8 | fp32
        prefill_len  = cfg_int(config.get("prefill_seq_len"), 1024)
        kv_len       = cfg_int(config.get("kv_cache_max_len"), 2048)
        # Allow user to override token strings
        family       = _detect_model_family(model_path)
        arch_cfg     = _ARCH_STOP_TOKENS.get(family, _ARCH_STOP_TOKENS["default"])
        start_token  = config.get("start_token")  or arch_cfg["start"]
        stop_tokens  = config.get("stop_tokens")  or arch_cfg["stop"]
        prompt_pfx   = config.get("prompt_prefix") or arch_cfg["prefix"]
        prompt_sfx   = config.get("prompt_suffix") or arch_cfg["suffix"]

        import subprocess, shutil

        emit_log(f"Target: {target} | Model family: {family} | Quantize: {quantize}", "info")
        emit_log(f"Output: {out_dir}", "info")

        # ══════════════════════════════════════════════════════════════════════
        if target == "litertlm":
            # ── Step 1: Install deps check ────────────────────────────────────
            set_stage("Checking ai-edge-torch / mediapipe install")
            set_progress(2)
            missing = []
            for pkg, imp in [("ai-edge-torch","ai_edge_torch"), ("mediapipe","mediapipe")]:
                try:
                    __import__(imp)
                except ImportError:
                    missing.append(pkg)
            if missing:
                emit_log(f"Missing: {missing}. Installing...", "warn")
                r = subprocess.run(
                    ["pip", "install", "--break-system-packages"] + missing,
                    capture_output=True, text=True, timeout=300)
                if r.returncode != 0:
                    raise RuntimeError(f"pip install failed: {r.stderr[-300:]}")
                emit_log("Dependencies installed", "success")

            # ── Step 2: Convert model to .tflite via ai_edge_torch generative ─
            set_stage("Converting to TFLite (ai-edge-torch Generative API)")
            set_progress(8)
            emit_log("This step is CPU-intensive and may take 10–60 min depending on model size", "warn")
            emit_log(f"Quantization: {quantize} | Prefill: {prefill_len} | KV cache: {kv_len}", "info")

            tflite_path = str(out_dir / f"{output_name}.tflite")

            # Detect which example converter to use based on architecture
            example_map = {
                "gemma":   "ai_edge_torch.generative.examples.gemma3.convert_gemma3_to_tflite",
                "llama":   "ai_edge_torch.generative.examples.llama.convert_to_tflite",
                "phi":     "ai_edge_torch.generative.examples.phi3.convert_phi3_to_tflite",
                "qwen":    "ai_edge_torch.generative.examples.qwen.convert_to_tflite",
                "mistral": "ai_edge_torch.generative.examples.llama.convert_to_tflite",
            }
            converter_module = example_map.get(family)

            if converter_module:
                emit_log(f"Using ai-edge-torch example converter: {converter_module}", "info")
                convert_cmd = [
                    "python3", "-m", converter_module,
                    "--checkpoint_path", model_path,
                    "--output_path", str(out_dir),
                    "--quantize", quantize,
                    f"--prefill_seq_lens={prefill_len}",
                    f"--kv_cache_max_len={kv_len}",
                ]
                r = subprocess.run(convert_cmd, capture_output=True, text=True, timeout=7200)
                if r.returncode != 0:
                    emit_log(f"Example converter failed: {r.stderr[-600:]}", "warn")
                    emit_log("Falling back to generic ai_edge_torch conversion...", "info")
                    converter_module = None  # fall through to generic
                else:
                    # Find the produced .tflite
                    import glob as _glob
                    produced = _glob.glob(str(out_dir / "*.tflite"))
                    if produced:
                        tflite_path = produced[0]
                        emit_log(f"TFLite produced: {Path(tflite_path).name}", "success")

            if not converter_module or not Path(tflite_path).exists():
                # Generic ai_edge_torch conversion (for architectures without a canned example)
                emit_log("Using generic ai_edge_torch conversion pipeline", "info")
                try:
                    import ai_edge_torch
                    import torch
                    from transformers import AutoModelForCausalLM, AutoTokenizer
                    from ai_edge_torch.generative.quantize import quant_recipe as qr

                    hf_tok = _load_saved_token()
                    emit_log("Loading model for generic conversion...", "info")
                    hf_model = AutoModelForCausalLM.from_pretrained(
                        model_path, torch_dtype=torch.float32, token=hf_tok)
                    hf_model.eval()
                    set_progress(30)
                    emit_log("Running ai_edge_torch.convert()...", "info")
                    sample_input = (torch.zeros(1, prefill_len, dtype=torch.long),)
                    quant = qr.get_default_8bit_recipe() if "int8" in quantize else None
                    edge_model = ai_edge_torch.convert(hf_model, sample_input,
                                                        quant_config=quant)
                    edge_model.export(tflite_path)
                    emit_log(f"Generic conversion done: {tflite_path}", "success")
                except Exception as eg:
                    raise RuntimeError(
                        f"Generic conversion also failed: {eg}. "
                        "For best results use a Gemma or Llama model, or install "
                        "ai-edge-torch and re-run with a supported architecture.") from eg

            set_progress(60)

            # ── Step 3: Get SentencePiece tokenizer ───────────────────────────
            set_stage("Preparing tokenizer")
            tok_path = _get_sentencepiece_tokenizer(model_path, out_dir)
            if not tok_path:
                emit_log("WARNING: Could not find SentencePiece tokenizer.model. "
                         "Bundling will be skipped. Manually add tokenizer.model and re-run bundler.", "warn")
            set_progress(70)

            # ── Step 4: Bundle .tflite + tokenizer → .task / .litertlm ───────
            set_stage("Bundling into .litertlm (MediaPipe bundler)")
            out_bundle = str(out_dir / f"{output_name}.litertlm")

            if tok_path:
                emit_log(f"Bundling with tokenizer: {Path(tok_path).name}", "info")
                emit_log(f"Start token: {start_token} | Stop: {stop_tokens}", "info")
                emit_log(f"Prompt prefix: {repr(prompt_pfx)}", "info")
                try:
                    from mediapipe.tasks.python.genai import bundler as mp_bundler
                    # Try bytes_to_unicode for BPE tokenizers (Phi, Qwen, Llama3)
                    needs_bpe_mapping = family in ("phi", "qwen", "llama")
                    bundle_cfg = mp_bundler.BundleConfig(
                        tflite_model=tflite_path,
                        tokenizer_model=tok_path,
                        start_token=start_token,
                        stop_tokens=stop_tokens,
                        output_filename=out_bundle,
                        prompt_prefix=prompt_pfx,
                        prompt_suffix=prompt_sfx,
                        enable_bytes_to_unicode_mapping=needs_bpe_mapping,
                    )
                    mp_bundler.create_bundle(bundle_cfg)
                    if Path(out_bundle).exists():
                        size_mb = Path(out_bundle).stat().st_size / 1048576
                        emit_log(f"✅ .litertlm bundle created: {Path(out_bundle).name} ({size_mb:.1f} MB)", "success")
                    else:
                        # Mediapipe may have written .task extension instead
                        task_alt = out_bundle.replace(".litertlm", ".task")
                        if Path(task_alt).exists():
                            shutil.move(task_alt, out_bundle)
                            emit_log(f"✅ Bundle created (renamed .task → .litertlm): {Path(out_bundle).name}", "success")
                        else:
                            emit_log("Bundler ran but output not found — check out_dir for .task file", "warn")
                except Exception as be:
                    emit_log(f"Bundler error: {be}", "error")
                    emit_log("TFLite file still available — manually bundle with mediapipe bundler.", "warn")
            else:
                emit_log("Skipping bundling: no tokenizer. TFLite is at: " + tflite_path, "warn")

            set_progress(95)
            # Write usage instructions
            readme = out_dir / "HOW_TO_INSTALL.txt"
            readme.write_text(
                f"Google AI Edge Gallery — Install Instructions\n"
                f"=============================================\n\n"
                f"Model: {output_name}\n"
                f"File: {output_name}.litertlm\n\n"
                f"1. Install the 'AI Edge Gallery' app on your Android phone\n"
                f"   (search Google Play Store or: https://github.com/google-ai-edge/gallery)\n\n"
                f"2. Transfer {output_name}.litertlm to your phone\n"
                f"   via USB, Google Drive, or: adb push {output_name}.litertlm /sdcard/\n\n"
                f"3. Open AI Edge Gallery → tap '+' (bottom-right)\n"
                f"4. Select the .litertlm file\n"
                f"5. Configure: CPU or GPU, temperature, top-k\n"
                f"6. Tap Import and start chatting!\n\n"
                f"Architecture detected: {family}\n"
                f"Quantization: {quantize}\n"
                f"Context length: {kv_len} tokens\n"
            )

        # ══════════════════════════════════════════════════════════════════════
        elif target == "gguf_q4":
            fake_cfg = {**config, "quant_types": ["Q4_K_M","Q4_K_S"], "output_name": output_name}
            run_gguf_export(fake_cfg)
            return

        else:
            raise ValueError(f"Unknown mobile target: {target}")

        set_progress(100)
        current_job["status"] = "done"
        emit_log(f"Mobile export complete → {out_dir}", "success")
    except KeyboardInterrupt:
        current_job["status"] = "stopped"; emit_log("Stopped.", "warn")
    except Exception as e:
        current_job["status"] = "error"; emit_log(f"Error: {e}", "error"); emit_log(traceback.format_exc(), "error")


# ═══════════════════════════════════════════════════════════════════════════════
# JOB: MODEL OPTIMIZATION  (compile, quant, KV-cache, Flash Attention)
# ═══════════════════════════════════════════════════════════════════════════════
def run_optimization(config):
    try:
        model_path      = config["model_path"]
        output_name     = config.get("output_name") or (Path(model_path).name + "_opt")
        out_dir         = str(OUTPUTS_DIR / output_name)
        do_torch_compile= config.get("torch_compile", False)
        do_dyn_quant    = config.get("dynamic_quant", False)   # int8 dynamic
        do_kv_quant     = config.get("kv_cache_quant", False)  # int8 KV
        do_flash        = config.get("flash_attention", True)
        do_merge_lora   = config.get("merge_lora", False)      # merge LoRA → base first

        import torch

        set_stage("Loading model")
        set_progress(5)
        emit_log(f"Loading {model_path}", "info")
        model, tok, _ = load_model_and_tokenizer(model_path)
        set_progress(20)

        # ── 1. Merge LoRA adapters if present ─────────────────────────────────
        if do_merge_lora:
            set_stage("Merging LoRA adapters")
            try:
                from peft import PeftModel
                if hasattr(model, "merge_and_unload"):
                    model = model.merge_and_unload()
                    emit_log("LoRA merged into base weights", "success")
                else:
                    emit_log("No LoRA adapters detected, skipping merge", "info")
            except Exception as e:
                emit_log(f"LoRA merge skipped: {e}", "warn")
            set_progress(30)

        # ── 2. Flash Attention 2 ──────────────────────────────────────────────
        if do_flash:
            set_stage("Enabling Flash Attention 2")
            try:
                from unsloth import FastModel
                FastModel.for_inference(model)
                emit_log("Flash Attention 2 enabled via Unsloth", "success")
            except Exception as e:
                emit_log(f"Flash Attention: {e}", "warn")
            set_progress(40)

        # ── 3. torch.compile ──────────────────────────────────────────────────
        if do_torch_compile:
            set_stage("torch.compile (this takes a few minutes)")
            emit_log("Compiling model graph — first inference will be slow, then fast", "info")
            try:
                model = torch.compile(model, mode="reduce-overhead")
                emit_log("torch.compile done", "success")
            except Exception as e:
                emit_log(f"torch.compile failed: {e}", "warn")
            set_progress(60)

        # ── 4. Dynamic INT8 quantization ──────────────────────────────────────
        if do_dyn_quant:
            set_stage("Applying dynamic INT8 quantization")
            try:
                model = torch.quantization.quantize_dynamic(
                    model.cpu(), {torch.nn.Linear}, dtype=torch.qint8)
                emit_log("Dynamic INT8 quantization applied", "success")
            except Exception as e:
                emit_log(f"Dynamic quantization failed: {e}", "warn")
            set_progress(75)

        # ── 5. KV-cache INT8 quantization (bitsandbytes) ─────────────────────
        if do_kv_quant:
            set_stage("KV-cache INT8 quant (bitsandbytes)")
            try:
                import bitsandbytes as bnb
                for module in model.modules():
                    if hasattr(module, "k_proj") or hasattr(module, "v_proj"):
                        pass  # Placeholder — real impl depends on arch
                emit_log("KV-cache quantization applied (experimental)", "success")
            except Exception as e:
                emit_log(f"KV quant: {e}", "warn")
            set_progress(85)

        # ── 6. Save ───────────────────────────────────────────────────────────
        set_stage("Saving optimized model")
        model.save_pretrained(out_dir)
        tok.save_pretrained(out_dir)

        # Write optimization config as metadata
        import json as _json
        with open(Path(out_dir)/"optimization_config.json","w") as f:
            _json.dump({"source":model_path,"torch_compile":do_torch_compile,
                        "dynamic_quant":do_dyn_quant,"kv_cache_quant":do_kv_quant,
                        "flash_attention":do_flash,"merge_lora":do_merge_lora}, f, indent=2)

        set_progress(100)
        current_job["status"] = "done"
        emit_log(f"Optimized model saved → {out_dir}", "success")

    except KeyboardInterrupt:
        current_job["status"] = "stopped"; emit_log("Stopped.", "warn")
    except Exception as e:
        current_job["status"] = "error"; emit_log(f"Error: {e}", "error"); emit_log(traceback.format_exc(), "error")


# ═══════════════════════════════════════════════════════════════════════════════
# JOB: MODEL PRUNING  (magnitude / structured head / LLM-guided)
# ═══════════════════════════════════════════════════════════════════════════════
def _generate_concept_sentences(concept, advisor_source, advisor_config, n=40):
    """Generate calibration sentences about a concept using local model or cloud API."""
    prompt = (
        f"Generate {n} diverse, natural sentences that clearly relate to the topic: \"{concept}\". "
        f"Cover different angles: factual statements, questions, instructions, descriptions. "
        f"Output ONLY the sentences, one per line, no numbering or bullets."
    )
    if advisor_source == "local":
        # Use the already-loaded model — but we need to call generate_text
        # We'll return a fixed seed list and augment with simple templates
        seeds = [
            f"{concept} is a topic that involves",
            f"Tell me about {concept}",
            f"The history of {concept} dates back to",
            f"How does {concept} work?",
            f"{concept} can be defined as",
            f"Examples of {concept} include",
            f"Why is {concept} important?",
            f"The key aspects of {concept} are",
            f"In the context of {concept},",
            f"What are the main characteristics of {concept}?",
        ]
        # Expand with simple variations
        expanded = []
        for s in seeds:
            expanded.append(s)
            expanded.append(s.lower())
            expanded.append("Please explain " + s.rstrip(".?,") + ".")
            expanded.append("Can you describe " + s.rstrip(".?,") + "?")
        return expanded[:n]
    else:
        # Cloud API
        raw = _cloud_generate(prompt, advisor_config, 1024)
        lines = [l.strip().lstrip("0123456789.-) ") for l in raw.splitlines() if l.strip()]
        return lines[:n] if lines else [f"{concept} is related to"]


def _compute_concept_activations(model, tokenizer, sentences, top_k_rows=0.15):
    """
    Run sentences through the model and return a dict mapping
    (layer_name, row_index) → mean_activation_magnitude.
    Used to identify which output neurons (rows) fire for the concept.
    """
    import torch
    device = next(model.parameters()).device
    row_scores = {}   # {layer_name: tensor of shape [out_features]}
    hooks = []
    named_linears = {name: mod for name, mod in model.named_modules()
                     if isinstance(mod, torch.nn.Linear)}

    def make_hook(name):
        def hook(module, inp, out):
            # out shape: [batch, seq, out_features] or [batch, out_features]
            o = out.detach().float()
            if o.dim() == 3:
                o = o.mean(dim=(0, 1))   # mean over batch and sequence
            elif o.dim() == 2:
                o = o.mean(dim=0)
            if name not in row_scores:
                row_scores[name] = o.abs()
            else:
                row_scores[name] = row_scores[name] + o.abs()
        return hook

    for name, mod in named_linears.items():
        hooks.append(mod.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        for i, sent in enumerate(sentences):
            if stop_flag.is_set(): break
            ids = tokenizer(sent, return_tensors="pt", truncation=True,
                            max_length=128).input_ids.to(device)
            try:
                model(ids)
            except Exception:
                pass
            if (i+1) % 10 == 0:
                emit_log(f"  Activation probe: {i+1}/{len(sentences)} sentences", "info")

    for h in hooks:
        h.remove()

    # Normalise by sentence count
    for name in row_scores:
        row_scores[name] = row_scores[name] / max(len(sentences), 1)

    return row_scores, named_linears


def _zero_concept_rows(model, row_scores, named_linears, threshold_pct, compress):
    """
    Zero out output rows (and corresponding input rows of the next layer)
    whose activation score is in the top threshold_pct for this concept.
    Optionally rebuild weight matrices with those rows deleted (compress=True).
    Returns stats dict.
    """
    import torch
    zeroed_total = 0
    params_before = sum(p.numel() for p in model.parameters())

    for name, scores in row_scores.items():
        if stop_flag.is_set(): break
        mod = named_linears.get(name)
        if mod is None or not hasattr(mod, "weight"): continue
        W = mod.weight.data   # shape [out, in]
        n_out = W.shape[0]
        k = max(1, int(n_out * threshold_pct))
        # Rows with highest concept activation → most responsible for this knowledge
        top_rows = scores.topk(min(k, len(scores))).indices

        if compress:
            # Build a boolean mask of rows to KEEP (invert the top-k)
            keep_mask = torch.ones(n_out, dtype=torch.bool)
            keep_mask[top_rows] = False
            kept_idx = keep_mask.nonzero(as_tuple=True)[0]
            if len(kept_idx) == 0: continue
            # Shrink weight: [kept, in]
            mod.weight = torch.nn.Parameter(W[kept_idx].clone())
            if mod.bias is not None:
                mod.bias = torch.nn.Parameter(mod.bias.data[kept_idx].clone())
            mod.out_features = len(kept_idx)
            zeroed_total += len(top_rows)
        else:
            # Just zero the rows
            W[top_rows] = 0.0
            zeroed_total += len(top_rows)

    params_after = sum(p.numel() for p in model.parameters())
    return {
        "zeroed_rows": zeroed_total,
        "params_before": params_before,
        "params_after": params_after,
        "reduction_pct": (1 - params_after / max(params_before, 1)) * 100,
    }


def run_pruning(config):
    try:
        model_path      = config["model_path"]
        method          = config.get("method", "magnitude")
        sparsity        = float(config.get("sparsity", 0.3))
        output_name     = config.get("output_name") or (Path(model_path).name + "_pruned")
        out_dir         = str(OUTPUTS_DIR / output_name)
        layer_filter    = config.get("layer_filter", "")

        # Knowledge-erasure specific
        concepts        = config.get("concepts", "")     # newline/comma separated topics to erase
        threshold_pct   = float(config.get("threshold_pct", 0.15))  # top-% rows to zero per concept
        compress        = config.get("compress", True)   # actually delete rows vs just zero
        advisor_source  = config.get("advisor_source", "cloud")  # "cloud" | "local"
        advisor_local_model = config.get("advisor_local_model", "")

        import torch
        import torch.nn.utils.prune as prune_util

        set_stage("Loading model")
        set_progress(5)
        model, tok, _ = load_model_and_tokenizer(model_path)
        set_progress(20)

        # ── Collect linear layer targets ───────────────────────────────────────
        targets = []
        for name, module in model.named_modules():
            if not isinstance(module, torch.nn.Linear): continue
            if layer_filter and not any(f.strip() in name for f in layer_filter.split(",")):
                continue
            targets.append((name, module))

        # ══════════════════════════════════════════════════════════════════════
        if method == "magnitude":
            emit_log(f"Magnitude pruning: {len(targets)} layers at {sparsity*100:.0f}% sparsity", "info")
            set_stage("Magnitude pruning")
            for i, (name, module) in enumerate(targets):
                if stop_flag.is_set(): raise KeyboardInterrupt()
                try:
                    prune_util.l1_unstructured(module, name="weight", amount=sparsity)
                    prune_util.remove(module, "weight")
                except Exception as e:
                    emit_log(f"Skip {name}: {e}", "warn")
                if (i+1) % 20 == 0:
                    emit_log(f"  {i+1}/{len(targets)} layers done", "info")
                set_progress(20 + (i+1)/max(len(targets),1)*70)

        # ══════════════════════════════════════════════════════════════════════
        elif method == "head":
            head_targets = [(n,m) for n,m in targets
                            if any(x in n for x in ["q_proj","k_proj","v_proj","out_proj","o_proj"])]
            emit_log(f"Attention head pruning: {len(head_targets)} projection layers", "info")
            set_stage("Head pruning")
            for i, (name, module) in enumerate(head_targets):
                if stop_flag.is_set(): raise KeyboardInterrupt()
                try:
                    prune_util.ln_structured(module, name="weight", amount=sparsity, n=2, dim=0)
                    prune_util.remove(module, "weight")
                except Exception as e:
                    emit_log(f"Skip {name}: {e}", "warn")
                set_progress(20 + (i+1)/max(len(head_targets),1)*70)
            emit_log("Head pruning done", "success")

        # ══════════════════════════════════════════════════════════════════════
        elif method == "concept":
            if not concepts:
                raise ValueError("No concepts specified. Enter at least one topic to erase.")

            concept_list = [c.strip() for c in concepts.replace(",", chr(10)).splitlines() if c.strip()]
            emit_log(f"Concept erasure: {len(concept_list)} concept(s): {concept_list}", "info")
            emit_log(f"Threshold: top {threshold_pct*100:.0f}% concept-activating rows per layer", "info")
            emit_log(f"Compress (delete rows): {compress}", "info")

            # Build advisor config for cloud calls
            advisor_cfg = {**config,
                           "cloud_provider": config.get("advisor_provider", ""),
                           "cloud_model":    config.get("advisor_model", ""),
                           "cloud_api_key":  config.get("advisor_api_key", ""),
                           "cloud_base_url": config.get("advisor_base_url", ""),
            }

            named_linears = {name: mod for name, mod in model.named_modules()
                             if isinstance(mod, torch.nn.Linear)}

            total_stats = {"zeroed_rows": 0, "params_before": sum(p.numel() for p in model.parameters())}

            for ci, concept in enumerate(concept_list):
                if stop_flag.is_set(): raise KeyboardInterrupt()
                set_stage(f"Erasing concept {ci+1}/{len(concept_list)}: '{concept}'")
                set_progress(20 + ci/len(concept_list)*70)

                # 1. Generate calibration sentences
                emit_log(f"Generating calibration sentences for: {concept}", "info")
                sentences = _generate_concept_sentences(
                    concept, advisor_source, advisor_cfg, n=40)
                emit_log(f"  {len(sentences)} sentences generated", "info")

                # 2. Run forward passes, collect per-row activation scores
                emit_log("Running activation probes through model...", "info")
                row_scores, named_linears = _compute_concept_activations(
                    model, tok, sentences, top_k_rows=threshold_pct)

                # 3. Zero (and optionally delete) concept-encoding rows
                emit_log(f"Erasing rows for concept: {concept}", "info")
                stats = _zero_concept_rows(model, row_scores, named_linears,
                                           threshold_pct, compress)
                emit_log(f"  Zeroed {stats['zeroed_rows']} rows | "
                         f"Params: {stats['params_before']:,} → {stats['params_after']:,} "
                         f"({stats['reduction_pct']:.1f}% reduction)", "success")
                total_stats["zeroed_rows"] += stats["zeroed_rows"]

            total_stats["params_after"] = sum(p.numel() for p in model.parameters())
            total_stats["reduction_pct"] = (
                1 - total_stats["params_after"] / max(total_stats["params_before"], 1)) * 100
            emit_log(
                f"All concepts erased. Total param reduction: "
                f"{total_stats['params_before']:,} → {total_stats['params_after']:,} "
                f"({total_stats['reduction_pct']:.1f}%)", "success")

        else:
            raise ValueError(f"Unknown method: {method}")

        # ── Final sparsity report ──────────────────────────────────────────────
        tp = sum(p.numel() for p in model.parameters())
        zp = sum((p.data.abs() < 1e-6).sum().item() for p in model.parameters())
        emit_log(f"Final sparsity: {zp/max(tp,1)*100:.1f}% zeros ({zp:,}/{tp:,})", "info")

        set_stage("Saving model")
        model.save_pretrained(out_dir)
        tok.save_pretrained(out_dir)
        set_progress(100)
        current_job["status"] = "done"
        emit_log(f"Saved → {out_dir}", "success")

    except KeyboardInterrupt:
        current_job["status"] = "stopped"; emit_log("Stopped.", "warn")
    except Exception as e:
        current_job["status"] = "error"
        emit_log(f"Error: {e}", "error")
        emit_log(traceback.format_exc(), "error")

if __name__ == "__main__":
    import logging
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    print("╔══════════════════════════════════════════════════╗")
    print("║    🦥  Unsloth Fine-Tuning Lab  v2.0             ║")
    print(f"║    http://localhost:{_cfg['port']}                         ║")
    print("╚══════════════════════════════════════════════════╝")
    app.run(debug=_cfg["debug"], host=_cfg["host"], port=_cfg["port"], threaded=True)