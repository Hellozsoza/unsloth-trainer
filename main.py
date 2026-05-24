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

# ─── Disable torch.compile (fixes Phi-4 LongRoPE + dynamo crash) ─────────────
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
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

# ─── Data directories ─────────────────────────────────────────────────────────
BASE_DATA_DIR = Path("/mnt/f/unsloth")
MODELS_DIR    = BASE_DATA_DIR / "models"
DATASETS_DIR  = BASE_DATA_DIR / "datasets"
OUTPUTS_DIR   = BASE_DATA_DIR / "outputs"
GEN_DIR       = BASE_DATA_DIR / "generated_datasets"
CLOUD_LOGS_DIR = BASE_DATA_DIR / "cloud_logs"
CLOUD_LOGS_DIR.mkdir(parents=True, exist_ok=True)
CLOUD_LOGS_DIR = BASE_DATA_DIR / "cloud_logs"
TOKEN_FILE    = BASE_DATA_DIR / ".hf_token"
TUNING_LOGS_FILE = BASE_DATA_DIR / "tuning_logs.json"

for d in [MODELS_DIR, DATASETS_DIR, OUTPUTS_DIR, GEN_DIR]:
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
        run_ts       = time.strftime("%Y-%m-%d_%H-%M-%S")
        out_name     = config.get("output_name") or f"autodistill_{mode}_{run_ts}"

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
        model = apply_lora(
            model, r=lora_r, alpha=lora_r,
            use_loftq=bool(config.get("use_loftq", False)),
            loftq_bits=cfg_int(config.get("loftq_bits"), 4),
            loftq_iter=cfg_int(config.get("loftq_iter"), 1),
        )
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
                    "LlamaForCausalLM":   "llama",      "MistralForCausalLM":    "mistral",
                    "GemmaForCausalLM":   "gemma",      "Gemma2ForCausalLM":     "gemma2",
                    "PhiForCausalLM":     "phi",        "Phi3ForCausalLM":       "phi3",
                    "Qwen2ForCausalLM":   "qwen2",      "Qwen3ForCausalLM":      "qwen2",
                    "GPT2LMHeadModel":       "gpt2",
                    "GPTNeoXForCausalLM": "gpt_neox",   "FalconForCausalLM":     "falcon",
                    "MixtralForCausalLM":       "mixtral",
                    "Qwen2MoeForCausalLM":      "qwen2_moe",
                    "DeepseekV2ForCausalLM":    "deepseek_v2",
                    "DeepseekV3ForCausalLM":    "deepseek_v3",
                    "OlmoeForCausalLM":         "olmoe",
                    "JambaForCausalLM":         "jamba",
                    "ArcticForCausalLM":        "arctic",
                    "PhiMoEForCausalLM":        "phimoe",
                    "Granitemoe10bForCausalLM": "granitemoe",
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

    # Helper: sanitize problematic fields in JSON config files inside a local model
    # directory so that plain transformers can load unsloth bnb-4bit checkpoints.
    # Unsloth stores torch dtype objects (not strings) in quantization_config and
    # sometimes tokenizer_config, causing "not a string" errors in transformers.
    def _sanitize_local_configs(path):
        """Patch config.json and tokenizer_config.json in-place, removing or
        stringifying fields that cause 'not a string' errors.
        Returns a restore callable that reverts all changes."""
        import json as _j
        restores = []
        model_dir = Path(path)

        def _patch_file(file_path, removals=(), dtype_fields=()):
            if not file_path.exists():
                return
            original = file_path.read_text()
            try:
                cfg = _j.loads(original)
            except Exception:
                return
            changed = False
            for key in removals:
                if key in cfg:
                    del cfg[key]; changed = True
            # Stringify any dtype fields that may be torch objects serialized oddly
            for key in dtype_fields:
                if key in cfg and not isinstance(cfg[key], str):
                    cfg[key] = str(cfg[key]); changed = True
            if changed:
                file_path.write_text(_j.dumps(cfg, indent=2))
                emit_log(f"Sanitized {file_path.name} for transformers fallback", "info")
                restores.append((file_path, original))

        _patch_file(model_dir / "config.json",
                    removals=("quantization_config",))
        _patch_file(model_dir / "tokenizer_config.json",
                    dtype_fields=("model_input_names",),
                    removals=())

        def _restore_all():
            for fp, orig in restores:
                try: fp.write_text(orig)
                except Exception: pass

        return _restore_all

    # Attempt 3: BitsAndBytes 4-bit — no CPU memory cap so full VRAM is used
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        emit_log("Falling back to transformers 4-bit...", "warn")
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4")
        _restore3 = _sanitize_local_configs(_resolved)
        try:
            # str() + use_fast=True: avoids Python 3.13 sentencepiece crash
            # where vocab_file Path object causes "not a string" TypeError
            tokenizer = AutoTokenizer.from_pretrained(
                str(_resolved), token=token, use_fast=True)
            model     = AutoModelForCausalLM.from_pretrained(
                str(_resolved), quantization_config=bnb_cfg,
                device_map="auto", token=token)
        finally:
            _restore3()
        emit_log(f"Loaded via transformers (4-bit)", "success")
        _active_model["model"] = model; _active_model["tokenizer"] = tokenizer
        return model, tokenizer, "transformers_bnb"
    except Exception as e:
        errors["Transformers_BnB"] = str(e)
        emit_log(f"BnB fallback failed: {e}", "warn")
        emit_log(traceback.format_exc(), "warn")

    # Attempt 4: float16 last resort
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        emit_log("Last resort: transformers float16...", "warn")
        _restore4 = _sanitize_local_configs(_resolved)
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                str(_resolved), token=token, use_fast=True)
            model     = AutoModelForCausalLM.from_pretrained(
                str(_resolved), torch_dtype=torch.float16,
                device_map="auto", token=token)
        finally:
            _restore4()
        emit_log(f"Loaded via transformers (float16)", "success")
        _active_model["model"] = model; _active_model["tokenizer"] = tokenizer
        return model, tokenizer, "transformers"
    except Exception as e:
        errors["Transformers_fp16"] = str(e)
        emit_log(traceback.format_exc(), "warn")

    error_summary = "\n".join(f"  {k}: {v}" for k, v in errors.items())
    raise RuntimeError(f"All loading methods failed for '{model_name}'.\n{error_summary}")

def _get_lora_target_modules(model):
    """Auto-detect LoRA target modules including MoE expert projection layers."""
    all_leaf = {name.split(".")[-1] for name, _ in model.named_modules()}
    base = ["q_proj", "k_proj", "v_proj", "o_proj"]
    for m in ["gate_proj", "up_proj", "down_proj", "fc1", "fc2"]:
        if m in all_leaf: base.append(m)
    for m in ["w1", "w2", "w3", "shared_expert_gate"]:
        if m in all_leaf: base.append(m)
    seen, result = set(), []
    for m in base:
        if m not in seen: seen.add(m); result.append(m)
    emit_log(f"LoRA target modules: {result}", "info")
    return result


def apply_lora(model, r=16, alpha=16, dropout=0.0, use_loftq=False, loftq_bits=4, loftq_iter=1):
    """Apply LoRA — auto-detects MoE expert projection layers (w1/w2/w3).
    If use_loftq=True, initializes LoRA weights using LoftQ quantization
    (better starting point than random init for quantized models).
    loftq_bits: quantization bit-width (2, 4, 8). Default 4.
    loftq_iter: number of alternating optimization iterations. Default 1.
    """
    target_modules = _get_lora_target_modules(model)
    if use_loftq:
        emit_log(f"LoftQ enabled (bits={loftq_bits}, iters={loftq_iter}) — using PEFT direct path", "info")
    else:
        try:
            from unsloth import FastModel
            return FastModel.get_peft_model(
                model, r=r, lora_alpha=alpha, lora_dropout=dropout,
                target_modules=target_modules,
                bias="none", use_gradient_checkpointing="unsloth", random_state=42)
        except Exception as e1:
            emit_log(f"FastModel LoRA failed ({e1}), trying FastLanguageModel...", "warn")
        try:
            from unsloth import FastLanguageModel
            return FastLanguageModel.get_peft_model(
                model, r=r, lora_alpha=alpha, lora_dropout=dropout,
                target_modules=target_modules,
                bias="none", use_gradient_checkpointing="unsloth", random_state=42)
        except Exception as e2:
            emit_log(f"FastLanguageModel LoRA failed ({e2}), falling back to PEFT...", "warn")
    from peft import get_peft_model, LoraConfig, TaskType
    if use_loftq:
        try:
            from peft import LoftQConfig
            loftq_config = LoftQConfig(loftq_bits=loftq_bits, loftq_iter=loftq_iter)
            lora_config = LoraConfig(
                r=r, lora_alpha=alpha, lora_dropout=dropout,
                target_modules=target_modules, bias="none", task_type=TaskType.CAUSAL_LM,
                init_lora_weights="loftq", loftq_config=loftq_config,
            )
            emit_log(f"LoftQ LoraConfig ready (bits={loftq_bits}, iters={loftq_iter})", "success")
        except ImportError:
            emit_log("peft.LoftQConfig not available — upgrade peft>=0.7. Falling back to standard LoRA init.", "warn")
            lora_config = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout,
                target_modules=target_modules, bias="none", task_type=TaskType.CAUSAL_LM)
    else:
        lora_config = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout,
            target_modules=target_modules, bias="none", task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    emit_log("Applied LoRA via PEFT directly", "success")
    return model
