# mtp_trainer.py  — imported by main.py
# Multi-Token Prediction (MTP) heads training

import traceback
import json as _json
from pathlib import Path as _Path


def run_mtp(config, OUTPUTS_DIR, stop_flag, current_job,
            emit_log, set_stage, set_progress, load_model_and_tokenizer):
    """
    Add MTP prediction heads to an existing model via lightweight fine-tuning.
    Frozen backbone + (optionally shared) heads trained on self-distilled soft targets.
    """
    try:
        import torch
        import torch.nn as nn

        model_path = config["model_path"]
        output_name = config.get("output_name") or (_Path(model_path).name + "_mtp")
        out_dir = str(OUTPUTS_DIR / output_name)

        num_heads     = int(config.get("num_heads", 1))
        share_weights = bool(config.get("share_weights", True))
        freeze_backbone = bool(config.get("freeze_backbone", True))
        proj_ratio    = float(config.get("proj_ratio", 1.0))
        dataset_id    = config["dataset"]
        dataset_split = config.get("dataset_split", "train")
        steps         = int(config.get("steps", 3000))
        batch_size    = int(config.get("batch_size", 4))
        lr            = float(config.get("lr", 2e-4))
        max_seq_length = int(config.get("max_seq_length", 512))
        grad_accum    = int(config.get("grad_accum", 4))
        load_in_4bit  = bool(config.get("load_in_4bit", True))
        self_distill  = bool(config.get("self_distill", True))
        vocab_compress = bool(config.get("vocab_compress", False))
        vocab_k       = int(config.get("vocab_k", 32000))

        # Load model
        set_stage("Loading model")
        set_progress(5)
        emit_log(f"Loading {model_path} (4-bit={load_in_4bit})", "info")
        model, tok, _ = load_model_and_tokenizer(model_path, load_in_4bit=load_in_4bit)
        model.eval()
        hidden_size = model.config.hidden_size
        vocab_size  = model.config.vocab_size
        emit_log(f"Model loaded — hidden={hidden_size}, vocab={vocab_size}", "success")
        set_progress(15)

        # Freeze backbone
        if freeze_backbone:
            for p in model.parameters():
                p.requires_grad_(False)
            emit_log("Backbone frozen.", "info")

        # MTP head definition
        class MTPHead(nn.Module):
            def __init__(self, h, p, v):
                super().__init__()
                self.norm = nn.LayerNorm(h)
                self.proj = nn.Linear(h, p, bias=False)
                self.out  = nn.Linear(p, v, bias=False)
            def forward(self, x):
                return self.out(self.proj(self.norm(x)))

        proj_size = max(64, int(hidden_size * proj_ratio))
        if share_weights:
            shared_head  = MTPHead(hidden_size, proj_size, vocab_size).to(model.device)
            mtp_heads    = [shared_head] * num_heads
            trainable    = nn.ModuleList([shared_head])
        else:
            mtp_heads = nn.ModuleList(
                [MTPHead(hidden_size, proj_size, vocab_size).to(model.device)
                 for _ in range(num_heads)]
            )
            trainable = mtp_heads

        n_params = sum(p.numel() for p in trainable.parameters())
        emit_log(f"MTP heads: {num_heads}x, proj={proj_size}, params={n_params:,}", "success")
        set_progress(20)

        # Dataset
        set_stage("Loading dataset")
        emit_log(f"Dataset: {dataset_id} ({dataset_split})", "info")
        from datasets import load_dataset
        ds = load_dataset(dataset_id, split=dataset_split, streaming=True)
        sample = next(iter(ds.take(1)))
        text_col = next(
            (c for c in ["text","content","article","document","input"] if c in sample),
            list(sample.keys())[0]
        )
        emit_log(f"Text column: '{text_col}'", "info")
        set_progress(25)

        # Vocab compression mask
        mask = None
        if vocab_compress and vocab_k < vocab_size:
            with torch.no_grad():
                norms   = model.get_input_embeddings().weight.norm(dim=1)
                top_ids = norms.topk(vocab_k).indices
                mask    = torch.zeros(vocab_size, dtype=torch.bool, device=model.device)
                mask[top_ids] = True
            emit_log(f"Vocab compressed to top-{vocab_k}", "info")

        # Optimiser
        opt = torch.optim.AdamW(trainable.parameters(), lr=lr, weight_decay=0.01)
        trainable.train()
        loss_fn = nn.KLDivLoss(reduction="batchmean") if self_distill else nn.CrossEntropyLoss()

        # Training loop
        set_stage("Training MTP heads")
        ds_iter   = iter(ds)
        accum_loss = 0.0
        opt.zero_grad()

        for step in range(1, steps + 1):
            if stop_flag.is_set():
                raise KeyboardInterrupt()

            texts = []
            for _ in range(batch_size):
                try:
                    row = next(ds_iter)
                except StopIteration:
                    ds_iter = iter(ds)
                    row = next(ds_iter)
                texts.append(str(row.get(text_col, ""))[:max_seq_length * 6])

            enc = tok(
                texts, return_tensors="pt", max_length=max_seq_length,
                truncation=True, padding=True,
            ).to(model.device)

            ctx = torch.no_grad() if freeze_backbone else torch.enable_grad()
            with ctx:
                out = model(**enc, output_hidden_states=True)

            hs = out.hidden_states[-1]
            step_loss = torch.tensor(0.0, device=model.device)

            for k, head in enumerate(mtp_heads):
                if hs.shape[1] < k + 2:
                    continue
                logits     = head(hs[:, :-(k+1), :])
                target_ids = enc["input_ids"][:, (k+1):]

                if self_distill:
                    with torch.no_grad():
                        sl = out.logits[:, (k+1):, :]
                        if mask is not None:
                            sl = sl.masked_fill(~mask, -1e9)
                        soft = torch.softmax(sl / 0.9, dim=-1)
                    lp = torch.log_softmax(logits, dim=-1)
                    B, T, V = lp.shape
                    step_loss = step_loss + loss_fn(lp.reshape(B*T, V), soft.reshape(B*T, V).detach())
                else:
                    B, T, V = logits.shape
                    step_loss = step_loss + nn.functional.cross_entropy(
                        logits.reshape(B*T, V), target_ids.reshape(B*T),
                        ignore_index=tok.pad_token_id or -100,
                    )

            step_loss = step_loss / max(num_heads, 1)
            (step_loss / grad_accum).backward()
            accum_loss += step_loss.item()

            if step % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable.parameters(), 1.0)
                opt.step()
                opt.zero_grad()

            if step % 100 == 0:
                avg = accum_loss / 100
                accum_loss = 0.0
                emit_log(f"Step {step}/{steps} — loss: {avg:.4f}", "info")
                set_progress(25 + int(step / steps * 65))

        # Save
        set_stage("Saving")
        import os
        os.makedirs(out_dir, exist_ok=True)
        model.save_pretrained(out_dir)
        tok.save_pretrained(out_dir)

        mtp_state = {}
        if share_weights:
            mtp_state["mtp_head.0"] = mtp_heads[0].state_dict()
        else:
            for i, h in enumerate(mtp_heads):
                mtp_state[f"mtp_head.{i}"] = h.state_dict()
        torch.save(mtp_state, _Path(out_dir) / "mtp_heads.pt")

        with open(_Path(out_dir) / "mtp_config.json", "w") as f:
            _json.dump({
                "num_heads": num_heads, "share_weights": share_weights,
                "proj_ratio": proj_ratio, "proj_size": proj_size,
                "hidden_size": hidden_size, "vocab_size": vocab_size,
                "self_distill": self_distill,
                "vocab_compress": vocab_compress,
                "vocab_k": vocab_k if vocab_compress else None,
                "freeze_backbone": freeze_backbone,
                "trained_steps": steps, "source_model": model_path,
            }, f, indent=2)

        set_progress(100)
        current_job["status"] = "done"
        emit_log(f"MTP model saved to {out_dir}", "success")
        emit_log(f"  {num_heads} head(s), {n_params:,} params — mtp_heads.pt", "info")
        emit_log("  For llama.cpp: export to GGUF + enable --spec-type mtp", "info")

    except KeyboardInterrupt:
        current_job["status"] = "stopped"
        emit_log("Stopped.", "warn")
    except Exception as e:
        current_job["status"] = "error"
        emit_log(f"Error: {e}", "error")
        emit_log(traceback.format_exc(), "error")
