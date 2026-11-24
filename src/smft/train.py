"""Fine-tune a small BERT on SST-2 on CPU and evaluate it on three test sets.

Usage:
    python -m smft.train --config configs/base.yaml [--set key=value ...] --out results/runs/NAME
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from smft.data import eval_sets, sst2_train_dev
from smft.metrics import classification_report


def load_config(path: str, overrides: list[str]) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    for item in overrides:
        key, value = item.split("=", 1)
        cfg[key] = yaml.safe_load(value)
    return cfg


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def build_model(cfg: dict):
    model = AutoModelForSequenceClassification.from_pretrained(cfg["model"], num_labels=2)
    method = cfg["method"]
    if method == "frozen":
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith("classifier")
    elif method == "lora":
        from peft import LoraConfig, get_peft_model

        lcfg = LoraConfig(
            r=int(cfg["lora_r"]),
            lora_alpha=int(cfg.get("lora_alpha", 2 * cfg["lora_r"])),
            lora_dropout=float(cfg.get("lora_dropout", 0.1)),
            target_modules=["query", "value"],
            modules_to_save=["classifier"],
            task_type="SEQ_CLS",
        )
        model = get_peft_model(model, lcfg)
    elif method != "full":
        raise ValueError(f"unknown method {method}")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return model, trainable, total


def encode(tok, texts, max_length):
    return tok(texts, truncation=True, max_length=max_length, padding=False)


def make_loader(tok, texts, labels, max_length, batch_size, shuffle, seed):
    enc = encode(tok, texts, max_length)
    items = [
        {"input_ids": enc["input_ids"][i], "attention_mask": enc["attention_mask"][i], "labels": labels[i]}
        for i in range(len(labels))
    ]

    def collate(batch):
        return tok.pad(batch, return_tensors="pt")

    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(items, batch_size=batch_size, shuffle=shuffle, collate_fn=collate, generator=g)


@torch.no_grad()
def predict(model, loader) -> tuple[np.ndarray, float]:
    model.eval()
    probs, losses, n = [], 0.0, 0
    for batch in loader:
        out = model(**batch)
        losses += float(out.loss) * len(batch["labels"])
        n += len(batch["labels"])
        probs.append(torch.softmax(out.logits, -1)[:, 1].numpy())
    return np.concatenate(probs), losses / n


def run(cfg: dict, out_dir: Path, save_model: bool = False) -> dict:
    torch.set_num_threads(int(cfg.get("threads", 2)))
    seed_everything(int(cfg["seed"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    train, dev = sst2_train_dev(int(cfg["train_size"]), int(cfg["dev_size"]), int(cfg["data_seed"]))
    model, trainable, total = build_model(cfg)

    train_loader = make_loader(tok, train.texts, train.labels, cfg["max_length"], cfg["batch_size"], True, cfg["seed"])
    train_eval_loader = make_loader(tok, train.texts[:2000], train.labels[:2000], cfg["max_length"], 128, False, 0)
    dev_loader = make_loader(tok, dev.texts, dev.labels, cfg["max_length"], 128, False, 0)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(cfg["lr"]), weight_decay=float(cfg.get("weight_decay", 0.01)))
    steps = len(train_loader) * int(cfg["epochs"])
    sched = get_linear_schedule_with_warmup(opt, int(steps * float(cfg.get("warmup_ratio", 0.06))), steps)

    patience = int(cfg.get("early_stopping_patience", 0))
    evals_per_epoch = int(cfg.get("evals_per_epoch", 2))
    eval_every = max(1, math.ceil(len(train_loader) / evals_per_epoch))
    curve, best, best_state, bad, step = [], (float("inf"), -1), None, 0, 0
    stopped_early = False
    t0 = time.time()
    for epoch in range(int(cfg["epochs"])):
        for batch in train_loader:
            model.train()
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if step % eval_every == 0 or step == steps:
                tr_p, tr_loss = predict(model, train_eval_loader)
                dv_p, dv_loss = predict(model, dev_loader)
                point = {
                    "step": step,
                    "epoch": round(step / len(train_loader), 3),
                    "train_loss": tr_loss,
                    "train_accuracy": float(((tr_p >= 0.5) == np.array(train.labels[:2000])).mean()),
                    "dev_loss": dv_loss,
                    "dev_accuracy": float(((dv_p >= 0.5) == np.array(dev.labels)).mean()),
                }
                curve.append(point)
                if dv_loss < best[0]:
                    best, bad = (dv_loss, step), 0
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                else:
                    bad += 1
                if patience and bad >= patience:
                    stopped_early = True
                    break
        if stopped_early:
            break
    train_seconds = time.time() - t0
    if cfg.get("restore_best", True) and best_state is not None:
        model.load_state_dict(best_state)

    results = {}
    predictions = {}
    for split in eval_sets(int(cfg.get("imdb_eval_size", 2000))):
        loader = make_loader(tok, split.texts, split.labels, int(cfg.get("eval_max_length", 128)), 128, False, 0)
        p, loss = predict(model, loader)
        rep = classification_report(p, split.labels)
        rep["loss"] = loss
        results[split.name] = rep
        predictions[split.name] = p
    dev_p, _ = predict(model, dev_loader)
    predictions["sst2_dev"] = dev_p

    record = {
        "config": cfg,
        "trainable_params": trainable,
        "total_params": total,
        "train_examples": len(train),
        "dev_examples": len(dev),
        "train_seconds": round(train_seconds, 1),
        "best_step": best[1],
        "total_steps_planned": steps,
        "stopped_early": stopped_early,
        "curve": curve,
        "eval": results,
        "hardware": {
            "device": "cpu",
            "cpu": platform.processor() or platform.machine(),
            "cpu_count": os.cpu_count(),
            "torch_threads": torch.get_num_threads(),
            "torch": torch.__version__,
            "python": platform.python_version(),
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(record, indent=2))
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    np.savez_compressed(out_dir / "predictions.npz", **predictions)
    if save_model:
        if cfg["method"] == "lora":
            model = model.merge_and_unload()
        model.save_pretrained(out_dir / "model")
        tok.save_pretrained(out_dir / "model")
    return record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-model", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    rec = run(cfg, Path(args.out), args.save_model)
    summary = {k: {m: round(v[m], 4) for m in ("accuracy", "f1", "roc_auc")} for k, v in rec["eval"].items()}
    print(json.dumps({"train_seconds": rec["train_seconds"], "best_step": rec["best_step"], "eval": summary}, indent=1))


if __name__ == "__main__":
    main()
