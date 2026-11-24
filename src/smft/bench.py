"""CPU inference benchmark: PyTorch fp32, PyTorch dynamic int8, ONNX Runtime fp32 and int8.

    python -m smft.bench --model results/main/model --out results/inference

Each variant runs in its own process so peak memory is measured cleanly. There is
no GPU on the test machine, so memory is process RAM (peak RSS), not VRAM.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

VARIANTS = ["torch_fp32", "torch_int8_dynamic", "onnx_fp32", "onnx_int8_dynamic"]


def rss_mb() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1024
    return float("nan")


def peak_rss_mb() -> float:
    """Peak RSS since the last reset_peak() (VmHWM), falling back to ru_maxrss."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM"):
                return int(line.split()[1]) / 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def reset_peak() -> bool:
    try:
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")
        return True
    except OSError:
        return False


def export_onnx(model_dir: Path, out_dir: Path) -> dict:
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    out_dir.mkdir(parents=True, exist_ok=True)
    fp32 = out_dir / "model_fp32.onnx"
    int8 = out_dir / "model_int8.onnx"
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
    enc = tok(["an example sentence"], return_tensors="pt")
    torch.onnx.export(
        model, (enc["input_ids"], enc["attention_mask"], enc["token_type_ids"]), fp32,
        input_names=["input_ids", "attention_mask", "token_type_ids"], output_names=["logits"],
        dynamic_axes={k: {0: "batch", 1: "seq"} for k in ["input_ids", "attention_mask", "token_type_ids"]}
        | {"logits": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)
    return {"onnx_fp32": fp32, "onnx_int8_dynamic": int8}


def run_variant(variant: str, model_dir: Path, onnx_dir: Path, threads: int, batch_sizes: list[int], n_eval: int) -> dict:
    base_rss = rss_mb()
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(threads)
    tok = AutoTokenizer.from_pretrained(model_dir)
    ds = load_dataset("stanfordnlp/sst2", split="validation")
    texts, labels = list(ds["sentence"])[:n_eval], np.array(ds["label"][:n_eval])
    import gc

    gc.collect()
    libs_rss = rss_mb()
    peak_reset = reset_peak()

    if variant.startswith("torch"):
        model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
        if variant == "torch_int8_dynamic":
            model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        tmp = onnx_dir / f"{variant}.pt"
        torch.save(model.state_dict(), tmp)
        size_mb = tmp.stat().st_size / 1e6
        tmp.unlink()

        def infer(enc):
            with torch.inference_mode():
                return torch.softmax(model(**enc).logits, -1)[:, 1].numpy()

        def encode(batch):
            return tok(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
    else:
        import onnxruntime as ort

        path = onnx_dir / ("model_fp32.onnx" if variant == "onnx_fp32" else "model_int8.onnx")
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        size_mb = path.stat().st_size / 1e6

        def infer(enc):
            logits = sess.run(["logits"], {k: v for k, v in enc.items()})[0]
            e = np.exp(logits - logits.max(-1, keepdims=True))
            return (e / e.sum(-1, keepdims=True))[:, 1]

        def encode(batch):
            return dict(tok(batch, padding=True, truncation=True, max_length=128, return_tensors="np"))
    model_rss = rss_mb()

    # accuracy pass over the whole evaluation set
    probs = np.concatenate([infer(encode(texts[i:i + 64])) for i in range(0, len(texts), 64)])
    acc = float(((probs >= 0.5) == labels).mean())

    latency = {}
    for bs in batch_sizes:
        batches = [texts[i:i + bs] for i in range(0, len(texts) - bs + 1, bs)]
        if bs == 1:
            batches = batches[:300]
        encs = [encode(b) for b in batches]
        for e in encs[:10]:
            infer(e)
        times = []
        for e in encs:
            t = time.perf_counter()
            infer(e)
            times.append(time.perf_counter() - t)
        times = np.array(times) * 1000
        latency[str(bs)] = {
            "batches": len(times),
            "p50_ms": float(np.percentile(times, 50)),
            "p95_ms": float(np.percentile(times, 95)),
            "mean_ms": float(times.mean()),
            "throughput_sent_per_s": float(bs * len(times) / (times.sum() / 1000)),
        }
    return {
        "variant": variant,
        "size_mb": size_mb,
        "accuracy_sst2_validation": acc,
        "eval_examples": len(labels),
        "rss_mb": {"before_imports": base_rss, "after_libs": libs_rss, "after_model_load": model_rss,
                   "model_load_delta": model_rss - libs_rss, "peak_after_libs": peak_rss_mb(),
                   "peak_delta_over_libs": peak_rss_mb() - libs_rss, "peak_reset_ok": peak_reset},
        "latency": latency,
        "probs": probs.tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=Path("results/main/model"))
    ap.add_argument("--out", type=Path, default=Path("results/inference"))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--batch-sizes", type=int, nargs="*", default=[1, 8, 32])
    ap.add_argument("--n-eval", type=int, default=872)
    ap.add_argument("--variant", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--onnx-dir", type=Path, default=Path("/tmp/smft_onnx"))
    args = ap.parse_args()

    if args.variant:
        res = run_variant(args.variant, args.model, args.onnx_dir, args.threads, args.batch_sizes, args.n_eval)
        print("RESULT " + json.dumps(res))
        return

    args.out.mkdir(parents=True, exist_ok=True)
    export_onnx(args.model, args.onnx_dir)
    results = []
    for v in VARIANTS:
        cmd = [sys.executable, "-m", "smft.bench", "--variant", v, "--model", str(args.model),
               "--onnx-dir", str(args.onnx_dir), "--threads", str(args.threads), "--n-eval", str(args.n_eval),
               "--batch-sizes", *map(str, args.batch_sizes)]
        out = subprocess.run(cmd, capture_output=True, text=True, env=os.environ, check=True).stdout
        results.append(json.loads(next(line for line in out.splitlines() if line.startswith("RESULT "))[7:]))
    ref = np.array(results[0]["probs"])
    for r in results:
        p = np.array(r.pop("probs"))
        r["agreement_with_torch_fp32"] = float(((p >= 0.5) == (ref >= 0.5)).mean())
        r["max_abs_prob_diff_vs_torch_fp32"] = float(np.abs(p - ref).max())
        r["accuracy_drop_vs_torch_fp32"] = results[0]["accuracy_sst2_validation"] - r["accuracy_sst2_validation"]
    report = {
        "hardware": {"device": "cpu (no GPU on this machine, so memory is RAM not VRAM)",
                     "cpu_count": os.cpu_count(), "threads_used": args.threads,
                     "platform": platform.platform(), "python": platform.python_version()},
        "max_length": 128,
        "padding": "dynamic, to the longest sentence in each batch",
        "results": results,
    }
    try:
        import onnxruntime
        import torch

        report["hardware"]["torch"] = torch.__version__
        report["hardware"]["onnxruntime"] = onnxruntime.__version__
        cpu = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
        report["hardware"]["cpu_model"] = next((ln.split(":", 1)[1].strip() for ln in cpu.splitlines() if ln.startswith("Model name")), "")
    except Exception:
        pass
    (args.out / "benchmark.json").write_text(json.dumps(report, indent=2))
    rows = []
    for r in results:
        for bs, lat in r["latency"].items():
            rows.append({"variant": r["variant"], "batch_size": int(bs), **{k: round(v, 3) for k, v in lat.items() if k != "batches"},
                         "size_mb": round(r["size_mb"], 2), "model_load_rss_mb": round(r["rss_mb"]["model_load_delta"], 1),
                         "peak_rss_over_baseline_mb": round(r["rss_mb"]["peak_delta_over_libs"], 1), "accuracy": round(r["accuracy_sst2_validation"], 4),
                         "agreement_with_fp32": round(r["agreement_with_torch_fp32"], 4)})
    import pandas as pd

    pd.DataFrame(rows).to_csv(args.out / "benchmark.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
