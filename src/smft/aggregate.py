"""Aggregate ablation runs (mean and std over seeds), significance tests and plots.

    python -m smft.aggregate
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from smft.baseline import tfidf_baseline  # noqa: E402
from smft.data import eval_sets  # noqa: E402
from smft.metrics import bootstrap_ci, mcnemar, paired_bootstrap_diff, reliability_bins  # noqa: E402

RUNS = Path("results/runs")
OUT = Path("results")
SETS = ["sst2_validation", "rotten_tomatoes_test", "imdb_test_2000"]


def load_runs() -> dict[str, list[dict]]:
    groups = defaultdict(list)
    for p in sorted(RUNS.glob("*/metrics.json")):
        name = re.sub(r"_s\d+$", "", p.parent.name)
        rec = json.loads(p.read_text())
        rec["_dir"] = p.parent
        groups[name].append(rec)
    return groups


def summary_table(groups) -> pd.DataFrame:
    rows = []
    for name, recs in groups.items():
        row = {"experiment": name, "seeds": len(recs),
               "trainable_params": recs[0]["trainable_params"],
               "train_examples": recs[0]["train_examples"],
               "train_seconds_mean": np.mean([r["train_seconds"] for r in recs])}
        for s in SETS:
            for m in ("accuracy", "f1", "roc_auc", "ece"):
                v = [r["eval"][s][m] for r in recs]
                row[f"{s}_{m}_mean"] = float(np.mean(v))
                row[f"{s}_{m}_std"] = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values("experiment")


def fmt(df, name, s="sst2_validation", m="accuracy"):
    r = df[df.experiment == name].iloc[0]
    return f"{r[f'{s}_{m}_mean']:.3f} ± {r[f'{s}_{m}_std']:.3f}"


def plot_sweeps(df):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    sweeps = [
        ("training examples", [(500, "size_500"), (1000, "size_1000"), (2000, "size_2000"), (4000, "size_4000"),
                               (8000, "method_full"), (16000, "size_16000")], True),
        ("learning rate (full fine-tune)", [(1e-5, "lr_1.0e-5"), (2e-5, "lr_2.0e-5"), (5e-5, "method_full"),
                                            (1e-4, "lr_1.0e-4"), (2e-4, "lr_2.0e-4")], True),
        ("LoRA rank", [(2, "lora_r2"), (4, "lora_r4"), (8, "method_lora_r8"), (16, "lora_r16"), (32, "lora_r32")], True),
    ]
    for ax, (label, pts, logx) in zip(axes, sweeps):
        pts = [(x, n) for x, n in pts if (df.experiment == n).any()]
        for s in SETS:
            xs = [x for x, _ in pts]
            m = [df[df.experiment == n][f"{s}_accuracy_mean"].iloc[0] for _, n in pts]
            e = [df[df.experiment == n][f"{s}_accuracy_std"].iloc[0] for _, n in pts]
            ax.errorbar(xs, m, yerr=e, marker="o", capsize=3, label=s)
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(label)
        ax.set_ylabel("accuracy (mean ± std, 3 seeds)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "ablation_sweeps.png", dpi=120)
    plt.close(fig)


def plot_overfit(groups):
    names = [n for n in ["overfit_no_es", "overfit_es", "overfit_wd0_no_es", "overfit_wd0.1_no_es"] if n in groups]
    if not names:
        return {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    summary = {}
    for n in names:
        rec = groups[n][0]
        c = rec["curve"]
        ep = [p["epoch"] for p in c]
        if n == "overfit_no_es":
            axes[0].plot(ep, [p["train_loss"] for p in c], "o-", label="train loss")
            axes[0].plot(ep, [p["dev_loss"] for p in c], "s-", label="dev loss")
            axes[0].axvline(ep[int(np.argmin([p["dev_loss"] for p in c]))], color="grey", ls="--", label="lowest dev loss")
            axes[0].set_xlabel("epoch")
            axes[0].set_title("1,000 examples, 15 epochs, seed 0")
            axes[0].legend()
        axes[1].plot(ep, [p["train_accuracy"] - p["dev_accuracy"] for p in c], label=n)
        summary[n] = {
            "seeds": len(groups[n]),
            "final_train_dev_acc_gap_mean": float(np.mean([g["curve"][-1]["train_accuracy"] - g["curve"][-1]["dev_accuracy"] for g in groups[n]])),
            "epochs_run_mean": float(np.mean([g["curve"][-1]["epoch"] for g in groups[n]])),
            "sst2_validation_accuracy_mean": float(np.mean([g["eval"]["sst2_validation"]["accuracy"] for g in groups[n]])),
            "sst2_validation_accuracy_std": float(np.std([g["eval"]["sst2_validation"]["accuracy"] for g in groups[n]], ddof=1)),
            "sst2_validation_ece_mean": float(np.mean([g["eval"]["sst2_validation"]["ece"] for g in groups[n]])),
            "train_seconds_mean": float(np.mean([g["train_seconds"] for g in groups[n]])),
        }
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("train acc minus dev acc")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "overfitting_curves.png", dpi=120)
    plt.close(fig)
    return summary


def temperature_scale(dev_p, dev_y, test_p):
    """Fit one temperature on the SST-2 dev split, apply to a test split."""
    from scipy.optimize import minimize_scalar

    def to_logit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    zd = to_logit(dev_p)

    def nll(t):
        q = 1 / (1 + np.exp(-zd / t))
        return -np.mean(dev_y * np.log(q + 1e-12) + (1 - dev_y) * np.log(1 - q + 1e-12))

    t = minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x
    return float(t), 1 / (1 + np.exp(-to_logit(test_p) / t))


def main():
    groups = load_runs()
    df = summary_table(groups)
    df.to_csv(OUT / "ablation_summary.csv", index=False)
    plot_sweeps(df)
    overfit = plot_overfit(groups)

    from smft.data import sst2_train_dev
    from smft.metrics import classification_report

    main_rec = json.loads((OUT / "main/metrics.json").read_text())
    cfg = main_rec["config"]
    train, dev = sst2_train_dev(cfg["train_size"], cfg["dev_size"], cfg["data_seed"])
    sets = {s.name: s for s in eval_sets(cfg["imdb_eval_size"])}
    preds = np.load(OUT / "main/predictions.npz")
    base = tfidf_baseline(train, dev, list(sets.values()))

    sig, calib, slices = {}, {}, {}
    for name in SETS:
        y = np.array(sets[name].labels)
        p = preds[name]
        b = base["probs"][name]
        bert_c = ((p >= 0.5) == y).astype(int)
        base_c = ((b >= 0.5) == y).astype(int)
        sig[name] = {
            "bert_accuracy": float(bert_c.mean()), "bert_accuracy_95ci": bootstrap_ci(bert_c),
            "tfidf_accuracy": float(base_c.mean()), "tfidf_accuracy_95ci": bootstrap_ci(base_c),
            "diff_95ci": paired_bootstrap_diff(bert_c, base_c), "mcnemar": mcnemar(bert_c, base_c),
        }
        t, p_cal = temperature_scale(preds["sst2_dev"], np.array(dev.labels), p)
        before, after = classification_report(p, y), classification_report(p_cal, y)
        calib[name] = {"temperature": t,
                       "before": {k: before[k] for k in ("ece", "brier", "accuracy")},
                       "after": {k: after[k] for k in ("ece", "brier", "accuracy")},
                       "reliability_before": reliability_bins(p, y), "reliability_after": reliability_bins(p_cal, y)}
        # error slices: length and negation words
        texts = sets[name].texts
        lengths = np.array([len(x.split()) for x in texts])
        neg = np.array([bool(re.search(r"\b(not|no|never|n't|nothing|hardly)\b", x.lower())) for x in texts])
        q = np.quantile(lengths, [0, 0.25, 0.5, 0.75, 1.0])
        rows = []
        for lo, hi in zip(q[:-1], q[1:]):
            m = (lengths >= lo) & (lengths <= hi)
            rows.append({"slice": f"words {int(lo)}-{int(hi)}", "n": int(m.sum()), "accuracy": float(bert_c[m].mean())})
        for flag, label in [(True, "has negation word"), (False, "no negation word")]:
            m = neg == flag
            rows.append({"slice": label, "n": int(m.sum()), "accuracy": float(bert_c[m].mean())})
        slices[name] = rows

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, name in zip(axes, SETS):
        ax.plot([0, 1], [0, 1], "k:")
        for key, lab in [("reliability_before", "before"), ("reliability_after", "after temperature scaling")]:
            bins = [b for b in calib[name][key] if b["count"] >= 10]
            ax.plot([b["mean_predicted"] for b in bins], [b["observed_positive_rate"] for b in bins], "o-",
                    label=f"{lab} (ECE {calib[name]['after' if 'after' in key else 'before']['ece']:.3f})")
        ax.set_title(name)
        ax.set_xlabel("predicted P(positive)")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("observed positive rate")
    fig.tight_layout()
    fig.savefig(OUT / "reliability.png", dpi=120)
    plt.close(fig)

    repro = {}
    if (OUT / "repro_main/metrics.json").exists():
        r2 = json.loads((OUT / "repro_main/metrics.json").read_text())
        p2 = np.load(OUT / "repro_main/predictions.npz")
        repro = {s: {"run1_accuracy": main_rec["eval"][s]["accuracy"], "run2_accuracy": r2["eval"][s]["accuracy"],
                     "max_abs_prob_diff": float(np.abs(p2[s] - preds[s]).max())} for s in SETS}

    report = {"main_run": {k: main_rec[k] for k in ("config", "trainable_params", "total_params", "train_examples",
                                                     "dev_examples", "train_seconds", "best_step", "eval", "hardware")},
              "reproducibility_same_seed_twice": repro,
              "tfidf_baseline": base["metrics"],
              "significance_vs_tfidf": sig, "calibration": calib, "error_slices": slices, "overfitting": overfit}
    (OUT / "evaluation_report.json").write_text(json.dumps(report, indent=2, default=float))
    print(df[["experiment", "seeds", "trainable_params", "train_examples"] +
             [f"{s}_accuracy_mean" for s in SETS] + [f"{s}_accuracy_std" for s in SETS]].round(4).to_string(index=False))
    print(json.dumps({"repro": repro, "sig": {k: {kk: v[kk] for kk in ("bert_accuracy", "tfidf_accuracy", "diff_95ci", "mcnemar")} for k, v in sig.items()},
                      "calib": {k: {"t": v["temperature"], "before": v["before"], "after": v["after"]} for k, v in calib.items()},
                      "overfit": overfit}, indent=1, default=float))


if __name__ == "__main__":
    main()
