"""Measure text overlap between SST-2 train and each evaluation set (contamination check)."""
import json
import re
from pathlib import Path

from datasets import load_dataset

from smft.data import eval_sets


def norm(t):
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", t.lower())).strip()


train = {norm(t) for t in load_dataset("stanfordnlp/sst2", split="train")["sentence"]}
out = {}
for s in eval_sets(2000):
    n = [norm(t) for t in s.texts]
    out[s.name] = {"examples": len(n), "exact_match_in_sst2_train": sum(x in train for x in n)}
    out[s.name]["share"] = out[s.name]["exact_match_in_sst2_train"] / len(n)
Path("results/overlap_with_sst2_train.json").write_text(json.dumps(out, indent=2))
print(out)

# Accuracy on the Rotten Tomatoes rows that do not appear in SST-2 train
import numpy as np  # noqa: E402

rt = next(s for s in eval_sets(2000) if s.name == "rotten_tomatoes_test")
clean = np.array([norm(t) not in train for t in rt.texts])
y = np.array(rt.labels)
for run in ["results/main", "results/reproduce_turc2019"]:
    p = np.load(f"{run}/predictions.npz")["rotten_tomatoes_test"]
    c = (p >= 0.5) == y
    out["rotten_tomatoes_test"][run] = {"accuracy_all": float(c.mean()), "accuracy_overlapping": float(c[~clean].mean()),
                                        "accuracy_clean": float(c[clean].mean()), "clean_rows": int(clean.sum())}
Path("results/overlap_with_sst2_train.json").write_text(json.dumps(out, indent=2))
print(out["rotten_tomatoes_test"])
