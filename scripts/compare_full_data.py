"""Compare the full data BERT-Mini run with TF-IDF trained on the same 66,349 SST-2 rows.

    PYTHONPATH=src python scripts/compare_full_data.py
"""
import json
from pathlib import Path

import numpy as np

from smft.baseline import tfidf_baseline
from smft.data import eval_sets, sst2_train_dev
from smft.metrics import bootstrap_ci, mcnemar, paired_bootstrap_diff

run = Path("results/reproduce_turc2019")
cfg = json.loads((run / "metrics.json").read_text())["config"]
train, dev = sst2_train_dev(cfg["train_size"], cfg["dev_size"], cfg["data_seed"])
sets = eval_sets(cfg["imdb_eval_size"])
base = tfidf_baseline(train, dev, sets)
preds = np.load(run / "predictions.npz")
out = {"train_rows": len(train), "tfidf_C": base["metrics"]["C_chosen_on_dev"]}
for s in sets:
    y = np.array(s.labels)
    a = ((preds[s.name] >= 0.5) == y).astype(int)
    b = ((base["probs"][s.name] >= 0.5) == y).astype(int)
    out[s.name] = {"bert_mini_accuracy": float(a.mean()), "bert_mini_95ci": bootstrap_ci(a),
                   "tfidf_accuracy": float(b.mean()), "tfidf_95ci": bootstrap_ci(b),
                   "diff_95ci": paired_bootstrap_diff(a, b), "mcnemar": mcnemar(a, b)}
Path("results/full_data_vs_tfidf.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=1))
