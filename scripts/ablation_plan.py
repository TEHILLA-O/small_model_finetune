"""Print one training command per line for the ablation grid. Run with xargs -P.

PYTHONPATH=src python scripts/ablation_plan.py | xargs -P 3 -I{} sh -c "{}"
"""
import sys

SEEDS = [0, 1, 2]
PY = f"{sys.executable} -m smft.train --config configs/base.yaml"
runs = []


def add(name, **kv):
    for s in SEEDS:
        sets = " ".join(f"{k}={v}" for k, v in {**kv, "seed": s}.items())
        out = f"results/runs/{name}_s{s}"
        runs.append(f"test -f {out}/metrics.json || {PY} --set {sets} --out {out} > /dev/null 2>&1")


# 1. Method: full fine-tune vs LoRA vs frozen encoder, each at a sensible learning rate
add("method_full")
add("method_lora_r8", method="lora", lora_r=8, lr=5.0e-4)
add("method_frozen", method="frozen", lr=1.0e-3)
# 2. LoRA rank
for r in [2, 4, 16, 32]:
    add(f"lora_r{r}", method="lora", lora_r=r, lr=5.0e-4)
# 3. Learning rate for full fine-tuning
for lr in ["1.0e-5", "2.0e-5", "1.0e-4", "2.0e-4"]:
    add(f"lr_{lr}", lr=lr)
# 4. Training set size
for n in [500, 1000, 2000, 4000, 16000]:
    add(f"size_{n}", train_size=n)
# 5. Overfitting: long training on a small set, with and without early stopping and weight decay
add("overfit_no_es", train_size=1000, epochs=15, evals_per_epoch=1, restore_best="false")
add("overfit_es", train_size=1000, epochs=15, evals_per_epoch=1, early_stopping_patience=2)
add("overfit_wd0_no_es", train_size=1000, epochs=15, evals_per_epoch=1, restore_best="false", weight_decay=0.0)
add("overfit_wd0.1_no_es", train_size=1000, epochs=15, evals_per_epoch=1, restore_best="false", weight_decay=0.1)

for r in runs:
    print(r)
