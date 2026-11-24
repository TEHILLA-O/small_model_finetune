# Small Model Fine-Tune

Fine-tuning, evaluation and CPU inference benchmarks for BERT-Mini (`google/bert_uncased_L-4_H-256_A-4`, 11.2M parameters) on SST-2 sentiment, with LoRA, ablations over 3 seeds, out of domain test sets and quantisation trade-offs. Every number in this repo comes from a run on a CPU only machine, and the result files are committed next to the code that produced them.

Work in progress: the ablation grid is still running. Results and the full write-up follow in the next commit.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src python -m smft.train --config configs/base.yaml --out results/main --save-model
```
