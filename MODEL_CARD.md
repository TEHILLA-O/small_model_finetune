# Model card: BERT-Mini fine-tuned on SST-2

| | |
| --- | --- |
| Base model | `google/bert_uncased_L-4_H-256_A-4` (BERT-Mini, 11.2M parameters, Turc et al. 2019) |
| Fine-tuned variants | Base config: 8,000 SST-2 sentences, 3 epochs, lr 5e-5 (`results/main`). Full data: 66,349 sentences, 4 epochs, lr 1e-4 (`results/reproduce_turc2019`) |
| Task | Binary sentiment of an English sentence or short review |
| Output | Probability of positive sentiment |
| Weights | Not committed. Rebuild with `python -m smft.train --config configs/base.yaml --out results/main --save-model` |

## Evaluation (accuracy, 24 November 2026, CPU)

| Test set | Base config | Full data |
| --- | --- | --- |
| SST-2 validation (872) | 0.784 | 0.862 |
| Rotten Tomatoes test (1,066) | 0.764 | 0.875 |
| Rotten Tomatoes, rows not in SST-2 train (609) | 0.729 | 0.834 |
| IMDB test sample (2,000) | 0.744 | 0.760 |

Calibration, error slices, significance tests and quantised variants are in the [README](README.md).

## Intended use and limits

- A portfolio and learning project for fine-tuning and evaluation on CPU, not a production sentiment service.
- Inputs longer than 128 tokens are truncated, which hurts long reviews (IMDB accuracy drops from 0.811 on the shortest quarter to 0.681 on the longest for the base model).
- Accuracy is lower on sentences with negation words (0.665 against 0.816 on SST-2 validation for the base model).
- Trained on English movie review sentences only. Expect lower accuracy on other domains.
- No fairness analysis: the data has no information about the people who wrote the reviews.
