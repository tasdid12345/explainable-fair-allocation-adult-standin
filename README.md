# Group-specific thresholds for allocation fairness on Adult

## Motivation
Allocation policies learned by reinforcement learning can reproduce or amplify group disparities unless fairness is built into the objective. The paper "Fairness aware reinforcement learning via proximal policy optimization" (2026) proposes a PPO-trained fairness-aware policy for this problem. We do not re-implement that method; instead we test a simpler, cheaper fairness intervention on a public stand-in task.

## Question
Can a class-balanced LightGBM with validation-tuned, group-specific decision thresholds give a smaller sex-based demographic parity gap in allocation decisions than a standard unconstrained logistic regression on the public Adult dataset, at a small cost in accuracy?

## Dataset
The public OpenML Adult dataset (data_id 1590), downloaded with the `openml` Python package. Each person is treated as an agent, the binary income label (>50K) is the reward-worthy outcome, and sex is the sensitive attribute. This is a stand-in task and is not the paper's own dataset.

## Baseline vs improved model
- **Baseline:** unconstrained standard logistic regression (one-hot categoricals, standardised numerics). Predicted positives are the allocations, using a single 0.5 threshold. It stands in for the paper's PPO policy, which is not re-implemented.
- **Improved:** class-balanced LightGBM. Separate thresholds for each sex are chosen on the validation split to equalise allocation rates.
- Both models use the same split, data, and evaluation function. The main metric is the test-set demographic parity difference.

## How to run
```
pip install -r requirements.txt
python analysis.py
```

## Results
Results are written to `results.json`. Charts are saved to `figures/`.

## Results (latest run)

On the public Adult dataset test set, demographic parity difference fell from 17.9% for a standard unconstrained logistic regression to 2.0% for my class-balanced LightGBM with per-sex thresholds, at 85.4% vs 85.0% accuracy, though the equal-opportunity gap rose from 12.7% to 32.2%.

| Metric | Value |
|---|---|
| baseline_score | 0.1788 |
| improved_score | 0.02033 |
| improvement | 0.1585 |
| baseline_score_std | 0 |
| improved_score_std | 0 |
| baseline_accuracy | 0.8541 |
| improved_accuracy | 0.85 |
| baseline_accuracy_std | 0 |
| improved_accuracy_std | 1.11e-16 |
| baseline_eo_gap | 0.127 |
| improved_eo_gap | 0.3224 |
| baseline_f1 | 0.6612 |
| improved_f1 | 0.6455 |

Run on 2026-10-20. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Single stratified split (seed 42); seeds 42-44 vary only model randomness. No hyperparameter search was run; LightGBM settings are fixed. Thresholds are selected on the validation split, never on test. Both models are trained on the fit split only. Sex is included as a feature in both models. Standard deviations are population std (ddof=0) over 3 seeds; logistic regression is deterministic so its std is 0. Demographic parity ignores base-rate differences and the equal-opportunity gap is reported as a secondary fairness check. Results are from the public Adult data, not the paper's reinforcement-learning setting.

## Reference

This project is motivated by the following paper:

> G La Malfa, JM Zhang, M Luck, E Black (2026). *Fairness aware reinforcement learning via proximal policy optimization*. Proceedings of the AAAI. https://ojs.aaai.org/index.php/AAAI/article/view/39434

Paper author: Prof. Elizabeth Black, King's College London (elizabeth.black@kcl.ac.uk)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
