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

## Reference

This project is motivated by the following paper:

> G La Malfa, JM Zhang, M Luck, E Black (2026). *Fairness aware reinforcement learning via proximal policy optimization*. Proceedings of the AAAI. https://ojs.aaai.org/index.php/AAAI/article/view/39434

Paper author: Prof. Elizabeth Black, King's College London (elizabeth.black@kcl.ac.uk)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
