# Evaluating Advanced Machine Learning Models for Financial Risk Prediction using FinBench

Comparing traditional machine learning models against a FinPT-like approach — converting
tabular data into natural-language customer profiles and embedding them with a sentence
transformer — on the [FinBench](https://huggingface.co/datasets/yuweiyin/FinBench) benchmark
for financial risk prediction.

Originally completed as a course project for Data Mining and Machine Learning (ELTE), and
cleaned up here into a reproducible repository.

## Objective

To compare the performance of traditional machine learning models (Logistic Regression,
Random Forest, Gradient Boosting, Neural Network) with a FinPT-like approach on FinBench
for financial risk prediction, following the methodology described in:

> Yin, Y. et al. (2023). *FinPT: Financial Risk Prediction with Profile Tuning on Pretrained
> Foundation Models.* [arXiv:2308.00065](https://arxiv.org/abs/2308.00065)

## Dataset

[FinBench](https://huggingface.co/datasets/yuweiyin/FinBench) is a benchmark for financial
risk prediction covering three risk types — default, fraud, and churn — across tabular and
natural-language profile inputs. This project focuses on two tasks:

| Task | Description | Dataset code | #Classes | #Features | #Train | #Val | #Test |
|---|---|---|---|---|---|---|---|
| Credit Default | Predict whether a user will default on the credit card | `cd2` | 2 | 23 | 18900 | 2100 | 9000 |
| Loan Default | Predict whether a user will default on the loan | `ld2` | 2 | 11 | 18041 | 2005 | 8592 |

`cd2` and `ld2` were chosen as the next-largest tasks after `ld3` in each category, large
enough to meaningfully separate baseline and FinPT-like model performance.

## Methodology

**1. Preprocessing.** Numerical columns are standardized (`StandardScaler`); categorical
columns are one-hot encoded — both fit only on the training split inside an sklearn
`Pipeline` to avoid leakage.

**2. Baselines.** Logistic Regression, Random Forest, Gradient Boosting, and a Neural
Network (MLP) are trained on the preprocessed tabular features and evaluated on accuracy,
precision, recall, F1-score, and AUC-ROC.

**3. FinPT-like model.** Each row is converted into a natural-language profile sentence
(e.g. `"age is 35, income is 45000, home_ownership is RENT, ..."`), embedded with
[`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
and classified with logistic regression on top of the resulting embeddings — a lightweight
stand-in for FinPT's full profile-tuning of a pretrained foundation model.

> **Note on a fix made here:** in the original course version, rows were converted to text
> *after* `StandardScaler` had already transformed the numeric columns, producing profiles
> like `"income is -0.27"` instead of `"income is 45000"`. That undermines the point of a
> natural-language profile — a pretrained sentence encoder has no useful prior over
> arbitrary z-scores. This repo's pipeline (`src/finpt_model.py`) always builds text
> profiles from raw, human-readable values.
>
> Re-running both tasks with this fix, the FinPT-like model's scores barely moved (e.g.
> `cd2` AUC-ROC: 0.738 → 0.739). That's itself a useful finding: `all-MiniLM-L6-v2` is a
> general-purpose encoder with no real numeric-reasoning prior either way, so raw vs.
> scaled values in the text didn't matter much to it. The bigger lever — consistent with
> what the FinPT paper argues — is profile-tuning the encoder on the actual task, not just
> cleaning up the text fed into a frozen one.

## Results

Results below were produced by re-running this repository's corrected pipeline
end-to-end (`python run_experiment.py --dataset cd2 ld2`). The `cd2` numbers match the
original course report. The `ld2` numbers are corrected — the original report's "ld2"
table was an accidental copy-paste of the `cd2` table (the two were byte-identical despite
the report's own bar chart for `ld2` clearly showing different, higher scores).

### `cd2` — Credit Default

| Model | Accuracy | Precision | Recall | F1-score | AUC-ROC |
|---|---|---|---|---|---|
| Logistic Regression | 0.821 | 0.806 | 0.821 | 0.799 | 0.772 |
| Random Forest | 0.813 | 0.796 | 0.813 | 0.795 | 0.776 |
| Gradient Boosting | 0.823 | 0.809 | 0.823 | 0.802 | 0.798 |
| Neural Network | 0.798 | 0.775 | 0.798 | 0.776 | 0.727 |
| FinPT-like Model | 0.787 | 0.776 | 0.787 | 0.717 | 0.739 |

### `ld2` — Loan Default (corrected)

| Model | Accuracy | Precision | Recall | F1-score | AUC-ROC |
|---|---|---|---|---|---|
| Logistic Regression | 0.864 | 0.856 | 0.864 | 0.854 | 0.863 |
| Random Forest | 0.930 | 0.932 | 0.930 | 0.925 | 0.918 |
| Gradient Boosting | 0.924 | 0.927 | 0.924 | 0.919 | 0.919 |
| Neural Network | 0.909 | 0.906 | 0.909 | 0.905 | 0.899 |
| FinPT-like Model | 0.788 | 0.789 | 0.788 | 0.705 | 0.749 |

On `cd2`, baseline models all score above 0.79, reflecting clear, low-noise features and a
fairly balanced class split. On `ld2`, Random Forest and Gradient Boosting pull further
ahead of Logistic Regression and the FinPT-like model, while Logistic Regression still
tracks the tree-based models reasonably closely — suggesting `ld2`'s features have a
fairly linear relationship with the target. On both tasks, the FinPT-like model trails the
tabular baselines by a wide margin, especially on F1-score, where its weighted score is
dragged down by poor recall on the minority (risky/default) class.

Notably, the FinPT-like model's scores are nearly identical across `cd2` and `ld2`
(0.787/0.739 vs. 0.788/0.749 accuracy/AUC-ROC) despite the baselines differing
substantially between the two tasks. That's consistent with the embeddings carrying a
fairly generic, task-agnostic signal rather than something tuned to either dataset's
specific structure — see the Discussion section below.

## Repository Structure

```
finbench-risk-prediction/
├── notebooks/
│   └── finbench_model_comparison.ipynb   # full end-to-end notebook (Colab-friendly)
├── src/
│   ├── data_utils.py        # FinBench loading + preprocessing
│   ├── baseline_models.py   # baseline model training/evaluation
│   ├── finpt_model.py       # text-profile + sentence-embedding pipeline
│   └── run_experiment.py    # CLI script: runs both tasks, saves results/charts
├── results/                 # generated metrics tables (.csv) and charts (.png)
├── report/                  # original course report (PDF)
├── requirements.txt
└── README.md
```

## Reproducing

```bash
git clone https://github.com/Bek-alt/finbench-risk-prediction.git
cd finbench-risk-prediction
pip install -r requirements.txt
```

Either run the notebook (`notebooks/finbench_model_comparison.ipynb`, works well in Google
Colab) or use the CLI script to run both tasks back-to-back and save results automatically:

```bash
cd src
python run_experiment.py --dataset cd2 ld2
```

This writes `results/<dataset>/model_performance.csv` and `.png` for each task.

> **Note:** Fetching FinBench requires access to the HuggingFace Hub, and the
> sentence-transformer model is also pulled from HuggingFace on first run — make sure
> you're running somewhere with normal internet access (a local machine or Colab).

## Discussion

**Interpretability.** Logistic Regression and tree-based models expose coefficients/feature
importances directly; the FinPT-like model's sentence embeddings are opaque without
additional explainability tooling (e.g. SHAP on top of the embedding space, or attention
inspection on the encoder itself).

**Training time.** Baselines train in seconds on these dataset sizes. The FinPT-like
pipeline is bottlenecked by encoding every row through a transformer, even with a small
encoder like MiniLM.

**Generalization.** `all-MiniLM-L6-v2` is a general-purpose sentence encoder, not fine-tuned
on financial profile text or this classification objective — here it's only used to produce
frozen embeddings for a linear probe, which is a much weaker setup than FinPT's actual
profile-tuning of a foundation model on labeled risk data. The fact that the FinPT-like
model scored almost identically on `cd2` and `ld2` despite those tasks having very
different baseline performance profiles supports this: the embeddings appear to encode a
fairly generic, task-agnostic signal rather than picking up on either dataset's specific
structure. That gap is the most likely explanation for the FinPT-like model
underperforming the tabular baselines in this experiment, and is broadly consistent with
limitations the original FinPT paper itself discusses for low-data tabular settings.

## Conclusion & Future Work

Traditional tabular models remain strong, fast, interpretable baselines for structured
financial risk data, and outperformed the FinPT-like approach implemented here on both
`cd2` and `ld2`. For the FinPT-like approach to close the gap, promising directions include:

- **Profile-tuning** the encoder (or a small LLM) directly on the classification objective,
  rather than freezing it and training only a linear probe on top.
- **Hybrid models** that combine structured tabular features with text embeddings.
- **Domain-adapted pretraining** of the encoder on financial text before applying it to
  FinBench profiles.

## References

- Yin, Y. et al. (2023). *FinPT: Financial Risk Prediction with Profile Tuning on Pretrained
  Foundation Models.* [arXiv:2308.00065](https://arxiv.org/abs/2308.00065)
- [FinPT GitHub repository](https://github.com/YuweiYin/FinPT)
- [FinBench dataset on HuggingFace](https://huggingface.co/datasets/yuweiyin/FinBench)
- [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [scikit-learn](https://scikit-learn.org/stable/)
