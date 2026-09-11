# Online Shoppers — Imbalanced Classification

Binary classification case study using the **Online Shoppers Purchasing Intention** dataset from the UCI Machine Learning Repository.

The project focuses on modeling under class imbalance and separates three related but distinct tasks:

- ranking sessions by purchase likelihood;
- choosing an operating threshold;
- producing calibrated probabilities for downstream decisions.

## Workflow

The notebook covers:

- data-quality checks and duplicate removal before splitting;
- stratified train/test splitting;
- training-only exploratory analysis;
- preprocessing of numerical and categorical features;
- majority-class baseline;
- Logistic Regression and Random Forest comparison;
- ROC-AUC and Average Precision;
- out-of-fold probability generation;
- threshold selection with F1 and F2;
- `class_weight="balanced"`;
- `SMOTENC` inside cross-validation;
- probability calibration and Brier score;
- Random Forest hyperparameter tuning;
- final threshold selection from OOF probabilities;
- one-time evaluation on the held-out test set.

The final modeling workflow uses a tuned class-weighted Random Forest, sigmoid probability calibration, and an operating threshold selected from out-of-fold predictions by maximizing F2.

## Key modeling decisions

- Exact duplicate records are removed before the train/test split.
- The held-out test set is isolated before target-guided exploratory analysis.
- Integer-coded categorical variables are one-hot encoded rather than treated as continuous quantities.
- Average Precision and ROC-AUC are used alongside threshold-dependent metrics because of the class imbalance.
- `SMOTENC` is applied inside the imbalanced-learn pipeline so oversampling occurs only within training folds.
- Threshold selection and calibration decisions are made using training/CV predictions, not the test set.

## Reproducibility

The notebook downloads the dataset through `ucimlrepo`, so no raw dataset file is stored in this repository.

Install the dependencies:

```bash
pip install -r requirements.txt
```

Then run:

```text
online_shoppers_imbalanced_classification_case_study.ipynb
```

from top to bottom.

A fixed `RANDOM_STATE` is used for the train/test split, cross-validation, Random Forest models, Logistic Regression, and SMOTENC.

## Dataset

**Online Shoppers Purchasing Intention Dataset**  
UCI Machine Learning Repository  
Dataset ID: 468  
DOI: 10.24432/C5F88Q

The original dataset contains 12,330 sessions, 17 predictors, and the binary target `Revenue`.

## Files

```text
04_online_shoppers_imbalanced_classification/
├── README.md
├── online_shoppers_imbalanced_classification_case_study.ipynb
└── requirements.txt
```
