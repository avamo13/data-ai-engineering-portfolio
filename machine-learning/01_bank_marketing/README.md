# Bank Marketing — Term Deposit Subscription Classification

End-to-end binary classification case study using the UCI Bank Marketing dataset.

The objective is to score clients **before the current campaign call** and estimate whether they will subscribe to a term deposit. This framing makes leakage analysis central to the project because several raw variables describe information generated during the current contact.

## Dataset

- **Source:** UCI Bank Marketing dataset
- **Raw observations:** 45,211
- **Target:** term-deposit subscription (`yes` / `no`)
- **Positive-class prevalence:** approximately 11%
- **Local data:** downloaded automatically by the notebook when needed

The dataset is loaded from the official UCI archive, so the raw CSV does not need to be committed to the repository.

## Problem framing and leakage

The model is intended to run before the current campaign call. Based on that definition, the following current-contact variables were excluded:

- `contact`
- `day`
- `month`
- `duration`
- `campaign`

Using these fields would allow the model to learn from information that is unavailable at scoring time.

The dataset does not provide a client identifier, so entity leakage across splits cannot be verified directly.

## Data quality

The raw dataset contains no native `NaN` values or exact duplicate rows, but unavailable information is represented through domain-specific values such as `"unknown"` and `pdays = -1`.

The main cleaning decisions were:

- retain `"unknown"` as a category for `job` and `education`;
- represent `pdays = -1` through a separate `contacted_before` indicator;
- keep valid elapsed days in `days_since_last_contact`;
- remove five records with an inconsistent previous-contact state;
- remove one isolated observation with `previous = 275`.

The resulting modeling table contains 45,205 observations.

## Exploratory findings

The analysis identified several characteristics relevant to preprocessing:

- more than 80% of `previous` values are zero, with a long right tail among nonzero values;
- `balance` includes negative values and a strong positive tail;
- `age` is concentrated mainly between 20 and 60;
- `days_since_last_contact` is multimodal and its observed conversion rate is not monotonic across elapsed-time bins;
- conversion rates vary across retained categorical variables;
- `housing` and `loan` showed a potentially useful interaction pattern.

These observations motivated nonlinear and distribution-aware transformations rather than applying a single preprocessing strategy to every numerical variable.

## Preprocessing

The scikit-learn preprocessing pipeline includes:

- one-hot encoding for categorical variables;
- `log1p` transformation plus standardization for `previous`;
- standardization for `age`;
- Yeo–Johnson transformation for `balance`;
- a custom K-Means + RBF proximity transformer for `days_since_last_contact`;
- passthrough of the remaining retained features.

All fitted transformations are contained inside model pipelines to avoid train/validation contamination during cross-validation.

## Evaluation

The positive class represents only about 11% of the data, so accuracy is not treated as the primary metric.

The evaluation includes:

- Precision
- Recall
- Specificity
- F1
- F2
- ROC AUC
- Average Precision

A uniform dummy classifier produced an Average Precision of approximately **0.117** and ROC AUC of **0.50**.

### Five-fold stratified cross-validation

| Model | Average Precision | ROC AUC | F1 | Recall |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.3508 ± 0.0119 | 0.7187 ± 0.0096 | 0.2676 ± 0.0144 | 0.1671 ± 0.0103 |
| Decision Tree | 0.1537 ± 0.0032 | 0.5813 ± 0.0046 | 0.2595 ± 0.0080 | 0.2683 ± 0.0090 |
| Random Forest | **0.3933 ± 0.0139** | **0.7382 ± 0.0075** | 0.2714 ± 0.0159 | 0.1695 ± 0.0114 |
| Gradient Boosting | 0.3375 ± 0.0188 | 0.6983 ± 0.0112 | **0.2913 ± 0.0122** | 0.1988 ± 0.0101 |
| SGD (hinge) | 0.2766 ± 0.0424 | 0.6423 ± 0.0372 | 0.2138 ± 0.1134 | 0.2882 ± 0.2479 |
| RBF SVC | 0.3081 ± 0.0128 | 0.6346 ± 0.0104 | 0.2746 ± 0.0067 | 0.1723 ± 0.0059 |

Random Forest achieved the strongest mean Average Precision and ROC AUC among the evaluated models.

## Hyperparameter search

### Random Forest

The grid search selected:

```text
n_estimators = 500
max_depth = 10
min_samples_leaf = 5
max_features = "sqrt"
max_samples = 1.0
```

Best cross-validated Average Precision:

```text
0.3951
```

### Logistic Regression

A randomized search over `C` and `l1_ratio` selected approximately:

```text
C = 0.9573
l1_ratio = 0.2643
```

Best cross-validated Average Precision:

```text
0.3510
```

A separate regularization experiment using F1 for refitting selected L1 regularization with `C = 10` from the tested grid.

## Learning curve

A learning curve was generated for the tuned Random Forest using Average Precision and five-fold stratified cross-validation to compare training and validation behavior as the available training sample increased.

## Project structure

```text
01_bank_marketing/
├── README.md
├── bank_marketing_case_study.ipynb
└── requirements.txt
```

## Scope

This version documents the work completed through:

- data quality and cleaning;
- leakage analysis;
- exploratory analysis;
- preprocessing;
- baseline evaluation;
- comparison of six classifier families;
- regularization experiments;
- stratified cross-validation;
- hyperparameter search;
- Random Forest learning-curve analysis.

Business-specific threshold optimization, locked final-test evaluation, detailed error segmentation, and unsupervised extensions are not presented in this version because they were not completed in the source analysis.
