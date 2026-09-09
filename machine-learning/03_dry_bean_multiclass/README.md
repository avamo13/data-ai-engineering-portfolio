# Dry Bean Variety Classification

Multiclass classification case study using the UCI Dry Bean dataset to distinguish seven bean varieties from geometric features extracted by a computer vision system.

## Dataset

- **Source:** UCI Machine Learning Repository — Dry Bean
- **Raw observations:** 13,611
- **Numerical features:** 16
- **Classes:** 7
- **Task:** multiclass classification
- **License:** CC BY 4.0
- **UCI DOI:** 10.24432/C50S4B

The notebook downloads the official UCI archive automatically when the local dataset is not already available.

## Data preparation

The raw dataset contains 68 exact duplicate rows. These are removed before the train/test split, leaving 13,543 observations.

The cleaned data is split using a stratified 80/20 random split:

- **Training:** 10,834 observations
- **Test:** 2,709 observations

The test set is held out before exploratory analysis and model selection.

## Modeling approach

The project evaluates:

- majority-class baseline;
- Logistic Regression;
- linear SVM;
- RBF SVM;
- Random Forest;
- PCA + RBF SVM.

Models that depend on feature geometry are evaluated inside a `StandardScaler` pipeline.

Model selection uses five-fold `StratifiedKFold` cross-validation with:

- accuracy;
- balanced accuracy;
- macro F1;
- weighted F1.

## Feature structure

The geometric feature set contains substantial redundancy.

Two variables are deterministically derived from other measurements:

\begin{equation}
\text{Roundness} = \frac{4\pi A}{P^2}
\end{equation}

and

\begin{equation}
\text{Compactness} =
\frac{\text{EquivDiameter}}
{\text{MajorAxisLength}}.
\end{equation}

PCA confirms this redundancy:

- 4 components explain at least 95% of variance;
- 7 components explain at least 99% of variance.

Using 8 principal components preserves essentially the same cross-validated RBF SVM performance as the original 16-feature representation.

## Model selection

The strongest untuned model is the RBF SVM, with macro F1 of approximately **0.9416**.

The final grid search selects:

```text
C = 300
gamma = 0.01
PCA components = 8
```

Cross-validated macro F1:

```text
0.9441 ± 0.0050
```

## Final test performance

```text
Accuracy          = 0.9236
Balanced accuracy = 0.9337
Macro F1          = 0.9343
Weighted F1       = 0.9236
```

`BOMBAY` is perfectly classified in the held-out test set. `SIRA` is the most difficult class, with F1 around 0.87.

The dominant confusion is between `SIRA` and `DERMASON`:

- 9.49% of true `SIRA` observations are classified as `DERMASON`;
- 6.21% of true `DERMASON` observations are classified as `SIRA`.

The same pair is the main confusion in cross-validation.

## Error analysis and interpretation

The error analysis compares feature medians for the full `SIRA` and `DERMASON` classes against the subsets that are confused with each other. Misclassified observations are substantially more similar across several geometric variables, supporting the observed overlap between these classes.

Permutation importance is also evaluated using macro F1. Because multiple geometric features are highly correlated or nearly redundant, individual permutation importance values are interpreted as marginal predictive dependence rather than causal importance.

## Project structure

```text
03_dry_bean_multiclass/
├── README.md
├── dry_bean_multiclass_case_study.ipynb
└── requirements.txt
```
