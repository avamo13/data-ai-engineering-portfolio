# YearPredictionMSD — Deep Learning Regression with PyTorch

PyTorch regression case study on the **UCI YearPredictionMSD** dataset, using more than 500k songs and a locked artist-aware official test split.

The project focuses on practical deep-learning training and model selection rather than a classical-machine-learning baseline.

## Highlights

- 515,345 songs and 90 continuous timbre features
- custom PyTorch `Dataset` / `DataLoader`
- reusable training, evaluation, early-stopping, and checkpoint utilities
- MLP training ablations
- custom multi-branch architecture
- AdamW, dropout, BatchNorm, and He initialization
- auxiliary decade classification for representation pretraining
- frozen transfer learning and end-to-end fine-tuning
- Optuna hyperparameter optimization
- validation-only model selection
- one-time evaluation on UCI's official test set
- residual and decade-level error analysis

## Results

| Model / experiment | Validation RMSE |
|---|---:|
| Baseline MLP — SGD | 8.66 years |
| Multi-branch MLP | 8.64 years |
| He init + AdamW + Dropout | 8.42 years |
| BatchNorm + AdamW + Dropout | 8.37 years |
| Regression from scratch (transfer architecture) | 8.38 years |
| Frozen auxiliary-pretrained model | 8.56 years |
| Optuna-selected MLP | 8.35 years |
| **Fine-tuned auxiliary-pretrained model** | **8.34 years** |

The fine-tuned auxiliary-pretrained model was selected before opening the official test set.

### Official test

- **MAE:** 5.88 years
- **RMSE:** 8.72 years
- **N:** 51,630 songs

The model performs best in the densely represented 1990s and 2000s. Error increases substantially for older decades, and residuals show a clear tendency to pull sparse historical examples toward more densely represented recent years.

## Methodology

### Data split

The project follows UCI's required split:

- first 463,715 rows: official training partition
- last 51,630 rows: official test partition

UCI states that this split avoids the *producer effect* by ensuring that songs from the same artist do not appear in both official partitions.

The official training partition is further divided into train and validation subsets for model development. Feature and target scalers are fitted on the training subset only.

### Model development

A `[256, 128, 64]` MLP trained with SGD is used as the internal neural-network baseline. Selected experiments then test optimization/regularization choices and a custom two-branch architecture for the two timbre feature groups.

A second modeling path pretrains a feature extractor to classify release decade. The learned representation is then transferred to year regression. Freezing the pretrained extractor does not outperform training from scratch, but end-to-end fine-tuning at a lower learning rate produces the best validation result.

Optuna independently searches a from-scratch MLP over learning rate, depth, width, dropout, weight decay, and batch size. The tuned MLP is competitive but remains slightly behind the fine-tuned transfer model on validation data.

## Error analysis

The official test results are strongly dependent on decade. Approximate examples:

| Decade | N | MAE | RMSE |
|---|---:|---:|---:|
| 1980s | 4,201 | 7.23 | 9.12 |
| 1990s | 12,580 | 4.71 | 5.90 |
| 2000s | 29,885 | 4.34 | 6.30 |
| 2010s | 1,033 | 7.32 | 8.32 |

Very early decades have much larger errors but also very small sample sizes. The broader pattern is consistent with severe target imbalance and regression toward the densely represented late-1990s/2000s region.

## Limitations

The internal validation split is random within the official training partition. Artist identifiers are not included in the benchmark feature matrix, so the internal split cannot reproduce the artist-level separation used by UCI's official test partition.

This may make validation somewhat optimistic relative to the final test distribution.

## Reproducibility

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```text
yearpredictionmsd_pytorch_case_study.ipynb
```

from top to bottom.

The notebook downloads the dataset automatically and writes local artifacts under `data/` and `artifacts/`; both directories are excluded from Git.

## Dataset

**Year Prediction MSD**  
UCI Machine Learning Repository  
515,345 instances, 90 real-valued audio features  
DOI: `10.24432/C50K61`

Dataset page: https://archive.ics.uci.edu/dataset/203/yearpredictionmsd

## Project structure

```text
01_yearpredictionmsd_pytorch/
├── .gitignore
├── README.md
├── requirements.txt
└── yearpredictionmsd_pytorch_case_study.ipynb
```
