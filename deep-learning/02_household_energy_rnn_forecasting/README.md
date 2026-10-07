# Household Energy Forecasting with Recurrent Neural Networks

PyTorch time-series forecasting case study using the **UCI Appliances Energy Prediction** dataset.

The project studies how recurrent architecture, historical context, recurrent depth, and forecast horizon affect short-term household appliance-energy prediction under strict chronological validation.

## Problem

Each observation represents a 10-minute measurement from a single household. The primary task predicts appliance energy consumption 10 minutes ahead from a sequence of recent appliance, temperature, humidity, weather, and cyclical time features.

The dataset contains **19,735 observations** spanning approximately 4.5 months.

## Modeling protocol

The workflow uses:

- chronological 70% / 15% / 15% train-validation-test partitions;
- feature scaling fitted on the training period only;
- cyclical time-of-day and day-of-week features;
- split-boundary historical context for validation and test windows;
- persistence and daily-seasonal baselines;
- vanilla RNN, LSTM, and GRU comparison;
- sequence-length experiments;
- one- versus two-layer recurrent models;
- validation-only hyperparameter search;
- final refit on train + validation;
- one-time evaluation on the locked chronological test period;
- residual diagnostics;
- direct multi-horizon forecasting up to 60 minutes ahead.

The primary model-selection metric is **validation RMSE**. MAE is reported as a complementary metric.

## Validation experiments

### Baselines

| Model | MAE (Wh) | RMSE (Wh) |
|---|---:|---:|
| Persistence — lag 1 | **26.16** | **66.43** |
| Daily seasonal — lag 144 | 53.32 | 115.38 |

The strong persistence result confirms that appliance consumption has substantial short-term continuity.

### Recurrent architecture

Using a 6-hour historical window:

| Model | Parameters | Validation MAE | Validation RMSE |
|---|---:|---:|---:|
| RNN | 6,209 | 26.58 | 58.62 |
| LSTM | 24,641 | 27.65 | 58.95 |
| **GRU** | 18,497 | **25.98** | **58.31** |

The GRU provides the best recurrent validation result and is selected for the remaining experiments.

### Historical context

| History | Timesteps | Validation MAE | Validation RMSE |
|---|---:|---:|---:|
| 1 hour | 6 | **25.97** | 58.41 |
| **6 hours** | **36** | 25.98 | **58.31** |
| 24 hours | 144 | 26.63 | 58.59 |

A full 24-hour context does not improve one-step forecasting. The 6-hour window is selected by validation RMSE.

### Recurrent depth

| Layers | Parameters | Validation MAE | Validation RMSE |
|---|---:|---:|---:|
| 1 | 18,497 | 25.98 | 58.31 |
| **2** | **43,457** | **25.25** | **58.14** |

The second recurrent layer provides a small but consistent validation improvement.

## Selected configuration

Validation-only hyperparameter search selects:

```text
Recurrent cell:  GRU
Sequence length: 36 timesteps (6 hours)
Layers:          2
Hidden size:     64
Dropout:         0.30
Optimizer:       AdamW
Learning rate:   3e-4
Weight decay:    1e-3
Selected epoch:  15
```

Selected validation performance:

- **MAE:** 25.75 Wh
- **RMSE:** 57.91 Wh

Some search trials produce slightly lower MAE, but configuration selection follows the predefined RMSE criterion.

## Locked test results

After selection, the model is reinitialized and trained on the combined train + validation period before one-time chronological test evaluation.

| Model | Test MAE | Test RMSE |
|---|---:|---:|
| Persistence | **26.74 Wh** | 66.84 Wh |
| **Final GRU** | 30.79 Wh | **59.94 Wh** |

The GRU reduces RMSE by roughly 10%, indicating better control of large errors, but persistence remains better in average absolute one-step error.

Residual analysis shows two recurrent error modes:

- frequent low-consumption observations are often mildly **overpredicted**;
- rare high-consumption spikes are often strongly **underpredicted**.

The median residual is approximately **-14.1 Wh**, while the largest positive residual exceeds **500 Wh**.

## Multi-horizon forecasting

The selected recurrent architecture is also trained to predict the next 60 minutes directly.

| Horizon | GRU MAE | GRU RMSE | Persistence MAE | Persistence RMSE |
|---|---:|---:|---:|---:|
| +10 min | 28.71 | **61.81** | **26.19** | 66.48 |
| +20 min | **34.16** | **76.55** | 37.71 | 93.89 |
| +30 min | **36.61** | **79.65** | 43.70 | 102.00 |
| +40 min | **37.54** | **80.79** | 46.89 | 105.54 |
| +50 min | **38.21** | **81.63** | 49.58 | 108.88 |
| +60 min | **38.91** | **82.04** | 51.63 | 111.50 |

Persistence is very competitive at the shortest horizon, but from +20 minutes onward the recurrent model outperforms it on both metrics. The advantage grows as the horizon increases.

## Main conclusions

- Short-term persistence is a strong benchmark for this series.
- All recurrent cells reduce validation RMSE relative to persistence.
- The GRU provides the best complexity-performance tradeoff.
- More historical context is not automatically better.
- Additional recurrent depth helps, but only modestly.
- The final GRU improves test RMSE while worsening test MAE, so it does not dominate persistence for every objective.
- Recurrent modeling becomes more valuable as the forecast horizon increases.
- Rare consumption spikes remain the main source of large errors.

## Limitations

The dataset contains only one household and approximately 4.5 months of observations. Results should therefore not be generalized to other households or seasons.

The project uses a single chronological validation interval rather than rolling-origin backtesting. Large consumption spikes are relatively rare and are difficult to predict from the available variables.

## Reproducibility

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```text
household_energy_rnn_forecasting.ipynb
```

from top to bottom.

The dataset is downloaded automatically from the UCI Machine Learning Repository and stored under `data/`.

## Project structure

```text
02_household_energy_rnn_forecasting/
├── .gitignore
├── README.md
├── requirements.txt
└── household_energy_rnn_forecasting.ipynb
```

## Dataset

**Appliances Energy Prediction**  
UCI Machine Learning Repository

https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction
