# Bike Sharing Demand — Regression with Temporal Validation

Classical machine learning case study using the UCI Bike Sharing dataset to forecast hourly bike-rental demand.

## Scope

The project covers:

- data-quality and leakage checks;
- chronological train/test separation;
- exploratory analysis on the training period;
- Fourier feature engineering for cyclic calendar variables;
- expanding-window temporal validation;
- comparison of linear and tree-based regressors;
- Ridge regularization;
- log-transformed target modeling;
- Random Forest hyperparameter tuning;
- final evaluation on a held-out future period;
- residual and segmented error analysis;
- impurity-based and permutation feature importance.

## Problem framing

The target is `cnt`, the total number of rentals during an hour.

The raw variables `casual` and `registered` are excluded because:

\begin{equation}
\texttt{cnt} = \texttt{casual} + \texttt{registered}.
\end{equation}

`instant` is excluded as an arbitrary sequential time proxy. The date is instead converted into elapsed time from a fixed reference date.

## Validation strategy

The observations are sorted chronologically.

- The final 20% is reserved as a future test period.
- The earlier observations are used for model development.
- Model selection uses expanding-window `TimeSeriesSplit`.

## Feature engineering

The preprocessing pipeline includes:

- one-hot encoding for `season` and `weekday`;
- Fourier representations of hour and month;
- optional Fourier interactions between hour and working-day status;
- elapsed years since January 1, 2011;
- passthrough of the remaining retained predictors.

## Models

The comparison includes:

- Dummy Regressor
- Linear Regression
- Ridge
- Elastic Net
- Decision Tree
- Random Forest
- Histogram Gradient Boosting

Histogram Gradient Boosting achieves the strongest untuned temporal-validation performance. Random Forest is selected for the final tuning and interpretation stages for pedagogical exploration.

## Final model analysis

The selected Random Forest is evaluated on the held-out future test period using:

- MAE
- RMSE
- $R^2$
- residual diagnostics
- segmented weather-condition errors
- impurity-based feature importance
- permutation importance

## Project structure

```text
02_bike_sharing_regression/
├── README.md
├── bike_sharing_regression_case_study.ipynb
└── requirements.txt
```
