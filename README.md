<div align="center">

# DATA · AI · ENGINEERING

### Applied Machine Learning · Deep Learning · Scientific Computing

`models → experiments → validation → analysis → reproducible systems`

<br>

A technical portfolio focused on building, understanding, and evaluating machine learning systems through hands-on projects.

</div>

---

## About

I'm an **Engineering Physicist and Data Scientist**, currently continuing my academic background in **Mathematics**.

My interests sit at the intersection of:

- machine learning;
- deep learning;
- statistical and mathematical modeling;
- scientific computing;
- software-oriented ML development.

This repository documents projects where I explore those areas through complete implementations rather than isolated examples.

The emphasis is not only on obtaining a good metric, but on understanding **why a system works, how it should be evaluated, and where it can fail**.

---

## Featured Work

### Machine Learning

| Project | Problem | Main Focus |
|---|---|---|
| [Bank Marketing](./machine-learning/01_bank_marketing/) | Binary Classification | Leakage-aware feature design, pipelines, cross-validation, model comparison |
| [Bike Sharing Demand](./machine-learning/02_bike_sharing_regression/) | Regression | Temporal validation, feature engineering, residual analysis |
| [Dry Bean](./machine-learning/03_dry_bean_multiclass/) | Multiclass Classification | SVM, PCA, hyperparameter tuning, multiclass evaluation |
| [Online Shoppers](./machine-learning/04_online_shoppers_imbalanced_classification/) | Imbalanced Classification | SMOTENC, threshold selection, calibration, Average Precision |
| [Online Retail II](./machine-learning/05_online_retail_customer_segmentation/) | Unsupervised Learning | Customer-level feature engineering, K-Means, GMM, DBSCAN, anomaly detection |

### Deep Learning

| Project | Problem | Main Focus |
|---|---|---|
| [YearPredictionMSD](./deep-learning/01_yearpredictionmsd_pytorch/) | Neural Regression | PyTorch, MLPs, training ablations, AdamW, BatchNorm, auxiliary pretraining, fine-tuning, Optuna |
| [Household Energy Forecasting](./deep-learning/02_household_energy_rnn_forecasting/) | Time-Series Forecasting | PyTorch, RNN/LSTM/GRU, chronological validation, sliding windows, multi-horizon forecasting |

---

## Engineering Approach

Most projects follow a workflow similar to:

```text
┌──────────────────────┐
│   Problem Definition │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│   Data Understanding │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│ Feature / Representation
│      Engineering     │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│       Modeling       │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│ Validation Strategy  │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│ Error & Diagnostic   │
│      Analysis        │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│ Reproducible Result  │
└──────────────────────┘
```

The exact workflow changes with the problem. A time-dependent regression problem should not be validated like IID classification, and an unsupervised problem should not be evaluated as though ground-truth labels existed.

That distinction is an important part of the work documented here.

---

## Machine Learning

The classical machine-learning projects cover several different modeling settings rather than repeatedly applying the same workflow to different datasets.

### Covered areas

**Supervised learning**

- binary classification;
- multiclass classification;
- regression;
- imbalanced learning;
- support vector machines.

**Model evaluation**

- cross-validation;
- temporal validation;
- leakage prevention;
- threshold selection;
- probability calibration;
- residual analysis;
- error analysis.

**Representation & preprocessing**

- feature engineering;
- dimensionality reduction with PCA;
- scaling and transformations;
- categorical encoding;
- resampling with SMOTENC.

**Unsupervised learning**

- K-Means;
- Gaussian Mixture Models;
- DBSCAN;
- anomaly detection;
- cluster stability and profiling.

→ [`machine-learning/`](./machine-learning/)

---

## Deep Learning

The deep-learning section focuses on implementing and understanding neural-network training with **PyTorch** rather than treating training as a black-box `.fit()` operation.

The current projects cover:

- tensors and GPU-aware data pipelines;
- custom `Dataset` and `DataLoader`;
- reusable training and evaluation loops;
- multilayer perceptrons;
- custom `nn.Module` architectures;
- initialization and activation experiments;
- Batch Normalization;
- SGD and adaptive optimization;
- AdamW and weight decay;
- dropout;
- learning-rate behavior;
- representation pretraining;
- transfer learning;
- fine-tuning;
- hyperparameter optimization with Optuna;
- checkpointing;
- recurrent neural networks (RNN, LSTM, GRU);
- chronological time-series validation;
- sliding-window sequence modeling;
- multi-horizon forecasting;
- residual and distribution-level error analysis.

→ [`deep-learning/`](./deep-learning/)

---

## Technical Stack

<div align="center">

`Python` · `NumPy` · `Pandas` · `SciPy` · `scikit-learn` · `PyTorch` · `Optuna` · `Matplotlib` · `Git`

</div>

---

## Repository Structure

```text
data-ai-engineering-portfolio/
│
├── machine-learning/
│   ├── 01_bank_marketing/
│   ├── 02_bike_sharing_regression/
│   ├── 03_dry_bean_multiclass/
│   ├── 04_online_shoppers_imbalanced_classification/
│   └── 05_online_retail_customer_segmentation/
│
├── deep-learning/
│   ├── 01_yearpredictionmsd_pytorch/
│   └── 02_household_energy_rnn_forecasting/
│
└── README.md
```

Projects are kept self-contained whenever possible:

```text
project/
├── README.md
├── notebook.ipynb
└── requirements.txt
```

Datasets and generated artifacts are generally excluded from version control when they can be downloaded or reproduced automatically.

---

## Project Philosophy

The purpose of this portfolio is not to collect as many algorithms as possible.

Projects are designed to strengthen and demonstrate the ability to reason through the complete modeling process:

- What exactly is one observation?
- What information is available at prediction time?
- Where can leakage occur?
- What validation scheme matches the intended generalization setting?
- Which metric reflects the actual objective?
- What assumptions does the model make?
- Where does the model fail?
- Are improvements large enough to justify additional complexity?

Because of that, notebooks often include alternative approaches, diagnostics, failed or weaker experiments, and explicit limitations when those results help explain the final modeling decision.

---

## Reproducibility

Each project includes its own environment requirements and dataset instructions.

A typical workflow is:

```bash
git clone <repository-url>
cd <project-directory>

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Then run the project notebook from top to bottom.

---

<div align="center">

### From mathematical foundations to working models.

`DATA // MODELS // VALIDATION // SYSTEMS`

</div>