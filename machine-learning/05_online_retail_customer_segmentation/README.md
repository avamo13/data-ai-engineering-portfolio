# Online Retail II Customer Segmentation

Customer-segmentation case study built from the **Online Retail II** transactional dataset from the UCI Machine Learning Repository.

The raw data contains more than one million invoice-line records spanning two years. The project converts those transactions into a customer-level analytical dataset and compares several unsupervised-learning approaches for behavioral segmentation.

## Project scope

The workflow covers:

- transaction-level data-quality auditing;
- cancellation and invalid-purchase handling;
- analytical-grain definition;
- RFM-style customer feature engineering;
- feature-space auditing and redundancy reduction;
- `log1p` transformation and standardization;
- PCA visualization;
- K-Means model selection and cluster profiling;
- K-Means stability analysis with Adjusted Rand Index;
- DBSCAN density-based clustering;
- Gaussian Mixture Models with AIC/BIC and soft assignments;
- post-hoc geographic profiling;
- Isolation Forest anomaly detection.

## Main modeling decisions

The source data has invoice-line grain, while the segmentation target is the customer. Customer-level behavioral features are therefore constructed before clustering.

Exact duplicate-looking invoice lines are retained rather than automatically removed because the source does not provide a unique invoice-line identifier. Removing them would require assuming that every identical line is an ingestion error rather than a legitimate repeated line.

The primary clustering representation uses:

- Recency
- Frequency
- Monetary Value
- Average Order Value
- Unique Products
- Customer Tenure Days

The features are transformed with `log1p` and standardized before clustering.

## Results

A three-cluster K-Means solution provides the clearest operational customer segmentation among the methods explored. The resulting groups distinguish broadly between:

- established but currently inactive customers;
- frequent, high-value customers;
- recent customers with limited purchase history.

K-Means is fitted with multiple initializations to improve stability across random seeds.

DBSCAN primarily identifies one large connected density region plus a small number of micro-clusters and noise observations, making it less useful as the primary segmentation approach for this dataset.

A six-component Gaussian Mixture Model is retained as a complementary probabilistic representation. It provides more granular activity profiles and soft membership probabilities for customers near segment boundaries.

Isolation Forest is used to inspect multivariate behavioral extremes. These observations are treated as potentially legitimate customer profiles rather than automatically removed.

## Data

The notebook downloads **Online Retail II** directly from the UCI Machine Learning Repository and does not store the raw workbook in the repository.

Dataset:

- Online Retail II
- UCI Machine Learning Repository
- DOI: `10.24432/C5CG6D`

## Reproducibility

Install the dependencies:

```bash
pip install -r requirements.txt
```

Then run:

```text
online_retail_customer_segmentation_case_study.ipynb
```

from top to bottom.

The first run downloads and extracts the UCI workbook into a local `data/` directory.

## Repository structure

```text
05_online_retail_customer_segmentation/
├── README.md
├── online_retail_customer_segmentation_case_study.ipynb
└── requirements.txt
```
