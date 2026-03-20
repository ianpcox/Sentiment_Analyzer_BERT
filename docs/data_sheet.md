# Data Sheet – Sentiment_Analyzer_BERT

## Data sources

- **Primary:** The notebook uses a pre-trained model `nlptown/bert-base-multilingual-uncased-sentiment` (no local dataset). For reproducible evaluation use a public sentiment dataset (e.g. IMDb, SST-2, or a review dataset with star ratings) with a stated train/val/test split.
- **License / version:** Model: Hugging Face; datasets: per dataset license (e.g. Apache 2.0, academic use).

## Splits

- **Train/val/test:** 80/10/10 or 70/15/15 with fixed seed. For inference-only (no fine-tuning), use a single test set and compare baseline vs. pre-trained BERT.

## Demographics, collection, and biases

- **Collection:** Source of text (reviews, social, support); language (model is multilingual). Document domain and language in report.
- **Known biases:** Domain shift (e.g. training on reviews, testing on tweets); label distribution and class imbalance. Note in report.
