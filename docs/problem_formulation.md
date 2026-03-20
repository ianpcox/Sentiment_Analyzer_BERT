# Problem Formulation – Sentiment_Analyzer_BERT

## Research / business question

Can we predict sentiment (e.g. star rating or positive/negative/neutral) from text using a BERT-based model, with performance exceeding a simple baseline (lexicon or logistic regression)?

## Primary success metric and threshold

- **F1 (macro)** for sentiment classes: target ≥ 0.75.
- **Accuracy:** Target ≥ 0.78.
- **ROC-AUC** (for binary or per-class): report where applicable.
- **Calibration:** Optional; report if probability thresholds are used.
- **Baseline:** Lexicon-based (e.g. VADER) or logistic regression on TF-IDF; BERT model should outperform.

## Stakeholders and decisions

- **Social listening / support:** Decisions about prioritization and reporting of sentiment trends.
- **Product:** Decisions about which feedback to act on.
