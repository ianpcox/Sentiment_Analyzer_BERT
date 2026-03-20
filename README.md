# Sentiment Analyzer API (Project Elevate)

This repository transforms a basic BERT tutorial into a **production-ready Sentiment Analysis API** using modern HuggingFace Transformers.

We evaluate three distinct approaches to binary sentiment classification (Positive vs. Negative) using the SST-2 (Stanford Sentiment Treebank) dataset. The results demonstrate the overwhelming superiority of contextual language models over traditional lexicon-based approaches.

## Project Structure

* `sentiment_pipeline.py` — The core reproducible evaluation pipeline. Runs TextBlob, VADER, and DistilBERT against 500 SST-2 reviews, computing metrics and generating 8 charts.
* `api.py` — A production-ready **FastAPI microservice** that exposes the DistilBERT model for real-time inference.
* `docs/report.md` — A comprehensive paper-style report detailing the methodology, model comparison, calibration, and error analysis.
* `docs/assets/` — Generated static charts supporting the report.
* `Sentiment Analyzer using BERT NN.ipynb` — The original legacy notebook.

## Key Findings

| Model | Accuracy | F1-Score | ROC-AUC |
|---|---|---|---|
| TextBlob (Baseline) | 62.8% | 0.603 | 0.750 |
| VADER (Baseline) | 64.6% | 0.624 | 0.747 |
| **DistilBERT** | **92.2%** | **0.922** | **0.977** |

DistilBERT not only outperforms the baselines by ~30 percentage points in accuracy, but it is also exceptionally well-calibrated, meaning its probability scores accurately reflect its true confidence.

Read the full analysis in [docs/report.md](docs/report.md).

## How to Run the API

Install the dependencies:
```bash
pip install -r requirements.txt
```

Start the FastAPI server:
```bash
uvicorn api:app --reload
```

Test the API in another terminal:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "The acting was superb and the story was deeply moving."}'
```
Response:
```json
{
  "prediction": "Positive",
  "positive_probability": 0.9998,
  "negative_probability": 0.0002,
  "model": "distilbert-base-uncased-finetuned-sst-2-english",
  "processing_time_ms": 42.1
}
```
