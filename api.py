"""
Sentiment Analysis API — FastAPI Microservice
Project Elevate — Sentiment_Analyzer_BERT

Endpoints:
  POST /predict          — Classify a single text
  POST /predict-batch    — Classify a list of texts
  GET  /health           — Health check
  GET  /model-info       — Model metadata

Usage:
  uvicorn api:app --reload
"""

import time
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Sentiment Analysis API",
    description=(
        "A production-ready sentiment classification API powered by "
        "DistilBERT (distilbert-base-uncased-finetuned-sst-2-english). "
        "Classifies text as Positive or Negative with calibrated confidence scores."
    ),
    version="1.0.0",
)

# ── Model loading (lazy, on first request) ────────────────────────────────────
_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline
        _pipeline = pipeline(
            "text-classification",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            truncation=True,
            max_length=512,
        )
    return _pipeline


# ── Schemas ───────────────────────────────────────────────────────────────────
class TextInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000,
                      example="The acting was superb and the story was deeply moving.")


class BatchInput(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=100,
                              example=["Great movie!", "Terrible waste of time."])


class SentimentResult(BaseModel):
    text: str
    prediction: str
    positive_probability: float
    negative_probability: float
    model: str
    processing_time_ms: float


class BatchResult(BaseModel):
    results: List[SentimentResult]
    total_texts: int
    total_processing_time_ms: float


# ── Helpers ───────────────────────────────────────────────────────────────────
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

def classify(text: str) -> dict:
    clf = get_pipeline()
    t0  = time.perf_counter()
    out = clf(text[:512])[0]
    elapsed_ms = (time.perf_counter() - t0) * 1000

    label = out["label"]
    score = out["score"]
    if label == "POSITIVE":
        pos_prob, neg_prob = score, 1 - score
        prediction = "Positive"
    else:
        neg_prob, pos_prob = score, 1 - score
        prediction = "Negative"

    return {
        "text": text,
        "prediction": prediction,
        "positive_probability": round(pos_prob, 6),
        "negative_probability": round(neg_prob, 6),
        "model": MODEL_NAME,
        "processing_time_ms": round(elapsed_ms, 2),
    }


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["Utility"])
def health_check():
    """Returns API health status."""
    return {"status": "ok", "model": MODEL_NAME}


@app.get("/model-info", tags=["Utility"])
def model_info():
    """Returns metadata about the deployed model."""
    return {
        "model": MODEL_NAME,
        "architecture": "DistilBERT (6-layer, 66M parameters)",
        "task": "Binary Sentiment Classification (Positive / Negative)",
        "dataset": "SST-2 (Stanford Sentiment Treebank)",
        "benchmark_accuracy": 0.922,
        "benchmark_roc_auc": 0.977,
        "max_input_length": 512,
        "source": "https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english",
    }


@app.post("/predict", response_model=SentimentResult, tags=["Inference"])
def predict(payload: TextInput):
    """
    Classify a single text as Positive or Negative.

    Returns the prediction label and calibrated probability scores.
    """
    try:
        return classify(payload.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/predict-batch", response_model=BatchResult, tags=["Inference"])
def predict_batch(payload: BatchInput):
    """
    Classify a batch of up to 100 texts.

    Returns individual results for each text plus total processing time.
    """
    t0 = time.perf_counter()
    try:
        results = [classify(t) for t in payload.texts]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
    total_ms = (time.perf_counter() - t0) * 1000
    return {
        "results": results,
        "total_texts": len(results),
        "total_processing_time_ms": round(total_ms, 2),
    }
