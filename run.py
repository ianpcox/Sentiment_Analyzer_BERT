"""
Entry point for Sentiment_Analyzer_BERT: baseline (rule) and main (BERT) on minimal example.
Requires: pip install transformers torch
"""
import sys

def baseline_sentiment(text: str) -> int:
    """Simple rule: positive words -> 4, negative -> 2, else 3 (1-5 scale)."""
    pos = {"good", "great", "best", "love", "excellent"}
    neg = {"bad", "worst", "hate", "terrible", "poor"}
    t = set(text.lower().split())
    if t & pos and not (t & neg): return 4
    if t & neg and not (t & pos): return 2
    return 3

def main():
    print("=== Sentiment_Analyzer_BERT ===\n")
    test = "It was good but couldve been better. Great"
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
        model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
        tokens = tokenizer.encode(test, return_tensors="pt")
        with torch.no_grad():
            out = model(tokens)
        pred = int(torch.argmax(out.logits, dim=1)) + 1  # 1-5
        print("Main (BERT) prediction (1-5):", pred)
    except Exception as e:
        print("Main (BERT) skipped:", e, file=sys.stderr)
        pred = None
    bl = baseline_sentiment(test)
    print("Baseline (rule) prediction (1-5):", bl)
    print("Full pipeline: run notebook 'Sentiment Analyzer using BERT NN.ipynb'.")

if __name__ == "__main__":
    main()
