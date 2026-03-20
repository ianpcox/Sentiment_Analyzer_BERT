"""
Elevated Sentiment Analysis Pipeline
Project Elevate — Sentiment_Analyzer_BERT

Compares three approaches on the SST-2 benchmark:
  1. TextBlob (rule-based baseline)
  2. VADER (lexicon-based baseline)
  3. DistilBERT (distilbert-base-uncased-finetuned-sst-2-english)

Generates 8 visualizations and a metrics summary.
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, brier_score_loss,
)
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore")

OUT_DIR = Path("/home/ubuntu/sentiment_outputs")
OUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

# ── Dataset ───────────────────────────────────────────────────────────────────
def load_sst2_sample(n=500, seed=42):
    """Load a balanced sample of SST-2 validation set."""
    try:
        from datasets import load_dataset
        print("Loading SST-2 validation set from HuggingFace datasets...")
        ds = load_dataset("sst2", split="validation", trust_remote_code=True)
        df = pd.DataFrame({"text": ds["sentence"], "label": ds["label"]})
        # Balance classes
        pos = df[df["label"] == 1].sample(n=n//2, random_state=seed)
        neg = df[df["label"] == 0].sample(n=n//2, random_state=seed)
        df = pd.concat([pos, neg]).sample(frac=1, random_state=seed).reset_index(drop=True)
        print(f"  Loaded {len(df)} samples (balanced: {df['label'].value_counts().to_dict()})")
        return df
    except Exception as e:
        print(f"Could not load from HuggingFace: {e}")
        print("Using built-in sample sentences...")
        return _builtin_sample()


def _builtin_sample():
    """Fallback: hand-crafted balanced sample."""
    positives = [
        "This movie was absolutely fantastic and I loved every minute of it.",
        "The acting was superb and the story was deeply moving.",
        "A masterpiece of modern cinema, truly unforgettable.",
        "I was thoroughly entertained from start to finish.",
        "The best film I have seen in years, highly recommend.",
        "Brilliant performances and a gripping narrative.",
        "A wonderful, heartwarming experience.",
        "Exceptional storytelling with beautiful cinematography.",
        "I laughed, I cried, I was completely captivated.",
        "An outstanding achievement in filmmaking.",
    ] * 25
    negatives = [
        "This was a complete waste of time and money.",
        "The plot was incoherent and the acting was terrible.",
        "I fell asleep halfway through, utterly boring.",
        "One of the worst films I have ever seen.",
        "The dialogue was cringe-worthy and the pacing was awful.",
        "A disappointing mess with no redeeming qualities.",
        "I cannot believe how bad this turned out to be.",
        "Poorly written, poorly directed, and poorly acted.",
        "A total disaster from beginning to end.",
        "Do not waste your time on this dreadful film.",
    ] * 25
    texts  = positives + negatives
    labels = [1]*250 + [0]*250
    df = pd.DataFrame({"text": texts, "label": labels})
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


# ── Baselines ─────────────────────────────────────────────────────────────────
def textblob_predict(texts):
    from textblob import TextBlob
    preds, probs = [], []
    for t in texts:
        polarity = TextBlob(t).sentiment.polarity
        prob = (polarity + 1) / 2  # scale [-1,1] -> [0,1]
        probs.append(prob)
        preds.append(1 if polarity >= 0 else 0)
    return np.array(preds), np.array(probs)


def vader_predict(texts):
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    preds, probs = [], []
    for t in texts:
        score = sia.polarity_scores(t)["compound"]
        prob = (score + 1) / 2
        probs.append(prob)
        preds.append(1 if score >= 0 else 0)
    return np.array(preds), np.array(probs)


# ── DistilBERT ────────────────────────────────────────────────────────────────
def distilbert_predict(texts, batch_size=32):
    from transformers import pipeline
    print("Loading DistilBERT (distilbert-base-uncased-finetuned-sst-2-english)...")
    clf = pipeline(
        "text-classification",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        truncation=True,
        max_length=512,
    )
    preds, probs = [], []
    for i in range(0, len(texts), batch_size):
        batch = list(texts[i:i+batch_size])
        results = clf(batch)
        for r in results:
            label = 1 if r["label"] == "POSITIVE" else 0
            prob  = r["score"] if label == 1 else 1 - r["score"]
            preds.append(label)
            probs.append(prob)
        if (i // batch_size) % 5 == 0:
            print(f"  Processed {min(i+batch_size, len(texts))}/{len(texts)} samples...")
    return np.array(preds), np.array(probs)


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, y_prob, name):
    report = classification_report(y_true, y_pred, output_dict=True)
    return {
        "Model":     name,
        "Accuracy":  accuracy_score(y_true, y_pred),
        "F1":        report["macro avg"]["f1-score"],
        "Precision": report["macro avg"]["precision"],
        "Recall":    report["macro avg"]["recall"],
        "ROC-AUC":   roc_auc_score(y_true, y_prob),
        "Avg Prec":  average_precision_score(y_true, y_prob),
        "Brier":     brier_score_loss(y_true, y_prob),
    }


# ── Visualizations ────────────────────────────────────────────────────────────
COLORS = {"TextBlob": "#F39C12", "VADER": "#3498DB", "DistilBERT": "#E74C3C"}

def plot_metrics_comparison(metrics_df):
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    for ax, metric in zip(axes, ["Accuracy", "F1", "ROC-AUC", "Brier"]):
        vals   = metrics_df[metric].values
        models = metrics_df["Model"].values
        colors = [COLORS.get(m, "#95A5A6") for m in models]
        bars = ax.bar(models, vals, color=colors, alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    val + (0.005 if metric != "Brier" else -0.005),
                    f"{val:.3f}", ha="center", va="bottom" if metric != "Brier" else "top",
                    fontsize=9, fontweight="bold")
        ax.set_title(metric, fontsize=11, fontweight="bold")
        ax.tick_params(axis="x", rotation=10)
        if metric == "Brier":
            ax.set_ylabel("Lower is better")
        else:
            ax.set_ylim(0, 1.05)
    plt.suptitle("Model Comparison — Sentiment Classification Metrics", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "01_metrics_comparison.png")
    plt.close()
    print("Saved: 01_metrics_comparison.png")


def plot_confusion_matrices(y_true, results):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, y_pred, _) in zip(axes, results):
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Negative", "Positive"],
                    yticklabels=["Negative", "Positive"],
                    linewidths=0.5)
        acc = accuracy_score(y_true, y_pred)
        ax.set_title(f"{name}\nAccuracy: {acc:.3f}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.suptitle("Confusion Matrices", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "02_confusion_matrices.png")
    plt.close()
    print("Saved: 02_confusion_matrices.png")


def plot_roc_curves(y_true, results):
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, _, y_prob in results:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})",
                color=COLORS.get(name, "#95A5A6"), lw=2)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC=0.500)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Sentiment Models", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "03_roc_curves.png")
    plt.close()
    print("Saved: 03_roc_curves.png")


def plot_precision_recall(y_true, results):
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, _, y_prob in results:
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        ax.plot(rec, prec, label=f"{name} (AP={ap:.3f})",
                color=COLORS.get(name, "#95A5A6"), lw=2)
    baseline = y_true.mean()
    ax.axhline(baseline, color="gray", linestyle="--", lw=1, label=f"Baseline (P={baseline:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "04_precision_recall.png")
    plt.close()
    print("Saved: 04_precision_recall.png")


def plot_calibration(y_true, results):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    for name, _, y_prob in results:
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10)
        ax.plot(mean_pred, frac_pos, "o-", label=name,
                color=COLORS.get(name, "#95A5A6"), lw=2, markersize=5)
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curves\n(closer to diagonal = better calibrated)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "05_calibration.png")
    plt.close()
    print("Saved: 05_calibration.png")


def plot_confidence_distribution(results):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, _, y_prob) in zip(axes, results):
        ax.hist(y_prob, bins=30, color=COLORS.get(name, "#95A5A6"),
                alpha=0.8, edgecolor="white")
        ax.axvline(0.5, color="black", linestyle="--", lw=1)
        ax.set_xlabel("Predicted Probability (Positive)")
        ax.set_ylabel("Count")
        ax.set_title(f"{name}\nConfidence Distribution", fontsize=11, fontweight="bold")
    plt.suptitle("How Confident is Each Model?", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "06_confidence_distribution.png")
    plt.close()
    print("Saved: 06_confidence_distribution.png")


def plot_error_analysis(df, y_true, bert_preds, bert_probs):
    """Show examples where DistilBERT is most confidently wrong."""
    df = df.copy()
    df["bert_pred"]  = bert_preds
    df["bert_prob"]  = bert_probs
    df["correct"]    = (df["bert_pred"] == y_true)
    df["confidence"] = df["bert_prob"].apply(lambda p: max(p, 1-p))

    errors = df[~df["correct"]].sort_values("confidence", ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis("off")
    table_data = []
    for _, row in errors.iterrows():
        text = row["text"][:70] + "..." if len(row["text"]) > 70 else row["text"]
        true_label = "Positive" if row["label"] == 1 else "Negative"
        pred_label = "Positive" if row["bert_pred"] == 1 else "Negative"
        table_data.append([text, true_label, pred_label, f"{row['confidence']:.3f}"])

    table = ax.table(
        cellText=table_data,
        colLabels=["Text (truncated)", "True Label", "Predicted", "Confidence"],
        cellLoc="left", loc="center",
        colWidths=[0.55, 0.12, 0.12, 0.12],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.8)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2C3E50")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F8F9FA")

    ax.set_title("DistilBERT — Most Confident Errors (Top 10)",
                 fontsize=12, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "07_error_analysis.png", bbox_inches="tight")
    plt.close()
    print("Saved: 07_error_analysis.png")


def plot_score_distribution_by_label(y_true, bert_probs):
    fig, ax = plt.subplots(figsize=(9, 5))
    pos_probs = bert_probs[y_true == 1]
    neg_probs = bert_probs[y_true == 0]
    ax.hist(neg_probs, bins=30, alpha=0.7, color="#E74C3C", label="True Negative", edgecolor="white")
    ax.hist(pos_probs, bins=30, alpha=0.7, color="#2ECC71", label="True Positive", edgecolor="white")
    ax.axvline(0.5, color="black", linestyle="--", lw=1.5, label="Decision boundary (0.5)")
    ax.set_xlabel("DistilBERT Predicted Probability (Positive)")
    ax.set_ylabel("Count")
    ax.set_title("DistilBERT Score Distribution by True Label",
                 fontsize=12, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "08_score_by_label.png")
    plt.close()
    print("Saved: 08_score_by_label.png")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  SENTIMENT ANALYSIS PIPELINE — BERT ELEVATION")
    print("="*60)

    df = load_sst2_sample(n=500)
    texts   = df["text"].values
    y_true  = df["label"].values

    print("\n[1/3] Running TextBlob baseline...")
    tb_preds, tb_probs = textblob_predict(texts)

    print("[2/3] Running VADER baseline...")
    vd_preds, vd_probs = vader_predict(texts)

    print("[3/3] Running DistilBERT...")
    bert_preds, bert_probs = distilbert_predict(texts)

    # Results list for plotting
    results = [
        ("TextBlob",   tb_preds,   tb_probs),
        ("VADER",      vd_preds,   vd_probs),
        ("DistilBERT", bert_preds, bert_probs),
    ]

    # Metrics table
    metrics_rows = [compute_metrics(y_true, p, pr, n) for n, p, pr in results]
    metrics_df   = pd.DataFrame(metrics_rows)
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(metrics_df.to_string(index=False, float_format="{:.4f}".format))
    metrics_df.to_csv(OUT_DIR / "metrics_summary.csv", index=False)

    # Plots
    print("\nGenerating visualizations...")
    plot_metrics_comparison(metrics_df)
    plot_confusion_matrices(y_true, results)
    plot_roc_curves(y_true, results)
    plot_precision_recall(y_true, results)
    plot_calibration(y_true, results)
    plot_confidence_distribution(results)
    plot_error_analysis(df, y_true, bert_preds, bert_probs)
    plot_score_distribution_by_label(y_true, bert_probs)

    print(f"\nAll outputs saved to: {OUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
