"""
Measure the rule-based detectors on a labelled CSV with columns `text,label`
(label 1 = phishing / fake news, 0 = legitimate):

    python -m cyber evaluate data/emails.csv --detector phishing --threshold 0.5
"""
import csv

from cyber.news import analyze_news
from cyber.phishing import analyze_text

DETECTORS = {"phishing": analyze_text, "news": analyze_news}


def evaluate_rows(rows, detector="phishing", threshold=0.5):
    analyze = DETECTORS[detector]
    tp = fp = tn = fn = 0
    for row in rows:
        label = int(row["label"])
        predicted = analyze(row["text"])["score"] >= threshold
        if predicted and label:
            tp += 1
        elif predicted:
            fp += 1
        elif label:
            fn += 1
        else:
            tn += 1
    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "n": total,
        "true_positives": tp, "false_positives": fp, "true_negatives": tn, "false_negatives": fn,
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        # A missed phishing email is the costly error: it reached the user.
        "false_negative_rate": fn / (tp + fn) if tp + fn else 0.0,
    }


def evaluate_csv(path, detector="phishing", threshold=0.5):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = {"text", "label"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        return evaluate_rows(reader, detector, threshold)
