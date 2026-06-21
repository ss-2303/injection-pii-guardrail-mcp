"""
Evaluation runner for the injection detector.

Loads the labeled eval set, runs each sample through scan_text(),
compares the detector's decision to the ground truth label, and
computes standard classification metrics.

Key concepts:ß
- We treat risk_level != "low" as a POSITIVE detection (flagged as injection)
- We treat risk_level == "low" as a NEGATIVE detection (cleared as benign)
- Then compare against the ground truth labels to get TP/FP/TN/FN

Metrics:
- Precision = TP / (TP + FP) — "of everything flagged, how much was real?"
- Recall    = TP / (TP + FN) — "of all real attacks, how many did we catch?"
- F1        = 2 * (P * R) / (P + R) — harmonic mean, balances both
- FPR       = FP / (FP + TN) — "how often do we cry wolf on clean text?"
"""

import json
import sys
import os

# Add the project root to the path so we can import guardrail
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from guardrail.injection_detector import scan_text


def load_eval_set(path: str) -> list[dict]:
    """Load the labeled evaluation set from JSON."""
    with open(path, "r") as f:
        data = json.load(f)
    return data["samples"]


def run_eval(samples: list[dict]) -> dict:
    """
    Run every sample through the detector and classify results.

    A detection is "positive" if risk_level is medium or high (flagged).
    A detection is "negative" if risk_level is low (cleared).

    Returns a dict with counts, metrics, and lists of misclassifications.
    """
    tp = 0  # True positive:  malicious text correctly flagged
    fp = 0  # False positive: benign text incorrectly flagged
    tn = 0  # True negative:  benign text correctly cleared
    fn = 0  # False negative: malicious text incorrectly cleared

    false_positives = []  # Benign samples that got flagged
    false_negatives = []  # Malicious samples that were missed

    details = []  # Full results for every sample

    for sample in samples:
        text = sample["text"]
        label = sample["label"]           # "malicious" or "benign"
        sample_id = sample["id"]

        result = scan_text(text)

        # Our decision: flagged (positive) or cleared (negative)
        flagged = result.risk_level != "low"

        # Ground truth: is it actually malicious?
        is_malicious = label == "malicious"

        # Classify into the confusion matrix
        if is_malicious and flagged:
            tp += 1
            outcome = "TP"
        elif is_malicious and not flagged:
            fn += 1
            outcome = "FN"
            false_negatives.append({
                "id": sample_id,
                "text": text,
                "category": sample.get("category"),
                "notes": sample.get("notes"),
                "detector_score": result.score,
                "detector_risk": result.risk_level,
            })
        elif not is_malicious and flagged:
            fp += 1
            outcome = "FP"
            false_positives.append({
                "id": sample_id,
                "text": text,
                "notes": sample.get("notes"),
                "detector_score": result.score,
                "detector_risk": result.risk_level,
                "matched_reasons": [m.reason for m in result.matches],
            })
        else:  # not is_malicious and not flagged
            tn += 1
            outcome = "TN"

        details.append({
            "id": sample_id,
            "label": label,
            "outcome": outcome,
            "score": result.score,
            "risk_level": result.risk_level,
        })

    # Compute metrics (guard against division by zero)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "counts": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "metrics": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_positive_rate": fpr,
        },
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "details": details,
        "total_samples": len(samples),
        "total_malicious": sum(1 for s in samples if s["label"] == "malicious"),
        "total_benign": sum(1 for s in samples if s["label"] == "benign"),
    }


def print_report(results: dict) -> None:
    """Print a human-readable evaluation report."""
    counts = results["counts"]
    metrics = results["metrics"]

    print("=" * 65)
    print("  INJECTION DETECTOR — EVALUATION REPORT")
    print("=" * 65)

    print(f"\n  Dataset: {results['total_samples']} samples "
          f"({results['total_malicious']} malicious, {results['total_benign']} benign)")

    print(f"\n  Confusion Matrix:")
    print(f"  {'':>20} {'Flagged':>10} {'Cleared':>10}")
    print(f"  {'Actually malicious':>20} {counts['tp']:>10} {counts['fn']:>10}")
    print(f"  {'Actually benign':>20} {counts['fp']:>10} {counts['tn']:>10}")

    print(f"\n  Metrics:")
    print(f"    Precision ............. {metrics['precision']:.2%}")
    print(f"    Recall ................ {metrics['recall']:.2%}")
    print(f"    F1 Score .............. {metrics['f1']:.2%}")
    print(f"    False Positive Rate ... {metrics['false_positive_rate']:.2%}")

    # Print false negatives (missed attacks) — the most important failures
    if results["false_negatives"]:
        print(f"\n  MISSED ATTACKS (False Negatives): {len(results['false_negatives'])}")
        print("  " + "-" * 63)
        for fn in results["false_negatives"]:
            print(f"  ID {fn['id']} [{fn['category']}] (score={fn['detector_score']}):")
            print(f"    \"{fn['text'][:80]}\"")
            print(f"    Note: {fn['notes']}")
    else:
        print(f"\n  MISSED ATTACKS: None — all malicious samples detected!")

    # Print false positives (benign text incorrectly flagged)
    if results["false_positives"]:
        print(f"\n  FALSE ALARMS (False Positives): {len(results['false_positives'])}")
        print("  " + "-" * 63)
        for fp_item in results["false_positives"]:
            print(f"  ID {fp_item['id']} (score={fp_item['detector_score']}, risk={fp_item['detector_risk']}):")
            print(f"    \"{fp_item['text'][:80]}\"")
            print(f"    Triggered: {fp_item['matched_reasons']}")
            print(f"    Note: {fp_item['notes']}")
    else:
        print(f"\n  FALSE ALARMS: None — no benign samples incorrectly flagged!")

    # Per-sample detail
    print(f"\n  Per-Sample Results:")
    print("  " + "-" * 63)
    for d in results["details"]:
        marker = "✓" if d["outcome"] in ("TP", "TN") else "✗"
        print(f"  {marker} ID {d['id']:>2} | {d['label']:>9} | "
              f"{d['outcome']} | score={d['score']:>3} | risk={d['risk_level']}")

    print("\n" + "=" * 65)


if __name__ == "__main__":
    eval_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_set.json")
    samples = load_eval_set(eval_path)
    results = run_eval(samples)
    print_report(results)