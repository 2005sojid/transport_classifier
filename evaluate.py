"""
Honest Leave-One-Out evaluation.

"Honest" means: when evaluating image X, ALL augmented versions of X
are removed from the gallery. This prevents data leakage from near-duplicate
augmentations inflating accuracy.

Outputs:
  - Per-class precision, recall, F1
  - Macro-averaged metrics
  - Weighted-average metrics (accounts for class imbalance)
  - Confusion matrix (saved as PNG)
  - JSON report

Usage:
  python evaluate.py [--gallery gallery/gallery.npz] [--k 11] [--out eval_results/]
"""
from __future__ import annotations

import argparse
import json
import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
import os
import time

import numpy as np
from PIL import Image
from tqdm import tqdm

import config
import backbone


def load_gallery_data(gallery_path: str):
    data = np.load(gallery_path, allow_pickle=True)
    return {
        "embeddings": data["embeddings"],
        "labels": data["labels"],
        "source_ids": data["source_ids"],
        "class_names": list(data["class_names"]),
        "image_paths": list(data["image_paths"]),
    }


def loo_predict(query_emb: np.ndarray, exclude_source_id: int, gallery: dict,
                k: int, temperature: float) -> tuple[int, float, np.ndarray]:
    """kNN prediction with one source excluded. Returns (pred_idx, confidence, scores)."""
    mask = gallery["source_ids"] != exclude_source_id
    embs = gallery["embeddings"][mask]
    labels = gallery["labels"][mask]
    class_names = gallery["class_names"]
    n_classes = len(class_names)

    sims = embs @ query_emb
    k_actual = min(k, len(sims))
    top_idx = np.argpartition(sims, -k_actual)[-k_actual:]
    top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]

    top_sims = sims[top_idx]
    top_labels = labels[top_idx]

    shifted = top_sims / temperature
    shifted -= shifted.max()
    weights = np.exp(shifted)
    weights /= weights.sum()

    scores = np.zeros(n_classes, dtype=np.float64)
    for w, lbl in zip(weights, top_labels):
        scores[lbl] += w

    pred_idx = int(np.argmax(scores))
    return pred_idx, float(scores[pred_idx]), scores


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    class_names: list[str]) -> dict:
    """Per-class precision, recall, F1 + macro/weighted averages."""
    results = {}

    for i, cls in enumerate(class_names):
        tp = int(((y_true == i) & (y_pred == i)).sum())
        fp = int(((y_true != i) & (y_pred == i)).sum())
        fn = int(((y_true == i) & (y_pred != i)).sum())
        support = int((y_true == i).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        results[cls] = {
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4),
            "support":   support,
            "tp": tp, "fp": fp, "fn": fn,
        }

    # Macro average (equal weight per class)
    macro_p = np.mean([results[c]["precision"] for c in class_names])
    macro_r = np.mean([results[c]["recall"]    for c in class_names])
    macro_f = np.mean([results[c]["f1"]        for c in class_names])

    # Weighted average (weight by support)
    total = sum(results[c]["support"] for c in class_names)
    w_p = sum(results[c]["precision"] * results[c]["support"] for c in class_names) / total
    w_r = sum(results[c]["recall"]    * results[c]["support"] for c in class_names) / total
    w_f = sum(results[c]["f1"]        * results[c]["support"] for c in class_names) / total

    accuracy = float((y_true == y_pred).mean())

    results["_macro"]    = {"precision": round(float(macro_p), 4),
                             "recall":    round(float(macro_r), 4),
                             "f1":        round(float(macro_f), 4)}
    results["_weighted"] = {"precision": round(float(w_p), 4),
                             "recall":    round(float(w_r), 4),
                             "f1":        round(float(w_f), 4)}
    results["_accuracy"] = round(accuracy, 4)
    return results


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], out_path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.colorbar(im, ax=ax)

        ax.set(xticks=np.arange(len(class_names)),
               yticks=np.arange(len(class_names)),
               xticklabels=class_names,
               yticklabels=class_names,
               ylabel="True label",
               xlabel="Predicted label",
               title="Confusion Matrix (LOO, counts)")

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)

        # Annotate cells
        thresh = cm.max() / 2.0
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax.text(j, i, f"{cm[i, j]}",
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=10)

        fig.tight_layout()
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"[evaluate] Confusion matrix → {out_path}")
    except ImportError:
        print("[evaluate] matplotlib not found, skipping confusion matrix plot")


def run_loo(gallery_path: str, k: int, temperature: float, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    gallery = load_gallery_data(gallery_path)
    class_names = gallery["class_names"]
    image_paths = gallery["image_paths"]
    n_sources = len(image_paths)

    print(f"\nRunning LOO evaluation on {n_sources} source images ...")
    print(f"  k={k}, temperature={temperature}\n")

    y_true = []
    y_pred = []
    per_image_results = []

    t0 = time.time()

    # We need embeddings for each original (un-augmented) image
    # Embed source images fresh (not augmented versions from gallery)
    for source_id in tqdm(range(n_sources), desc="LOO"):
        # Get label from gallery
        mask = gallery["source_ids"] == source_id
        true_lbl = int(gallery["labels"][mask][0])

        # Embed fresh from disk — not from gallery augmentations
        try:
            img = Image.open(image_paths[source_id]).convert("RGB")
        except Exception as e:
            print(f"[WARNING] Cannot open {image_paths[source_id]}: {e}")
            continue

        query_emb = backbone.embed_image(img)

        pred_idx, conf, scores = loo_predict(
            query_emb, source_id, gallery, k=k, temperature=temperature
        )

        y_true.append(true_lbl)
        y_pred.append(pred_idx)

        per_image_results.append({
            "path": image_paths[source_id],
            "true": class_names[true_lbl],
            "pred": class_names[pred_idx],
            "confidence": round(conf, 4),
            "correct": true_lbl == pred_idx,
            "scores": {c: round(float(scores[i]), 4)
                       for i, c in enumerate(class_names)},
        })

    elapsed = time.time() - t0
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Metrics
    metrics = compute_metrics(y_true, y_pred, list(class_names))

    # Confusion matrix
    n_cls = len(class_names)
    cm = np.zeros((n_cls, n_cls), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    # Print report
    print("\n" + "=" * 60)
    print("LOO EVALUATION RESULTS")
    print("=" * 60)
    print(f"{'Class':<15} {'Prec':>6} {'Recall':>7} {'F1':>6} {'Support':>8}")
    print("-" * 60)
    for cls in class_names:
        m = metrics[cls]
        print(f"{cls:<15} {m['precision']:>6.3f} {m['recall']:>7.3f} "
              f"{m['f1']:>6.3f} {m['support']:>8}")
    print("-" * 60)
    print(f"{'macro avg':<15} {metrics['_macro']['precision']:>6.3f} "
          f"{metrics['_macro']['recall']:>7.3f} {metrics['_macro']['f1']:>6.3f}")
    print(f"{'weighted avg':<15} {metrics['_weighted']['precision']:>6.3f} "
          f"{metrics['_weighted']['recall']:>7.3f} {metrics['_weighted']['f1']:>6.3f}")
    print(f"\nAccuracy: {metrics['_accuracy']:.4f}")
    print(f"Evaluated: {len(y_true)} images in {elapsed:.1f}s "
          f"({elapsed/len(y_true)*1000:.1f}ms/img)")

    # Misclassified
    wrong = [r for r in per_image_results if not r["correct"]]
    print(f"\nMisclassified ({len(wrong)}/{len(per_image_results)}):")
    for r in wrong:
        print(f"  {os.path.basename(r['path']):40s} "
              f"true={r['true']:12s} pred={r['pred']:12s} conf={r['confidence']:.3f}")

    # Save
    report = {
        "k": k,
        "temperature": temperature,
        "n_evaluated": len(y_true),
        "elapsed_sec": round(elapsed, 2),
        "metrics": metrics,
        "confusion_matrix": cm.tolist(),
        "class_names": list(class_names),
        "per_image": per_image_results,
    }
    json_path = os.path.join(out_dir, "eval_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nFull report → {json_path}")

    cm_path = os.path.join(out_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, list(class_names), cm_path)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gallery", default=os.path.join(config.GALLERY_DIR, "gallery.npz"))
    parser.add_argument("--k", type=int, default=config.K_NEAREST)
    parser.add_argument("--temperature", type=float, default=config.SIMILARITY_TEMPERATURE)
    parser.add_argument("--out", default="eval_results")
    args = parser.parse_args()

    run_loo(args.gallery, args.k, args.temperature, args.out)
