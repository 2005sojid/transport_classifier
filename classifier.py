"""
Weighted cosine kNN classifier.

Algorithm:
  1. Compute cosine similarity between query and all gallery vectors
     (fast via dot product since vectors are L2-normalized)
  2. Take top-K neighbors
  3. Weight each neighbor: w_i = softmax(sim_i / T)
  4. Aggregate weighted votes per class
  5. Return predicted class + confidence

Confidence is the weighted vote share of the winning class.
If confidence < CONFIDENCE_REJECT → return "unknown".
"""
from __future__ import annotations

import os
import numpy as np
from dataclasses import dataclass
from PIL import Image

import config
import backbone


@dataclass
class Prediction:
    label: str          # winning class name (or "unknown")
    confidence: float   # [0, 1] weighted vote share
    scores: dict        # {class_name: score} for all classes
    top_neighbors: list[dict]  # [{path, class, similarity}, ...]
    low_confidence: bool       # True if below CONFIDENCE_LOW threshold


class GalleryKNN:
    def __init__(self, gallery_path: str | None = None):
        if gallery_path is None:
            gallery_path = os.path.join(config.GALLERY_DIR, "gallery.npz")
        self._load(gallery_path)

    def _load(self, path: str):
        data = np.load(path, allow_pickle=True)
        self.embeddings: np.ndarray = data["embeddings"]   # (N, D) float32, L2-normed
        self.labels: np.ndarray = data["labels"]           # (N,) int32
        self.source_ids: np.ndarray = data["source_ids"]   # (N,) int32
        self.class_names: list[str] = list(data["class_names"])
        self.image_paths: list[str] = list(data["image_paths"])
        self._n_classes = len(self.class_names)
        print(f"[classifier] Gallery loaded: {len(self.embeddings)} vectors, "
              f"{self._n_classes} classes")

    def predict_embedding(
        self,
        query_emb: np.ndarray,
        k: int = config.K_NEAREST,
        temperature: float = config.SIMILARITY_TEMPERATURE,
        exclude_source_ids: set[int] | None = None,
    ) -> Prediction:
        """
        Classify a pre-computed L2-normalized query embedding.
        exclude_source_ids: source images to exclude from gallery (for LOO eval).
        """
        gallery_emb = self.embeddings

        if exclude_source_ids:
            mask = ~np.isin(self.source_ids, list(exclude_source_ids))
            gallery_emb = gallery_emb[mask]
            labels = self.labels[mask]
            source_ids = self.source_ids[mask]
        else:
            labels = self.labels
            source_ids = self.source_ids

        # Cosine similarities (dot product, vectors are L2-normalized)
        sims = gallery_emb @ query_emb  # (N,)

        # Top-k
        k_actual = min(k, len(sims))
        top_idx = np.argpartition(sims, -k_actual)[-k_actual:]
        top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]  # sort descending

        top_sims = sims[top_idx]
        top_labels = labels[top_idx]

        # Softmax weighting
        shifted = top_sims / temperature
        shifted -= shifted.max()  # numerical stability
        weights = np.exp(shifted)
        weights /= weights.sum()

        # Aggregate votes
        scores = np.zeros(self._n_classes, dtype=np.float64)
        for w, lbl in zip(weights, top_labels):
            scores[lbl] += w

        pred_idx = int(np.argmax(scores))
        confidence = float(scores[pred_idx])

        label = self.class_names[pred_idx]
        if confidence < config.CONFIDENCE_REJECT:
            label = "unknown"

        # Build top_neighbors — one entry per unique source image (deduplicate augmentations)
        # top_idx indexes into the (possibly filtered) gallery_emb / labels / source_ids arrays
        top_neighbors = []
        seen_sources: set[int] = set()
        rank = 1
        for i, sim in zip(top_idx, top_sims):
            src_id = int(source_ids[i])
            if src_id in seen_sources:
                continue
            seen_sources.add(src_id)
            top_neighbors.append({
                "rank": rank,
                "class": self.class_names[int(labels[i])],
                "similarity": float(sim),
                "path": self.image_paths[src_id] if src_id < len(self.image_paths) else "?",
            })
            rank += 1

        return Prediction(
            label=label,
            confidence=confidence,
            scores={cls: float(scores[i]) for i, cls in enumerate(self.class_names)},
            top_neighbors=top_neighbors,
            low_confidence=confidence < config.CONFIDENCE_LOW,
        )

    def predict(
        self,
        image: Image.Image,
        k: int = config.K_NEAREST,
    ) -> Prediction:
        """End-to-end: PIL Image → Prediction."""
        emb = backbone.embed_image(image)
        return self.predict_embedding(emb, k=k)


# Singleton for repeated inference
_instance: GalleryKNN | None = None


def get_classifier() -> GalleryKNN:
    global _instance
    if _instance is None:
        _instance = GalleryKNN()
    return _instance


def classify_image(image: Image.Image) -> Prediction:
    """Convenience function for production use."""
    return get_classifier().predict(image)
