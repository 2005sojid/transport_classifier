"""
Build the reference gallery: embed all images + augmentations, save to disk.

Gallery layout (saved as .npz):
  embeddings  : float32 (N, D)
  labels      : int32   (N,)   — class index
  source_ids  : int32   (N,)   — which original image each row came from
  class_names : list of str
  image_paths : list of str    — original file path (one per source_id)

Usage:
  python build_gallery.py
"""
import os
import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from PIL import Image
from tqdm import tqdm

import config
import backbone
import augmentation


def load_dataset(data_dir: str, classes: list[str]) -> list[dict]:
    """Return list of {path, label_idx, class_name} dicts."""
    VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    records = []
    for idx, cls in enumerate(classes):
        folder = os.path.join(data_dir, cls)
        if not os.path.isdir(folder):
            print(f"[WARNING] Missing folder: {folder}")
            continue
        files = [
            f for f in sorted(os.listdir(folder))
            if os.path.splitext(f)[1].lower() in VALID_EXT
        ]
        print(f"  {cls:15s}: {len(files)} images")
        for f in files:
            records.append({
                "path": os.path.join(folder, f),
                "label_idx": idx,
                "class_name": cls,
            })
    return records


def build(data_dir: str = config.DATA_DIR,
          gallery_dir: str = config.GALLERY_DIR,
          classes: list[str] = config.CLASSES,
          batch_size: int = 16):

    os.makedirs(gallery_dir, exist_ok=True)
    print(f"\nScanning {data_dir} ...")
    records = load_dataset(data_dir, classes)
    print(f"Total source images: {len(records)}\n")

    all_embeddings = []
    all_labels = []
    all_source_ids = []
    all_paths = []

    # Embed in batches for efficiency
    image_buffer: list[Image.Image] = []
    meta_buffer: list[tuple[int, int]] = []  # (source_id, label_idx)

    def flush_buffer():
        if not image_buffer:
            return
        embs = backbone.embed_images(image_buffer)
        for emb, (src_id, lbl) in zip(embs, meta_buffer):
            all_embeddings.append(emb)
            all_labels.append(lbl)
            all_source_ids.append(src_id)
        image_buffer.clear()
        meta_buffer.clear()

    for source_id, rec in enumerate(tqdm(records, desc="Building gallery")):
        try:
            img = Image.open(rec["path"]).convert("RGB")
        except Exception as e:
            print(f"[WARNING] Cannot open {rec['path']}: {e}")
            continue

        all_paths.append(rec["path"])

        n_variants = (
            config.N_VARIANTS_FIRE_TRUCK
            if rec["class_name"] == "fire_truck"
            else config.N_VARIANTS_DEFAULT
        )
        variants = augmentation.get_variants(img, n=n_variants, seed=source_id)

        for v in variants:
            image_buffer.append(v)
            meta_buffer.append((source_id, rec["label_idx"]))
            if len(image_buffer) >= batch_size:
                flush_buffer()

    flush_buffer()

    embeddings = np.array(all_embeddings, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.int32)
    source_ids = np.array(all_source_ids, dtype=np.int32)

    out_path = os.path.join(gallery_dir, "gallery.npz")
    np.savez_compressed(
        out_path,
        embeddings=embeddings,
        labels=labels,
        source_ids=source_ids,
        class_names=np.array(classes),
        image_paths=np.array(all_paths),
    )

    print(f"\nGallery saved → {out_path}")
    print(f"  Total vectors : {len(embeddings)}")
    print(f"  Embedding dim : {embeddings.shape[1]}")
    print(f"  Source images : {len(all_paths)}")
    for idx, cls in enumerate(classes):
        mask = labels == idx
        n_src = len(set(source_ids[mask]))
        print(f"  {cls:15s}: {mask.sum():4d} vectors from {n_src} images")


if __name__ == "__main__":
    build()
