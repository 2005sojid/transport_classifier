# Transport Classifier

Vision-based classifier for special transport vehicles: ambulance, police, fire truck, bus, minibus vs regular car.

**Architecture**: DINOv2 ViT-B/14 → L2-normalized CLS embedding (768-dim) → weighted cosine kNN

## Classes

| Class | Description |
|-------|-------------|
| `ambulance` | White minivan with red/orange stripe |
| `police` | White sedan (Cobalt/BYD) with blue markings |
| `fire_truck` | Bright red truck with equipment |
| `bus` | Large city bus (Yutong) |
| `minibus` | White van (Gazelle-type) |
| `car` | Regular passenger car (negative class) |

## How it works

1. **Build gallery** — embed all reference images + augmentations into a `.npz` index (once)
2. **Inference** — embed query image, find k nearest neighbors by cosine similarity, weighted vote

Color is the primary discriminator: fire truck (red), ambulance (red stripe), police (blue stripes).
No model training — adding new reference images only requires rebuilding the gallery.

## Results (Leave-One-Out evaluation, 391 images)

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| ambulance | 0.737 | 0.875 | 0.800 | 48 |
| bus | 1.000 | 1.000 | 1.000 | 52 |
| car | 0.805 | 0.692 | 0.744 | 143 |
| fire_truck | 0.909 | 0.909 | 0.909 | 11 |
| minibus | 1.000 | 0.911 | 0.954 | 45 |
| police | 0.701 | 0.815 | 0.754 | 92 |
| **weighted avg** | **0.823** | **0.816** | **0.816** | |

**Accuracy: 81.6%** — honest LOO (all augmented copies of test image excluded from gallery)
Inference: 44.8 ms/image on GPU (CUDA)

## Setup

```bash
pip install -r requirements.txt
```

DINOv2 model (~330 MB) downloads automatically on first run via `torch.hub`.

## Usage

### 1. Prepare reference images

Organize images into folders by class:
```
your_refs/
  ambulance/  *.jpg
  bus/        *.jpg
  car/        *.jpg
  fire_truck/ *.jpg
  minibus/    *.jpg
  police/     *.jpg
```

Set the path in `config.py` or via environment variable:
```bash
set VEHICLE_REFS_DIR=path/to/your_refs   # Windows
export VEHICLE_REFS_DIR=path/to/your_refs  # Linux/Mac
```

### 2. Build gallery

```bash
python build_gallery.py
```

Embeds all reference images + augmentations into `gallery/gallery.npz`.

### 3. Classify an image

```bash
python infer.py path/to/image.jpg
python infer.py path/to/image.jpg --verbose   # show scores and top neighbors
```

### 4. Evaluate (Leave-One-Out)

```bash
python evaluate.py
```

Outputs per-class metrics, confusion matrix PNG, and full JSON report in `eval_results/`.

## Files

| File | Description |
|------|-------------|
| `backbone.py` | DINOv2 ViT-B/14 via torch.hub, L2-normalized embeddings |
| `augmentation.py` | Geometric augmentations (no color shift — color is discriminative) |
| `build_gallery.py` | Build reference gallery from image folders |
| `classifier.py` | `GalleryKNN` — cosine kNN with softmax weighting |
| `evaluate.py` | Honest LOO evaluation with confusion matrix |
| `infer.py` | CLI inference |
| `config.py` | All hyperparameters |

## Configuration (`config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `K_NEAREST` | 11 | Number of neighbors |
| `SIMILARITY_TEMPERATURE` | 0.10 | Softmax temperature for weighting |
| `N_VARIANTS_DEFAULT` | 8 | Augmentation variants per image |
| `N_VARIANTS_FIRE_TRUCK` | 16 | Extra variants for underrepresented class |
| `CONFIDENCE_REJECT` | 0.35 | Below this → "unknown" |
| `CONFIDENCE_LOW` | 0.55 | Below this → low confidence warning |
