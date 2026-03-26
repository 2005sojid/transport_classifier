"""Central configuration for transport classifier."""
import os

# Data — override with env var VEHICLE_REFS_DIR or edit directly
DATA_DIR = os.environ.get("VEHICLE_REFS_DIR", "D:/vehicle_refs")
GALLERY_DIR = os.path.join(os.path.dirname(__file__), "gallery")

# Classes — order defines class index stored in gallery (do not reorder after build)
CLASSES = ["ambulance", "bus", "car", "fire_truck", "minibus", "police"]

# Model (loaded via torch.hub: facebookresearch/dinov2 → dinov2_vitl14)
USE_PATCH_TOKENS = False  # CLS only — patch tokens add noise, -0.4% accuracy

# Gallery build
N_VARIANTS_DEFAULT = 8   # augmented variants per image
N_VARIANTS_FIRE_TRUCK = 16  # more variants for under-represented class

# kNN inference
K_NEAREST = 11           # number of nearest neighbors for voting
SIMILARITY_TEMPERATURE = 0.10  # softmax temperature for weighting

# Confidence thresholds
CONFIDENCE_REJECT = 0.35   # below this → "unknown"
CONFIDENCE_LOW    = 0.55   # below this → warn low confidence
