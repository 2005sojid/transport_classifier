"""
DINOv2 ViT-B/14 feature extractor via torch.hub (no HuggingFace auth needed).
Returns L2-normalized embeddings ready for cosine similarity via dot product.
"""
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from PIL import Image

import config

_model = None
_device = None

# DINOv2 standard preprocessing (ImageNet stats, 518x518 for ViT-B/14)
_DINO_TRANSFORM = T.Compose([
    T.Resize(518, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(518),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _load():
    global _model, _device
    if _model is not None:
        return
    _device = "cpu"
    print(f"[backbone] Loading dinov2_vitb14 via torch.hub on {_device} ...")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _model = torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_vitb14",
            pretrained=True,
        )
    _model.eval().to(_device)
    print("[backbone] Ready.")


@torch.no_grad()
def embed_images(images: list[Image.Image]) -> np.ndarray:
    """
    Embed a list of PIL images.
    Returns float32 array of shape (N, D), L2-normalized.
    """
    _load()
    tensors = torch.stack([_DINO_TRANSFORM(img) for img in images]).to(_device)

    if config.USE_PATCH_TOKENS:
        out = _model.forward_features(tensors)
        cls = out["x_norm_clstoken"]          # (N, 768)
        patches = out["x_norm_patchtokens"]   # (N, P, 768)
        mean_patch = patches.mean(dim=1)       # (N, 768)
        emb = torch.cat([cls, mean_patch], dim=1)  # (N, 1536)
    else:
        emb = _model(tensors)                  # (N, 768) — CLS token

    emb = F.normalize(emb, dim=1)
    return emb.cpu().float().numpy()


def embed_image(image: Image.Image) -> np.ndarray:
    """Single image → (D,) normalized vector."""
    return embed_images([image])[0]
