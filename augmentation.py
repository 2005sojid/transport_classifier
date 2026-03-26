"""
Image augmentations for gallery building.

Rules:
- NO hue/saturation shift — color is primary discriminator (fire_truck=red, etc.)
- Geometric + mild photometric only
- Each call to get_variants() returns deterministic augmented PIL images
"""
import random
from PIL import Image, ImageEnhance


def _brightness(img: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Brightness(img).enhance(factor)


def _contrast(img: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Contrast(img).enhance(factor)


def _crop_center(img: Image.Image, scale: float) -> Image.Image:
    w, h = img.size
    nw, nh = int(w * scale), int(h * scale)
    left = (w - nw) // 2
    top = (h - nh) // 2
    return img.crop((left, top, left + nw, top + nh)).resize((w, h), Image.LANCZOS)


def _crop_random(img: Image.Image, scale: float, rng: random.Random) -> Image.Image:
    w, h = img.size
    nw, nh = int(w * scale), int(h * scale)
    left = rng.randint(0, w - nw)
    top = rng.randint(0, h - nh)
    return img.crop((left, top, left + nw, top + nh)).resize((w, h), Image.LANCZOS)


def _rotate(img: Image.Image, angle: float) -> Image.Image:
    return img.rotate(angle, expand=False, fillcolor=(128, 128, 128))


def get_variants(img: Image.Image, n: int = 8, seed: int = 0) -> list[Image.Image]:
    """
    Return exactly n augmented versions of img (deterministic for same seed).
    First variant is always the original (unmodified).
    """
    rng = random.Random(seed)
    variants = [img]  # variant 0: original

    aug_pool = [
        lambda i: i.transpose(Image.FLIP_LEFT_RIGHT),
        lambda i: _brightness(i, 0.85),
        lambda i: _brightness(i, 1.15),
        lambda i: _contrast(i, 0.85),
        lambda i: _contrast(i, 1.15),
        lambda i: _crop_center(i, 0.92),
        lambda i: _crop_center(i, 0.85),
        lambda i: _rotate(i, 5),
        lambda i: _rotate(i, -5),
        lambda i: _rotate(i, 3),
        lambda i: _crop_random(i, 0.90, rng),
        lambda i: _crop_random(i, 0.85, rng),
        lambda i: _brightness(_contrast(i, 1.1), 0.9),
        lambda i: _brightness(_contrast(i, 0.9), 1.1),
        lambda i: i.transpose(Image.FLIP_LEFT_RIGHT).rotate(3),
        lambda i: _crop_center(i, 0.88).transpose(Image.FLIP_LEFT_RIGHT),
    ]

    rng.shuffle(aug_pool)

    for fn in aug_pool:
        if len(variants) >= n:
            break
        variants.append(fn(img))

    # If somehow we need more, repeat with different brightness combos
    while len(variants) < n:
        f = 0.80 + rng.random() * 0.40
        variants.append(_brightness(img, f))

    return variants[:n]
