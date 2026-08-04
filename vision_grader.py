"""
Module 1: Vision Grader (Mocked Computer Vision)
================================================================================
Simulates a YOLO-style ripeness classifier for Fresh Fruit Bunch (FFB) photos.

For this MVP we deliberately do NOT train/run a real object-detection model.
Instead this is a lightweight *color-informed* mock: we sample the image's
dominant hue and "dark pixel" ratio (a rough proxy for the blackening seen on
over-ripe/rotted bunches) and use that to bias a weighted random draw across
the four ripeness categories. The result feels responsive to what was
actually uploaded, while remaining a clearly-labeled, easily swappable
placeholder for a future real CV model (e.g. a fine-tuned YOLOv8 detector).
"""

import colorsys
import os
import random

import numpy as np
from PIL import Image, ImageDraw

RIPENESS_SCORE_MAP = {"Underripe": 0, "Ripe": 1, "Overripe": 2, "Rotted": 3}

RIPENESS_META = {
    "Underripe": {"color": "#65a30d", "emoji": "🟢", "note": "Low oil yield risk"},
    "Ripe":      {"color": "#d97706", "emoji": "🟠", "note": "Optimal harvest window"},
    "Overripe":  {"color": "#c2410c", "emoji": "🟤", "note": "Elevated native FFA"},
    "Rotted":    {"color": "#7f1d1d", "emoji": "⚫", "note": "Severe FFA contamination risk"},
}

PROCESSING_STEPS = [
    "Detecting fruitlets...",
    "Sampling surface color profile...",
    "Cross-referencing ripeness heuristics...",
]


def analyze_ffb_image(image: Image.Image) -> dict:
    """Run the mocked CV pipeline on a PIL image and return a ripeness verdict.

    The weighting is influenced by the image's average hue and darkness so
    that, e.g., a mostly dark/black photo skews toward "Rotted" and a
    mostly orange-red photo skews toward "Ripe"/"Overripe" -- but the final
    pick is still a weighted random draw, matching a real classifier's
    probabilistic confidence rather than a deterministic rule.
    """
    img_small = image.convert("RGB").resize((60, 60))
    pixels = np.asarray(img_small, dtype=float).reshape(-1, 3)

    avg_r, avg_g, avg_b = (float(v) for v in pixels.mean(axis=0))
    hue, _, _ = colorsys.rgb_to_hsv(avg_r / 255, avg_g / 255, avg_b / 255)
    hue_deg = hue * 360

    brightness = pixels.mean(axis=1)
    dark_ratio = float((brightness < 60).mean())

    if dark_ratio > 0.30:
        weights = {"Rotted": 0.55, "Overripe": 0.30, "Ripe": 0.10, "Underripe": 0.05}
    elif 5 <= hue_deg <= 35:
        weights = {"Ripe": 0.50, "Overripe": 0.30, "Underripe": 0.12, "Rotted": 0.08}
    elif 35 < hue_deg <= 75:
        weights = {"Underripe": 0.55, "Ripe": 0.30, "Overripe": 0.10, "Rotted": 0.05}
    else:
        weights = {"Ripe": 0.40, "Underripe": 0.25, "Overripe": 0.25, "Rotted": 0.10}

    categories = list(weights)
    probs = list(weights.values())
    category = random.choices(categories, weights=probs, k=1)[0]
    confidence = round(random.uniform(0.78, 0.97), 2)

    return {
        "category": category,
        "confidence": confidence,
        "ripeness_score": RIPENESS_SCORE_MAP[category],
        "avg_color_rgb": (int(avg_r), int(avg_g), int(avg_b)),
        "dominant_hue_deg": round(hue_deg, 1),
        "meta": RIPENESS_META[category],
    }


def _swatch_image(base_rgb, size=(360, 360), seed=0):
    """Generate a small speckled color-swatch PNG standing in for a real FFB photo,
    used only for the one-click demo samples in the UI."""
    rng = np.random.default_rng(seed)
    arr = np.tile(np.array(base_rgb, dtype=np.uint8), (size[1], size[0], 1))
    noise = rng.integers(-18, 18, size=arr.shape)
    arr = np.clip(arr.astype(int) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="RGB")

    draw = ImageDraw.Draw(img)
    rng2 = np.random.default_rng(seed + 1)
    for _ in range(int(size[0] * size[1] * 0.003)):
        x, y = int(rng2.integers(0, size[0])), int(rng2.integers(0, size[1]))
        r = int(rng2.integers(4, 14))
        shade = np.clip(np.array(base_rgb) * rng2.uniform(0.5, 0.85), 0, 255).astype(int)
        draw.ellipse([x - r, y - r, x + r, y + r], fill=tuple(shade))
    return img


def ensure_sample_images(folder: str = "assets") -> dict:
    """Create a few synthetic demo swatches (color-only stand-ins for real FFB
    photos) so judges can one-click test the pipeline without a real upload.
    Returns {label: filepath}. Idempotent -- only generates files once."""
    os.makedirs(folder, exist_ok=True)
    samples = {
        "Underripe": (124, 150, 42),
        "Ripe": (196, 108, 20),
        "Overripe": (140, 62, 22),
        "Rotted": (42, 34, 30),
    }
    paths = {}
    for i, (label, rgb) in enumerate(samples.items()):
        path = os.path.join(folder, f"sample_{i}.png")
        if not os.path.exists(path):
            _swatch_image(rgb, seed=i).save(path)
        paths[label] = path
    return paths
