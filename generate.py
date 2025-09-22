"""Texture variant generator script.

This script processes PNG textures found in the ``input/`` directory and outputs
variant textures (zombie, frozen, corrupted, toxic, oceanic, nether, celestial)
under ``output/<variant>/``. Variants apply configurable HSV tint shifts and can
optionally blend overlays if present.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

from PIL import Image, ImageDraw, ImageFilter
import numpy as np
from scipy import ndimage
import random

from utils_procedural import (
    generate_perlin_noise,
    generate_poisson_disc_samples,
    random_walk_lines,
)

# Directory configuration
INPUT_DIR = Path("input")
OVERLAY_DIR = Path("overlays")
MASK_DIR = Path("masks")
OUTPUT_DIR = Path("output")

# Variant configuration
VARIANTS: Dict[str, Dict[str, object]] = {
    "zombie": {
        "tint": {"hue_shift": -0.10, "saturation_shift": -0.30, "value_shift": -0.10},
        "overlay_opacity": 0.4,
    },
    "frozen": {
        "tint": {"hue_shift": 0.55, "saturation_shift": -0.15, "value_shift": 0.10},
        "overlay_opacity": 0.6,
    },
    "corrupted": {
        "tint": {"hue_shift": -0.30, "saturation_shift": 0.25, "value_shift": -0.05},
        "overlay_opacity": 0.7,
    },
    "toxic": {
        "tint": {"hue_shift": 0.20, "saturation_shift": 0.30, "value_shift": 0.05},
        "overlay_opacity": 0.5,
    },
    "oceanic": {
        "tint": {"hue_shift": 0.45, "saturation_shift": 0.10, "value_shift": -0.05},
        "overlay_opacity": 0.35,
    },
    "nether": {
        "tint": {"hue_shift": -0.55, "saturation_shift": 0.35, "value_shift": 0.10},
        "overlay_opacity": 0.65,
    },
    "celestial": {
        "tint": {"hue_shift": 0.75, "saturation_shift": -0.05, "value_shift": 0.20},
        "overlay_opacity": 0.5,
    },
}

# Variant exclusions based on base texture name fragments
EXCLUSIONS = {
    "zombie": ["zombie"],
    "frozen": ["stray", "snow_golem"],
    "corrupted": ["enderman", "shulker", "endermite"],
    "toxic": [],
    "oceanic": ["drowned", "guardian", "elder_guardian", "squid", "turtle"],
    "nether": [
        "zombie_pigman",
        "wither_skeleton",
        "blaze",
        "ghast",
        "magma_cube",
        "piglin",
        "hoglin",
        "strider",
        "wither",
    ],
    "celestial": [],
}

OVERLAY_CACHE: Dict[Tuple[str, Tuple[int, int]], Image.Image | None] = {}
MASK_CACHE: Dict[Tuple[str, Tuple[int, int]], Image.Image | None] = {}
FEATURE_CACHE: Dict[Tuple[int, int], Dict[str, object]] = {}


def _normalize_noise(noise: np.ndarray) -> np.ndarray:
    minimum = float(np.min(noise))
    maximum = float(np.max(noise))
    if maximum - minimum <= 1e-6:
        return np.zeros_like(noise, dtype=np.float32)
    return (noise - minimum) / (maximum - minimum)


def _default_features(size: Tuple[int, int]) -> Dict[str, object]:
    width, height = size
    eye_radius = max(3, min(width, height) // 18)
    joint_radius = max(4, min(width, height) // 14)
    return {
        "eyes": [
            (int(width * 0.35), int(height * 0.28), eye_radius),
            (int(width * 0.65), int(height * 0.28), eye_radius),
        ],
        "joints": [
            (int(width * 0.30), int(height * 0.60), joint_radius),
            (int(width * 0.70), int(height * 0.60), joint_radius),
            (int(width * 0.30), int(height * 0.85), joint_radius),
            (int(width * 0.70), int(height * 0.85), joint_radius),
        ],
        "edges": None,
    }


def _get_features_for_size(size: Tuple[int, int]) -> Dict[str, object]:
    if size not in FEATURE_CACHE:
        FEATURE_CACHE[size] = _default_features(size)
    return FEATURE_CACHE[size]


def _create_radial_glow(
    size: Tuple[int, int], color: Tuple[int, int, int], radius_factor: float, intensity: int
) -> Image.Image:
    width, height = size
    glow_mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(glow_mask)
    radius = int(min(width, height) * radius_factor)
    cx, cy = width // 2, height // 2
    draw.ellipse(
        (cx - radius, cy - radius, cx + radius, cy + radius),
        fill=intensity,
    )
    blur_radius = max(2, int(min(width, height) * 0.1))
    glow_mask = glow_mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    glow_layer = Image.new("RGBA", size, (*color, 0))
    glow_layer.putalpha(glow_mask)
    return glow_layer


def generate_overlay_if_missing(variant: str, size: Tuple[int, int]) -> Image.Image | None:
    cache_key = (variant, size)
    if cache_key in OVERLAY_CACHE:
        cached = OVERLAY_CACHE[cache_key]
        return cached.copy() if cached is not None else None

    overlay_path = OVERLAY_DIR / f"{variant}.png"
    overlay: Image.Image | None = None
    if overlay_path.exists():
        try:
            with Image.open(overlay_path) as overlay_image:
                overlay = overlay_image.convert("RGBA")
        except OSError as exc:
            print(f"⚠️  No se pudo abrir el overlay {overlay_path}: {exc}")
            overlay = None
        if overlay is not None and overlay.size != size:
            overlay = overlay.resize(size, Image.LANCZOS)
    else:
        overlay = generate_variant_overlay(variant, size)
        if overlay is not None:
            OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
            try:
                overlay.save(overlay_path)
                print(f"🎨 Overlay generado: {overlay_path}")
            except OSError as exc:
                print(f"⚠️  No se pudo guardar el overlay {overlay_path}: {exc}")
    OVERLAY_CACHE[cache_key] = overlay
    return overlay.copy() if overlay is not None else None


def generate_mask_if_missing(variant: str, size: Tuple[int, int]) -> Image.Image | None:
    cache_key = (variant, size)
    if cache_key in MASK_CACHE:
        cached = MASK_CACHE[cache_key]
        return cached.copy() if cached is not None else None

    mask_path = MASK_DIR / f"{variant}.png"
    mask: Image.Image | None = None
    if mask_path.exists():
        try:
            with Image.open(mask_path) as mask_image:
                mask = mask_image.convert("RGBA")
        except OSError as exc:
            print(f"⚠️  No se pudo abrir la máscara {mask_path}: {exc}")
            mask = None
        if mask is not None and mask.size != size:
            mask = mask.resize(size, Image.LANCZOS)
    else:
        mask = generate_variant_mask(variant, size)
        if mask is not None:
            MASK_DIR.mkdir(parents=True, exist_ok=True)
            try:
                mask.save(mask_path)
                print(f"💡 Mask generada: {mask_path}")
            except OSError as exc:
                print(f"⚠️  No se pudo guardar la máscara {mask_path}: {exc}")
    MASK_CACHE[cache_key] = mask
    return mask.copy() if mask is not None else None

class TextureProcessingError(RuntimeError):
    """Raised when the texture processing pipeline encounters an error."""


def detect_anatomical_features(image: Image.Image) -> Dict[str, object]:
    """Detect simple anatomical landmarks with graceful fallbacks."""

    width, height = image.size
    grayscale = np.array(image.convert("L"), dtype=np.float32)
    normalized = grayscale / 255.0 if np.max(grayscale) > 0 else grayscale

    features = _default_features((width, height))

    try:
        top_start = int(height * 0.18)
        top_end = max(top_start + 1, int(height * 0.45))
        top_region = normalized[top_start:top_end, :]
        if top_region.size:
            column_scores = top_region.mean(axis=0)
            candidate_indices = np.argsort(column_scores)[:4]
            detected_eyes: list[Tuple[int, int, int]] = []
            radius = max(3, min(width, height) // 18)
            for idx in candidate_indices:
                x_pos = int(idx)
                if any(abs(x_pos - existing[0]) < max(2, width // 12) for existing in detected_eyes):
                    continue
                column = top_region[:, x_pos]
                y_offset = int(np.argmin(column)) if column.size else 0
                detected_eyes.append((x_pos, top_start + y_offset, radius))
                if len(detected_eyes) >= 2:
                    break
            if len(detected_eyes) == 2:
                detected_eyes.sort(key=lambda item: item[0])
                features["eyes"] = detected_eyes
    except Exception:
        # Fall back to defaults if detection fails
        pass

    try:
        lower_start = int(height * 0.55)
        lower_region = normalized[lower_start:, :]
        if lower_region.size:
            column_variance = lower_region.var(axis=0)
            candidate_cols = np.argsort(column_variance)[-6:]
            joints: list[Tuple[int, int, int]] = []
            radius = max(4, min(width, height) // 14)
            for idx in reversed(candidate_cols):
                x_pos = int(idx)
                column = lower_region[:, x_pos]
                y_offset = int(np.argmin(column)) if column.size else 0
                candidate = (x_pos, lower_start + y_offset, radius)
                if all(abs(candidate[0] - existing[0]) > max(4, width // 10) for existing in joints):
                    joints.append(candidate)
                if len(joints) >= 4:
                    break
            if joints:
                joints.sort(key=lambda item: item[0])
                features["joints"] = joints
    except Exception:
        pass

    try:
        gradient_x = ndimage.sobel(normalized, axis=1, mode="reflect")
        gradient_y = ndimage.sobel(normalized, axis=0, mode="reflect")
        magnitude = np.hypot(gradient_x, gradient_y)
        max_magnitude = float(np.max(magnitude))
        if max_magnitude > 0:
            edge_image = (magnitude / max_magnitude * 255.0).astype(np.uint8)
            features["edges"] = Image.fromarray(edge_image, mode="L")
    except Exception:
        features["edges"] = None

    FEATURE_CACHE[image.size] = features
    return features


def generate_variant_overlay(variant: str, size: Tuple[int, int]) -> Image.Image | None:
    """Generate a procedural overlay tailored to ``variant``."""

    width, height = size
    if width == 0 or height == 0:
        return None

    base_overlay = Image.new("RGBA", size, (0, 0, 0, 0))
    variant_key = variant.lower()
    features = _get_features_for_size(size)

    if variant_key == "zombie":
        noise = _normalize_noise(generate_perlin_noise(width, height, scale=8))
        blotches = np.zeros((height, width, 4), dtype=np.uint8)
        green_mask = noise > 0.58
        brown_mask = (noise > 0.42) & ~green_mask
        blotches[green_mask, :3] = (72, 120, 64)
        blotches[green_mask, 3] = (noise[green_mask] * 190).astype(np.uint8)
        blotches[brown_mask, :3] = (94, 72, 44)
        blotches[brown_mask, 3] = (noise[brown_mask] * 160).astype(np.uint8)
        overlay = Image.alpha_composite(base_overlay, Image.fromarray(blotches, "RGBA"))
        cracks = Image.new("RGBA", size, (0, 0, 0, 0))
        crack_draw = ImageDraw.Draw(cracks)
        crack_count = max(6, width // 10)
        for _ in range(crack_count):
            start_x = random.randint(0, width - 1)
            start_y = random.randint(0, height - 1)
            points = random_walk_lines(start_x, start_y, length=60 + height // 3, step_size=2)
            crack_draw.line(points, fill=(18, 22, 18, 220), width=1)
        cracks = cracks.filter(ImageFilter.GaussianBlur(radius=0.8))
        return Image.alpha_composite(overlay, cracks)

    if variant_key == "frozen":
        noise = _normalize_noise(generate_perlin_noise(width, height, scale=12))
        frost = np.zeros((height, width, 4), dtype=np.uint8)
        frost_mask = noise > 0.45
        frost[frost_mask, :3] = (180, 220, 255)
        frost[frost_mask, 3] = (noise[frost_mask] * 170).astype(np.uint8)
        frost_layer = Image.fromarray(frost, "RGBA").filter(ImageFilter.GaussianBlur(radius=1.5))
        overlay = Image.alpha_composite(base_overlay, frost_layer)
        crack_layer = Image.new("RGBA", size, (0, 0, 0, 0))
        crack_draw = ImageDraw.Draw(crack_layer)
        crack_count = max(5, width // 12)
        for _ in range(crack_count):
            start_x = random.randint(0, width - 1)
            start_y = random.randint(0, max(1, int(height * 0.6)))
            points = random_walk_lines(start_x, start_y, length=70 + height // 2, step_size=3)
            crack_draw.line(points, fill=(140, 190, 255, 230), width=2)
        crack_glow = crack_layer.filter(ImageFilter.GaussianBlur(radius=1.2))
        overlay = Image.alpha_composite(overlay, crack_glow)
        return Image.alpha_composite(overlay, crack_layer)

    if variant_key == "corrupted":
        noise = _normalize_noise(generate_perlin_noise(width, height, scale=10))
        blotches = np.zeros((height, width, 4), dtype=np.uint8)
        dark_mask = noise > 0.5
        blotches[dark_mask, :3] = (12, 12, 12)
        blotches[dark_mask, 3] = (noise[dark_mask] * 200).astype(np.uint8)
        overlay = Image.alpha_composite(base_overlay, Image.fromarray(blotches, "RGBA"))
        veins = Image.new("RGBA", size, (0, 0, 0, 0))
        vein_draw = ImageDraw.Draw(veins)
        vein_count = max(7, width // 9)
        for _ in range(vein_count):
            start_x = random.randint(0, width - 1)
            start_y = random.randint(0, height - 1)
            points = random_walk_lines(start_x, start_y, length=80 + height // 2, step_size=2)
            vein_draw.line(points, fill=(120, 0, 120, 220), width=2)
        glow = veins.filter(ImageFilter.GaussianBlur(radius=1.0))
        overlay = Image.alpha_composite(overlay, glow)
        return Image.alpha_composite(overlay, veins)

    if variant_key == "toxic":
        overlay = Image.alpha_composite(base_overlay, _create_radial_glow(size, (96, 220, 96), 0.8, 190))
        noise = _normalize_noise(generate_perlin_noise(width, height, scale=14))
        speckles = np.zeros((height, width, 4), dtype=np.uint8)
        speck_mask = noise > 0.65
        speckles[speck_mask, :3] = (70, 200, 64)
        speckles[speck_mask, 3] = (noise[speck_mask] * 140).astype(np.uint8)
        overlay = Image.alpha_composite(overlay, Image.fromarray(speckles, "RGBA"))
        joint_layer = Image.new("RGBA", size, (0, 0, 0, 0))
        joint_draw = ImageDraw.Draw(joint_layer)
        for x, y, radius in features.get("joints", []):
            joint_radius = max(radius, 4)
            joint_draw.ellipse(
                (x - joint_radius, y - joint_radius, x + joint_radius, y + joint_radius),
                fill=(150, 255, 150, 220),
            )
        joint_layer = joint_layer.filter(ImageFilter.GaussianBlur(radius=2.0))
        return Image.alpha_composite(overlay, joint_layer)

    if variant_key == "oceanic":
        gradient = np.zeros((height, width, 4), dtype=np.uint8)
        for y in range(height):
            t = y / max(1, height - 1)
            r = int(20 + 30 * (1 - t))
            g = int(90 + 90 * t)
            b = int(150 + 70 * (1 - t))
            alpha = int(80 + 110 * (1 - abs(0.5 - t) * 2))
            gradient[y, :, 0] = r
            gradient[y, :, 1] = g
            gradient[y, :, 2] = b
            gradient[y, :, 3] = alpha
        overlay = Image.alpha_composite(base_overlay, Image.fromarray(gradient, "RGBA"))
        algae_noise = _normalize_noise(generate_perlin_noise(width, height, scale=9))
        algae = np.zeros((height, width, 4), dtype=np.uint8)
        algae_mask = algae_noise > 0.6
        algae[algae_mask, :3] = (40, 140, 130)
        algae[algae_mask, 3] = (algae_noise[algae_mask] * 150).astype(np.uint8)
        overlay = Image.alpha_composite(overlay, Image.fromarray(algae, "RGBA"))
        bubbles = Image.new("RGBA", size, (0, 0, 0, 0))
        bubble_draw = ImageDraw.Draw(bubbles)
        min_distance = max(4, min(width, height) // 10)
        sample_count = max(12, width // 2)
        for x, y in generate_poisson_disc_samples(width, height, min_distance, num_samples=sample_count):
            radius = random.randint(max(2, min(width, height) // 30), max(4, min(width, height) // 12))
            bubble_draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=(210, 240, 255, 180),
            )
        bubbles = bubbles.filter(ImageFilter.GaussianBlur(radius=1.0))
        return Image.alpha_composite(overlay, bubbles)

    if variant_key == "nether":
        noise = _normalize_noise(generate_perlin_noise(width, height, scale=12))
        lava = np.zeros((height, width, 4), dtype=np.uint8)
        lava_mask = noise > 0.48
        lava[lava_mask, :3] = (200, 60, 20)
        lava[lava_mask, 3] = (noise[lava_mask] * 180).astype(np.uint8)
        overlay = Image.alpha_composite(base_overlay, Image.fromarray(lava, "RGBA"))
        crack_dark = Image.new("RGBA", size, (0, 0, 0, 0))
        crack_hot = Image.new("RGBA", size, (0, 0, 0, 0))
        draw_dark = ImageDraw.Draw(crack_dark)
        draw_hot = ImageDraw.Draw(crack_hot)
        crack_count = max(10, width // 8)
        for _ in range(crack_count):
            start_x = random.randint(0, width - 1)
            start_y = random.randint(0, height - 1)
            points = random_walk_lines(start_x, start_y, length=90 + height // 2, step_size=2)
            draw_dark.line(points, fill=(30, 0, 0, 240), width=3)
            draw_hot.line(points, fill=(230, 70, 0, 210), width=1)
        crack_hot = crack_hot.filter(ImageFilter.GaussianBlur(radius=1.5))
        overlay = Image.alpha_composite(overlay, crack_dark)
        return Image.alpha_composite(overlay, crack_hot)

    if variant_key == "celestial":
        overlay = Image.alpha_composite(base_overlay, _create_radial_glow(size, (255, 215, 140), 0.7, 150))
        halo_layer = Image.new("RGBA", size, (0, 0, 0, 0))
        halo_draw = ImageDraw.Draw(halo_layer)
        for x, y, radius in features.get("eyes", []):
            halo_radius = max(radius * 3, min(width, height) // 6)
            halo_draw.ellipse(
                (x - halo_radius, y - halo_radius, x + halo_radius, y + halo_radius),
                fill=(255, 214, 120, 140),
            )
        halo_layer = halo_layer.filter(ImageFilter.GaussianBlur(radius=2.5))
        stars = Image.new("RGBA", size, (0, 0, 0, 0))
        stars_draw = ImageDraw.Draw(stars)
        min_distance = max(3, min(width, height) // 12)
        sample_count = max(15, width // 2 + 10)
        for x, y in generate_poisson_disc_samples(width, height, min_distance, num_samples=sample_count):
            radius = random.randint(1, max(2, min(width, height) // 18))
            color = random.choice([(255, 255, 255, 240), (255, 220, 170, 220)])
            stars_draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
        stars_blur = stars.filter(ImageFilter.GaussianBlur(radius=0.8))
        overlay = Image.alpha_composite(overlay, halo_layer)
        overlay = Image.alpha_composite(overlay, stars_blur)
        return Image.alpha_composite(overlay, stars)

    return base_overlay


def generate_variant_mask(variant: str, size: Tuple[int, int]) -> Image.Image | None:
    """Generate a procedural emissive mask for ``variant``."""

    width, height = size
    if width == 0 or height == 0:
        return None

    mask = Image.new("RGBA", size, (0, 0, 0, 0))
    variant_key = variant.lower()
    features = _get_features_for_size(size)
    draw = ImageDraw.Draw(mask)

    if variant_key == "zombie":
        for x, y, radius in features.get("eyes", []):
            eye_radius = max(radius, 4)
            draw.ellipse(
                (x - eye_radius, y - eye_radius, x + eye_radius, y + eye_radius),
                fill=(90, 255, 120, 255),
            )
        return mask

    if variant_key == "frozen":
        for x, y, radius in features.get("eyes", []):
            eye_radius = max(radius, 4)
            draw.ellipse(
                (x - eye_radius, y - eye_radius, x + eye_radius, y + eye_radius),
                fill=(150, 220, 255, 255),
            )
        crack_layer = Image.new("RGBA", size, (0, 0, 0, 0))
        crack_draw = ImageDraw.Draw(crack_layer)
        crack_count = max(6, width // 10)
        for _ in range(crack_count):
            start_x = random.randint(0, width - 1)
            start_y = random.randint(0, max(1, int(height * 0.6)))
            points = random_walk_lines(start_x, start_y, length=80 + height // 2, step_size=3)
            crack_draw.line(points, fill=(200, 240, 255, 255), width=2)
        crack_layer = crack_layer.filter(ImageFilter.GaussianBlur(radius=1.0))
        return Image.alpha_composite(mask, crack_layer)

    if variant_key == "corrupted":
        for x, y, radius in features.get("eyes", []):
            eye_radius = max(radius, 4)
            draw.ellipse(
                (x - eye_radius, y - eye_radius, x + eye_radius, y + eye_radius),
                fill=(180, 40, 200, 255),
            )
        veins = Image.new("RGBA", size, (0, 0, 0, 0))
        vein_draw = ImageDraw.Draw(veins)
        vein_count = max(8, width // 8)
        for _ in range(vein_count):
            start_x = random.randint(0, width - 1)
            start_y = random.randint(0, height - 1)
            points = random_walk_lines(start_x, start_y, length=90 + height // 2, step_size=2)
            vein_draw.line(points, fill=(122, 0, 122, 255), width=2)
        glow = veins.filter(ImageFilter.GaussianBlur(radius=1.2))
        mask = Image.alpha_composite(mask, glow)
        return Image.alpha_composite(mask, veins)

    if variant_key == "toxic":
        outline_width = max(2, min(width, height) // 24)
        draw.rectangle(
            (outline_width, outline_width, width - outline_width, height - outline_width),
            outline=(120, 255, 120, 255),
            width=outline_width,
        )
        joint_layer = Image.new("RGBA", size, (0, 0, 0, 0))
        joint_draw = ImageDraw.Draw(joint_layer)
        for x, y, radius in features.get("joints", []):
            joint_radius = max(radius, 5)
            joint_draw.ellipse(
                (x - joint_radius, y - joint_radius, x + joint_radius, y + joint_radius),
                fill=(170, 255, 160, 255),
            )
        joint_layer = joint_layer.filter(ImageFilter.GaussianBlur(radius=1.5))
        return Image.alpha_composite(mask, joint_layer)

    if variant_key == "oceanic":
        for x, y, radius in features.get("eyes", []):
            eye_radius = max(radius, 4)
            draw.ellipse(
                (x - eye_radius, y - eye_radius, x + eye_radius, y + eye_radius),
                fill=(65, 146, 146, 255),
            )
        bubbles = Image.new("RGBA", size, (0, 0, 0, 0))
        bubble_draw = ImageDraw.Draw(bubbles)
        min_distance = max(5, min(width, height) // 9)
        sample_count = max(10, width // 2)
        for i, (x, y) in enumerate(
            generate_poisson_disc_samples(width, height, min_distance, num_samples=sample_count)
        ):
            max_radius = max(4, min(width, height) // 10)
            radius = random.randint(max_radius // 2, max_radius)
            color = (255, 255, 255, 255) if i % 3 == 0 else (200, 240, 250, 255)
            bubble_draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
        bubbles = bubbles.filter(ImageFilter.GaussianBlur(radius=1.2))
        return Image.alpha_composite(mask, bubbles)

    if variant_key == "nether":
        for x, y, radius in features.get("eyes", []):
            eye_radius = max(radius, 4)
            draw.ellipse(
                (x - eye_radius, y - eye_radius, x + eye_radius, y + eye_radius),
                fill=(255, 60, 30, 255),
            )
        cracks = Image.new("RGBA", size, (0, 0, 0, 0))
        crack_draw = ImageDraw.Draw(cracks)
        crack_count = max(10, width // 8)
        for _ in range(crack_count):
            start_x = random.randint(0, width - 1)
            start_y = random.randint(0, height - 1)
            points = random_walk_lines(start_x, start_y, length=90 + height // 2, step_size=2)
            crack_draw.line(points, fill=(255, 120, 80, 255), width=2)
        cracks = cracks.filter(ImageFilter.GaussianBlur(radius=1.0))
        return Image.alpha_composite(mask, cracks)

    if variant_key == "celestial":
        for x, y, radius in features.get("eyes", []):
            eye_radius = max(radius, 4)
            draw.ellipse(
                (x - eye_radius, y - eye_radius, x + eye_radius, y + eye_radius),
                fill=(255, 220, 140, 255),
            )
        halo = _create_radial_glow(size, (255, 240, 200), 0.75, 170)
        halo = halo.filter(ImageFilter.GaussianBlur(radius=2.5))
        stars = Image.new("RGBA", size, (0, 0, 0, 0))
        stars_draw = ImageDraw.Draw(stars)
        min_distance = max(3, min(width, height) // 14)
        sample_count = max(18, width // 2 + 8)
        for x, y in generate_poisson_disc_samples(width, height, min_distance, num_samples=sample_count):
            radius = random.randint(1, max(2, min(width, height) // 16))
            color = random.choice([(255, 255, 255, 255), (255, 230, 170, 255)])
            stars_draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
        stars = stars.filter(ImageFilter.GaussianBlur(radius=1.0))
        mask = Image.alpha_composite(mask, halo)
        return Image.alpha_composite(mask, stars)

    return mask


def ensure_directories(variants: Iterable[str]) -> None:
    """Ensure the expected output directory structure exists."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for variant in variants:
        (OUTPUT_DIR / variant).mkdir(parents=True, exist_ok=True)


def ensure_rgba(image: Image.Image) -> Image.Image:
    """Ensure the image is in RGBA mode."""
    return image.convert("RGBA") if image.mode != "RGBA" else image


def get_height_map(image: Image.Image) -> np.ndarray:
    """Convert image to grayscale and normalize to [0, 1] range."""
    grayscale = image.convert("L")
    return np.array(grayscale, dtype=np.float32) / 255.0


def load_overlay(variant: str, size: Tuple[int, int]) -> Image.Image | None:
    """Load and resize an overlay for the variant if available."""
    overlay_path = OVERLAY_DIR / f"{variant}.png"
    if not overlay_path.exists():
        return None

    overlay = Image.open(overlay_path).convert("RGBA")
    if overlay.size != size:
        overlay = overlay.resize(size, Image.LANCZOS)
    return overlay


def generate_emissive_map(
    size: Tuple[int, int],
    variant: str,
    base_output_path: Path,
    mask_image: Image.Image | None = None,
) -> None:
    """Generate an emissive map for ``variant`` using the provided mask."""

    mask_path = MASK_DIR / f"{variant}.png"
    mask = None

    if mask_image is not None:
        mask = mask_image.convert("L")
    elif mask_path.exists():
        try:
            with Image.open(mask_path) as opened_mask:
                mask = opened_mask.convert("L")
        except OSError as exc:
            print(f"⚠️  No se pudo abrir la máscara emisiva {mask_path}: {exc}")
            return
    else:
        return

    try:
        if mask.size != size:
            mask = mask.resize(size, Image.LANCZOS)

        emissive = Image.new("RGBA", size, (0, 0, 0, 255))
        mask_pixels = mask.load()
        emissive_pixels = emissive.load()
        width, height = size
        for y in range(height):
            for x in range(width):
                emissive_pixels[x, y] = (
                    (255, 255, 255, 255) if mask_pixels[x, y] > 0 else (0, 0, 0, 255)
                )

        emissive_path = base_output_path.with_name(f"{base_output_path.stem}_e.png")
        emissive.save(emissive_path)
        print(f"💡 Mapa emisivo generado: {emissive_path}")
    except OSError as exc:
        print(f"⚠️  No se pudo guardar el mapa emisivo para {mask_path}: {exc}")
    except Exception as exc:
        print(f"⚠️  Error al generar el mapa emisivo {mask_path}: {exc}")


def generate_normal_map(image: Image.Image, output_path: Path) -> None:
    """Generate a normal map based on ``image`` height information."""

    try:
        working_image = ensure_rgba(image)
        height_map = get_height_map(working_image)

        gradient_x = ndimage.sobel(height_map, axis=1, mode="reflect")
        gradient_y = ndimage.sobel(height_map, axis=0, mode="reflect")

        nz = np.ones_like(gradient_x, dtype=np.float32)
        magnitude = np.sqrt(gradient_x**2 + gradient_y**2 + nz**2)
        magnitude = np.where(magnitude == 0, 1.0, magnitude)

        normal_x = -gradient_x / magnitude
        normal_y = -gradient_y / magnitude
        normal_z = nz / magnitude

        normal_rgb = np.stack(
            [
                np.clip((normal_x * 0.5 + 0.5) * 255.0, 0, 255),
                np.clip((normal_y * 0.5 + 0.5) * 255.0, 0, 255),
                np.clip((normal_z * 0.5 + 0.5) * 255.0, 0, 255),
            ],
            axis=-1,
        ).astype(np.uint8)

        alpha_channel = np.array(working_image.getchannel("A"))
        normal_rgb[alpha_channel == 0] = 0

        normal_image = Image.fromarray(normal_rgb, mode="RGB")
        normal_path = output_path.with_name(f"{output_path.stem}_n.png")
        normal_image.save(normal_path)
        print(f"🌐 Mapa normal generado: {normal_path}")
    except OSError as exc:
        print(f"⚠️  No se pudo guardar el mapa normal para {output_path}: {exc}")
    except Exception as exc:
        print(f"⚠️  Error al generar el mapa normal para {output_path}: {exc}")


def apply_tint(image: Image.Image, config: Dict[str, float]) -> Image.Image:
    """Apply HSV shifts to an RGBA image according to ``config``.

    ``config`` should contain ``hue_shift``, ``saturation_shift`` and
    ``value_shift`` expressed as fractional adjustments.
    """

    working_image = ensure_rgba(image)
    rgba = np.array(working_image).astype(np.float32)
    rgb = rgba[..., :3] / 255.0
    alpha = rgba[..., 3:4] / 255.0

    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.max(rgb, axis=-1)
    minc = np.min(rgb, axis=-1)
    delta = maxc - minc

    hue = np.zeros_like(maxc)
    mask = delta != 0
    r_is_max = (maxc == r) & mask
    g_is_max = (maxc == g) & mask
    b_is_max = (maxc == b) & mask

    hue[r_is_max] = ((g - b)[r_is_max] / delta[r_is_max]) % 6
    hue[g_is_max] = ((b - r)[g_is_max] / delta[g_is_max]) + 2
    hue[b_is_max] = ((r - g)[b_is_max] / delta[b_is_max]) + 4
    hue = hue / 6.0  # Normalize to [0, 1)

    saturation = np.zeros_like(maxc)
    non_zero_max = maxc != 0
    saturation[non_zero_max] = delta[non_zero_max] / maxc[non_zero_max]

    value = maxc

    hue_shift = float(config.get("hue_shift", 0.0))
    saturation_shift = float(config.get("saturation_shift", 0.0))
    value_shift = float(config.get("value_shift", 0.0))

    hue = (hue + hue_shift) % 1.0
    saturation = np.clip(saturation + saturation_shift, 0.0, 1.0)
    value = np.clip(value + value_shift, 0.0, 1.0)

    # HSV back to RGB
    h6 = hue * 6.0
    i = np.floor(h6).astype(int) % 6
    f = h6 - np.floor(h6)

    p = value * (1 - saturation)
    q = value * (1 - saturation * f)
    t = value * (1 - saturation * (1 - f))

    rgb_out = np.zeros_like(rgb)

    i_eq_0 = i == 0
    i_eq_1 = i == 1
    i_eq_2 = i == 2
    i_eq_3 = i == 3
    i_eq_4 = i == 4
    i_eq_5 = i == 5

    rgb_out[..., 0] = np.select(
        [i_eq_0, i_eq_1, i_eq_2, i_eq_3, i_eq_4, i_eq_5],
        [value, q, p, p, t, value],
        default=value,
    )
    rgb_out[..., 1] = np.select(
        [i_eq_0, i_eq_1, i_eq_2, i_eq_3, i_eq_4, i_eq_5],
        [t, value, value, q, p, p],
        default=value,
    )
    rgb_out[..., 2] = np.select(
        [i_eq_0, i_eq_1, i_eq_2, i_eq_3, i_eq_4, i_eq_5],
        [p, p, t, value, value, q],
        default=value,
    )

    rgba_out = np.concatenate((rgb_out, alpha), axis=-1)
    rgba_out = np.clip(rgba_out * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(rgba_out, mode="RGBA")


def process_texture(image_path: Path) -> None:
    """Process a single texture across all configured variants."""
    try:
        base_image = Image.open(image_path).convert("RGBA")
    except OSError as exc:
        raise TextureProcessingError(f"No se pudo abrir la imagen {image_path}: {exc}") from exc

    base_name = image_path.stem
    detect_anatomical_features(base_image)
    base_name_lower = base_name.lower()
    for variant, config in VARIANTS.items():
        excluded_words = EXCLUSIONS.get(variant, [])
        if any(excluded_word in base_name_lower for excluded_word in excluded_words):
            print(f"⏭️  Saltando variante {variant} para {base_name} (exclusión)")
            continue
        tint_config = config.get("tint", {})
        overlay_opacity = config.get("overlay_opacity", 0.0)

        tinted = apply_tint(base_image, tint_config)
        overlay = generate_overlay_if_missing(variant, tinted.size)
        if overlay is not None and overlay_opacity > 0:
            overlay_layer = overlay.copy()
            if overlay_opacity < 1.0:
                alpha_channel = overlay_layer.getchannel("A")
                alpha_channel = alpha_channel.point(
                    lambda value: int(max(0, min(255, value * overlay_opacity)))
                )
                overlay_layer.putalpha(alpha_channel)
            tinted = Image.alpha_composite(tinted, overlay_layer)

        mask_image = generate_mask_if_missing(variant, tinted.size)

        output_path = OUTPUT_DIR / variant / f"{base_name}_{variant}.png"
        tinted.save(output_path)
        generate_normal_map(tinted, output_path)
        generate_emissive_map(tinted.size, variant, output_path, mask_image=mask_image)
        print(f"✓ {image_path.name} → {output_path}")


def iter_input_images(directory: Path) -> Iterable[Path]:
    """Iterate over PNG images inside ``directory``."""
    if not directory.exists():
        return []
    return sorted(path for path in directory.glob("*.png") if path.is_file())


def main(argv: Iterable[str] | None = None) -> int:
    """Entry point for the texture generator."""
    ensure_directories(VARIANTS.keys())

    images = list(iter_input_images(INPUT_DIR))
    if not images:
        print("No se encontraron imágenes en la carpeta 'input/'.")
        return 0

    for image_path in images:
        try:
            process_texture(image_path)
        except TextureProcessingError as exc:
            print(f"Error: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
