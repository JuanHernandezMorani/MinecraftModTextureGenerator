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

from PIL import Image
import numpy as np

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


class TextureProcessingError(RuntimeError):
    """Raised when the texture processing pipeline encounters an error."""


def ensure_directories(variants: Iterable[str]) -> None:
    """Ensure the expected output directory structure exists."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for variant in variants:
        (OUTPUT_DIR / variant).mkdir(parents=True, exist_ok=True)


def load_overlay(variant: str, size: Tuple[int, int]) -> Image.Image | None:
    """Load and resize an overlay for the variant if available."""
    overlay_path = OVERLAY_DIR / f"{variant}.png"
    if not overlay_path.exists():
        return None

    overlay = Image.open(overlay_path).convert("RGBA")
    if overlay.size != size:
        overlay = overlay.resize(size, Image.LANCZOS)
    return overlay


def adjust_overlay_opacity(overlay: Image.Image, opacity: float) -> Image.Image:
    """Return a copy of the overlay with adjusted opacity."""
    opacity = float(opacity)
    opacity = max(0.0, min(1.0, opacity))
    overlay = overlay.copy()
    alpha = overlay.getchannel("A")
    lut = [int(round(value * opacity)) for value in range(256)]
    overlay.putalpha(alpha.point(lut))
    return overlay


def apply_tint(image: Image.Image, config: Dict[str, float]) -> Image.Image:
    """Apply HSV shifts to an RGBA image according to ``config``.

    ``config`` should contain ``hue_shift``, ``saturation_shift`` and
    ``value_shift`` expressed as fractional adjustments.
    """

    if image.mode != "RGBA":
        image = image.convert("RGBA")

    rgba = np.array(image).astype(np.float32)
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
    base_name_lower = base_name.lower()
    for variant, config in VARIANTS.items():
        excluded_words = EXCLUSIONS.get(variant, [])
        if any(excluded_word in base_name_lower for excluded_word in excluded_words):
            print(f"⏭️  Saltando variante {variant} para {base_name} (exclusión)")
            continue
        tint_config = config.get("tint", {})
        overlay_opacity = config.get("overlay_opacity", 0.0)

        tinted = apply_tint(base_image, tint_config)
        overlay = load_overlay(variant, tinted.size)
        if overlay is not None and overlay_opacity > 0:
            overlay = adjust_overlay_opacity(overlay, overlay_opacity)
            tinted = Image.alpha_composite(tinted, overlay)

        output_path = OUTPUT_DIR / variant / f"{base_name}_{variant}.png"
        tinted.save(output_path)
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
