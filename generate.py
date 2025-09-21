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
from scipy import ndimage

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

OVERLAY_CACHE = {}
MASK_CACHE = {}

def generate_overlay_if_missing(variant: str, size: Tuple[int,int]):
    if variant in OVERLAY_CACHE:
        return OVERLAY_CACHE[variant]
    # ... lógica de generación
    OVERLAY_CACHE[variant] = None #overlay

    return None #overlay

class TextureProcessingError(RuntimeError):
    """Raised when the texture processing pipeline encounters an error."""

def detect_anatomical_features(image: Image.Image) -> Dict:
    # Fallback para cuando la detección automática no sea posible
    width, height = image.size
    return {
        'eyes': [(width//3, height//4, 5), (2*width//3, height//4, 5)],  # Posiciones por defecto
        'joints': [],
        'edges': None
    }

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


def generate_emissive_map(size: Tuple[int, int], variant: str, base_output_path: Path) -> None:
    """Generate an emissive map for ``variant`` based on the configured mask."""

    mask_path = MASK_DIR / f"{variant}.png"
    if not mask_path.exists():
        return

    try:
        with Image.open(mask_path) as mask_image:
            mask = mask_image.convert("L")
    except OSError as exc:
        print(f"⚠️  No se pudo abrir la máscara emisiva {mask_path}: {exc}")
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
            tinted = Image.blend(tinted, overlay, overlay_opacity)

        output_path = OUTPUT_DIR / variant / f"{base_name}_{variant}.png"
        tinted.save(output_path)
        generate_normal_map(tinted, output_path)
        generate_emissive_map(tinted.size, variant, output_path)
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
