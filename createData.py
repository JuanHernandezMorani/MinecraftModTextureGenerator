"""Conservative dataset creation pipeline for Minecraft texture features.

This refactor applies semantic filtering and improved heuristics to avoid
false positives when detecting texture features such as eyes, glow regions
and cracks. The script generates ``data.csv`` alongside an audit report and
annotated previews under ``validation_report``.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy import ndimage

# ---------------------------------------------------------------------------
# Semantic classification keywords
# ---------------------------------------------------------------------------

MOB_KEYWORDS: Tuple[str, ...] = (
    "zombie",
    "skeleton",
    "dragon",
    "wither",
    "warden",
    "creeper",
    "villager",
    "golem",
    "pig",
    "cow",
    "sheep",
    "wolf",
    "fox",
    "bee",
    "enderman",
    "blaze",
    "spider",
    "witch",
    "guardian",
    "drowned",
    "hoglin",
    "ghast",
    "shulker",
    "camel",
    "sniffer",
    "llama",
    "panda",
    "strider",
    "warden",
    "frog",
    "bat",
)

NON_MOB_KEYWORDS: Tuple[str, ...] = (
    "fern",
    "flower",
    "stick",
    "banner",
    "icon",
    "pack",
    "button",
    "background",
    "particle",
    "cloud",
    "item",
    "sword",
    "shield",
    "pickaxe",
    "helmet",
    "chestplate",
    "log",
    "planks",
    "brick",
    "ore",
    "gem",
    "ingot",
    "tool",
    "block",
    "door",
    "bed",
    "bucket",
    "map",
    "book",
    "ui",
)

MAGICAL_KEYWORDS: Tuple[str, ...] = (
    "totem",
    "magic",
    "enchanted",
    "orb",
    "crystal",
    "glow",
    "shine",
    "spectral",
    "heart",
)

CRACK_SUSCEPTIBLE_KEYWORDS: Tuple[str, ...] = (
    "golem",
    "warden",
    "wither",
    "dragon",
    "stone",
    "brick",
    "ancient",
    "statue",
    "shell",
    "egg",
)

VARIANT_SUFFIXES: Tuple[str, ...] = ("_e", "_eyes", "_heart", "_spots", "_crackiness")

MAX_DETECTIONS_PER_TYPE = 5

# ---------------------------------------------------------------------------
# Helper dataclasses and utilities
# ---------------------------------------------------------------------------


@dataclass
class TextureInfo:
    path: Path
    is_mob: bool
    is_non_mob: bool
    is_magical: bool
    is_crack_susceptible: bool
    stem: str
    variant_suffix: Optional[str]


@dataclass
class Detection:
    label: str
    x: int
    y: int
    width: int
    height: int
    area: int
    area_ratio: float
    aspect_ratio: float
    detection_confidence: float
    extra: Dict[str, float]

    def as_record(self, texture: TextureInfo) -> Dict[str, float]:
        record = {
            "file_name": texture.path.name,
            "file_path": str(texture.path),
            "x": float(self.x),
            "y": float(self.y),
            "width": float(self.width),
            "height": float(self.height),
            "label": self.label,
            "area_ratio": float(self.area_ratio),
            "aspect_ratio": float(self.aspect_ratio),
            "detection_confidence": float(np.clip(self.detection_confidence, 0.0, 1.0)),
            "split": "train",
        }
        for key, value in self.extra.items():
            record[key] = value
        return record


# ---------------------------------------------------------------------------
# Semantic helpers
# ---------------------------------------------------------------------------


def _match_any(stem: str, keywords: Sequence[str]) -> bool:
    return any(keyword in stem for keyword in keywords)


def classify_texture_name(file_path: Path) -> TextureInfo:
    stem = file_path.stem.lower()
    is_variant = False
    variant_suffix: Optional[str] = None
    for suffix in VARIANT_SUFFIXES:
        if stem.endswith(suffix):
            is_variant = True
            variant_suffix = suffix
            stem = stem[: -len(suffix)]
            break

    lower_name = file_path.name.lower()
    is_non_mob = _match_any(stem, NON_MOB_KEYWORDS) or _match_any(lower_name, NON_MOB_KEYWORDS)
    is_mob = _match_any(stem, MOB_KEYWORDS) or _match_any(lower_name, MOB_KEYWORDS)

    # Strong non-mob keywords override mob detection.
    if is_non_mob and not is_mob:
        is_mob = False
    elif is_mob and not is_non_mob:
        is_non_mob = False

    is_magical = _match_any(lower_name, MAGICAL_KEYWORDS) or (variant_suffix in {"_e", "_heart"})
    is_crack_susceptible = _match_any(lower_name, CRACK_SUSCEPTIBLE_KEYWORDS)

    return TextureInfo(
        path=file_path,
        is_mob=is_mob,
        is_non_mob=is_non_mob,
        is_magical=is_magical,
        is_crack_susceptible=is_crack_susceptible,
        stem=stem,
        variant_suffix=variant_suffix,
    )


# ---------------------------------------------------------------------------
# Image loading helpers
# ---------------------------------------------------------------------------


def load_texture_rgba(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with Image.open(path) as img:
        rgba = img.convert("RGBA")
    arr = np.asarray(rgba).astype(np.float32)
    rgb = arr[..., :3]
    alpha = arr[..., 3:] / 255.0
    rgb = rgb * alpha + (1.0 - alpha) * 255.0  # place transparent pixels on white
    brightness = np.dot(rgb, np.array([0.299, 0.587, 0.114], dtype=np.float32))
    return brightness, alpha[..., 0]


def compute_gradient_map(brightness: np.ndarray) -> np.ndarray:
    grad_x = ndimage.sobel(brightness, axis=1, mode="reflect")
    grad_y = ndimage.sobel(brightness, axis=0, mode="reflect")
    gradient = np.hypot(grad_x, grad_y)
    return gradient


def region_stats(mask: np.ndarray) -> Tuple[int, int, int, int, int]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return 0, 0, 0, 0, 0
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    width = int(x1 - x0 + 1)
    height = int(y1 - y0 + 1)
    area = int(mask.sum())
    return int(x0), int(y0), width, height, area


def limit_detections(detections: List[Detection]) -> List[Detection]:
    detections_sorted = sorted(detections, key=lambda det: det.detection_confidence, reverse=True)
    return detections_sorted[:MAX_DETECTIONS_PER_TYPE]


# ---------------------------------------------------------------------------
# Eye detection
# ---------------------------------------------------------------------------


def detect_eyes_improved(
    brightness: np.ndarray,
    gradient: np.ndarray,
    texture: TextureInfo,
) -> List[Detection]:
    if not texture.is_mob:
        return []

    h, w = brightness.shape
    if h * w == 0:
        return []

    top_limit = max(1, int(0.4 * h))
    top_region = brightness[:top_limit]
    top_grad = gradient[:top_limit]
    grad_threshold = top_grad.mean() + top_grad.std()
    if grad_threshold <= 0:
        return []

    candidate_mask = top_grad > grad_threshold
    candidate_mask = ndimage.binary_opening(candidate_mask, structure=np.ones((2, 2), dtype=bool))

    labeled, num = ndimage.label(candidate_mask)
    detections: List[Detection] = []
    total_area = float(h * w)
    top_mean = float(top_region.mean())
    top_std = float(top_region.std() + 1e-6)

    for label_index in range(1, num + 1):
        mask = labeled == label_index
        x0, y0, width, height, area_pixels = region_stats(mask)
        if area_pixels == 0:
            continue

        y_global = y0
        area_ratio = area_pixels / total_area
        if area_ratio < 0.001 or area_ratio > 0.02:
            continue

        aspect_ratio = width / height if height > 0 else 0.0
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            continue

        sub_region = top_region[y0:y0 + height, x0:x0 + width]
        region_mean = float(sub_region.mean())
        region_std = float(sub_region.std())
        if region_std < 3:  # extremely flat areas are unlikely to be eyes
            continue

        contrast_score = min(abs(region_mean - top_mean) / (top_std * 2.5), 1.0)
        vertical_center = (y_global + height / 2.0) / max(1.0, top_limit)
        vertical_score = float(np.clip(1.0 - vertical_center, 0.0, 1.0))
        size_score = math.exp(-((area_ratio - 0.006) ** 2) / (2 * (0.004 ** 2)))

        confidence = 0.20
        confidence += 0.30 * (1.0 if texture.is_mob else 0.0)
        confidence += 0.25 * vertical_score
        confidence += 0.15 * size_score
        confidence += 0.10 * contrast_score

        detections.append(
            Detection(
                label="auto_eyes",
                x=x0,
                y=y_global,
                width=width,
                height=height,
                area=area_pixels,
                area_ratio=area_ratio,
                aspect_ratio=aspect_ratio,
                detection_confidence=confidence,
                extra={
                    "brightness_mean": region_mean,
                    "brightness_std": region_std,
                },
            )
        )

    if not detections:
        return []

    # Symmetry bonus: find mirrored pairs with similar heights.
    centers = np.array([(det.x + det.width / 2.0, det.y + det.height / 2.0) for det in detections])
    heights = np.array([det.height for det in detections])

    for i, det_i in enumerate(detections):
        cx_i, cy_i = centers[i]
        for j in range(i + 1, len(detections)):
            det_j = detections[j]
            cx_j, cy_j = centers[j]
            vertical_diff = abs(cy_i - cy_j) / max(heights[i], heights[j], 1)
            if vertical_diff > 0.15:
                continue
            mirrored_x = w - cx_j
            symmetry_error = abs(cx_i - mirrored_x) / max(w, 1)
            if symmetry_error > 0.12:
                continue
            size_ratio = detections[i].area / max(detections[j].area, 1)
            if size_ratio < 0.5 or size_ratio > 2.0:
                continue
            detections[i].detection_confidence = min(detections[i].detection_confidence + 0.15, 0.99)
            detections[j].detection_confidence = min(detections[j].detection_confidence + 0.15, 0.99)

    return limit_detections(detections)


# ---------------------------------------------------------------------------
# Glow detection
# ---------------------------------------------------------------------------


def detect_glow_improved(
    brightness: np.ndarray,
    gradient: np.ndarray,
    texture: TextureInfo,
) -> List[Detection]:
    if not (texture.is_mob or texture.is_magical):
        return []

    h, w = brightness.shape
    if h * w == 0:
        return []

    image_mean = float(brightness.mean())
    if image_mean <= 0:
        return []

    glow_threshold = image_mean * 1.5
    candidate_mask = brightness > glow_threshold
    candidate_mask = ndimage.binary_opening(candidate_mask, structure=np.ones((2, 2), dtype=bool))
    candidate_mask = ndimage.binary_closing(candidate_mask, structure=np.ones((3, 3), dtype=bool))

    labeled, num = ndimage.label(candidate_mask)
    total_area = float(h * w)
    detections: List[Detection] = []

    for label_index in range(1, num + 1):
        mask = labeled == label_index
        x0, y0, width, height, area_pixels = region_stats(mask)
        if area_pixels == 0:
            continue

        area_ratio = area_pixels / total_area
        if area_ratio <= 0 or area_ratio > 0.20:
            continue

        sub_brightness = brightness[y0:y0 + height, x0:x0 + width]
        region_mean = float(sub_brightness.mean())
        region_std = float(sub_brightness.std())
        brightness_ratio = region_mean / image_mean
        if brightness_ratio < 1.5:
            continue

        gradient_region = gradient[y0:y0 + height, x0:x0 + width]
        gradient_mean = float(gradient_region.mean())
        gradient_std = float(gradient_region.std())
        if gradient_mean <= 1:
            continue
        smoothness = np.clip(1.0 - (gradient_std / (gradient_mean * 2.5)), 0.0, 1.0)

        area_score = math.exp(-((area_ratio - 0.02) ** 2) / (2 * (0.015 ** 2)))
        brightness_score = np.clip((brightness_ratio - 1.5) / 1.0, 0.0, 1.0)

        confidence = 0.30
        confidence += 0.30 * brightness_score
        confidence += 0.20 * area_score
        confidence += 0.10 * smoothness
        confidence += 0.10 * (1.0 if texture.is_mob else 0.7)

        detections.append(
            Detection(
                label="auto_glow",
                x=x0,
                y=y0,
                width=width,
                height=height,
                area=area_pixels,
                area_ratio=area_ratio,
                aspect_ratio=width / height if height > 0 else 0.0,
                detection_confidence=confidence,
                extra={
                    "brightness_mean": region_mean,
                    "brightness_std": region_std,
                    "brightness_ratio": brightness_ratio,
                    "gradient_mean": gradient_mean,
                    "gradient_std": gradient_std,
                },
            )
        )

    return limit_detections(detections)


# ---------------------------------------------------------------------------
# Crack detection
# ---------------------------------------------------------------------------


def detect_cracks_improved(
    brightness: np.ndarray,
    gradient: np.ndarray,
    texture: TextureInfo,
) -> List[Detection]:
    if not (texture.is_mob or texture.is_crack_susceptible):
        return []

    h, w = brightness.shape
    if h * w == 0:
        return []

    image_mean = float(brightness.mean())
    crack_threshold = image_mean * 0.75
    candidate_mask = brightness < crack_threshold
    candidate_mask = ndimage.binary_opening(candidate_mask, structure=np.ones((2, 2), dtype=bool))

    labeled, num = ndimage.label(candidate_mask)
    total_area = float(h * w)
    detections: List[Detection] = []

    for label_index in range(1, num + 1):
        mask = labeled == label_index
        x0, y0, width, height, area_pixels = region_stats(mask)
        if area_pixels == 0:
            continue

        area_ratio = area_pixels / total_area
        if area_ratio <= 0 or area_ratio > 0.05:
            continue

        aspect_ratio = width / height if height > 0 else 0.0
        if not (aspect_ratio > 1.5 or aspect_ratio < 0.66):
            continue

        sub_brightness = brightness[y0:y0 + height, x0:x0 + width]
        region_mean = float(sub_brightness.mean())
        darkness_score = np.clip((image_mean - region_mean) / max(image_mean, 1.0), 0.0, 1.0)
        if darkness_score < 0.2:
            continue

        gradient_region = gradient[y0:y0 + height, x0:x0 + width]
        gradient_mean = float(gradient_region.mean())
        gradient_std = float(gradient_region.std())

        area_score = math.exp(-((area_ratio - 0.01) ** 2) / (2 * (0.01 ** 2)))
        aspect_score = np.clip(min(aspect_ratio / 4.0, 4.0 / max(aspect_ratio, 1e-6)), 0.0, 1.0)

        confidence = 0.30
        confidence += 0.30 * darkness_score
        confidence += 0.15 * area_score
        confidence += 0.15 * aspect_score
        confidence += 0.10 * (1.0 if texture.is_mob else 0.7)

        detections.append(
            Detection(
                label="auto_cracks",
                x=x0,
                y=y0,
                width=width,
                height=height,
                area=area_pixels,
                area_ratio=area_ratio,
                aspect_ratio=aspect_ratio,
                detection_confidence=confidence,
                extra={
                    "brightness_mean": region_mean,
                    "brightness_std": float(sub_brightness.std()),
                    "gradient_mean": gradient_mean,
                    "gradient_std": gradient_std,
                },
            )
        )

    return limit_detections(detections)


# ---------------------------------------------------------------------------
# Dataset creation helpers
# ---------------------------------------------------------------------------


def discover_textures(root: Path, include_input: bool) -> List[TextureInfo]:
    directories: List[Path] = []
    if root.exists() and root.is_dir():
        directories.append(root)
    if include_input:
        alt = Path("input")
        if alt.exists():
            directories.append(alt)

    textures: List[TextureInfo] = []
    seen: set[Path] = set()
    for directory in directories:
        for path in directory.rglob("*.png"):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            textures.append(classify_texture_name(resolved))
    textures.sort(key=lambda info: info.path.name)
    return textures


def analyse_texture(texture: TextureInfo) -> List[Detection]:
    brightness, alpha = load_texture_rgba(texture.path)
    brightness = brightness * alpha + (1.0 - alpha) * brightness.mean()
    gradient = compute_gradient_map(brightness)

    detections: List[Detection] = []
    detections.extend(detect_eyes_improved(brightness, gradient, texture))
    detections.extend(detect_glow_improved(brightness, gradient, texture))
    detections.extend(detect_cracks_improved(brightness, gradient, texture))

    if texture.is_non_mob:
        # Non mob textures should not keep auto detections.
        detections = [det for det in detections if not det.label.startswith("auto_")]

    return detections


# ---------------------------------------------------------------------------
# Validation utilities
# ---------------------------------------------------------------------------


def ensure_validation_directory() -> Path:
    output_dir = Path("validation_report")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def draw_detections(texture: TextureInfo, detections: Sequence[Detection]) -> None:
    if not detections:
        return
    output_dir = ensure_validation_directory()
    with Image.open(texture.path).convert("RGBA") as img:
        draw = ImageDraw.Draw(img)
        color_map = {
            "auto_eyes": (255, 0, 0, 255),
            "auto_glow": (255, 215, 0, 255),
            "auto_cracks": (0, 255, 255, 255),
            "base_texture": (0, 255, 0, 255),
        }
        for det in detections:
            color = color_map.get(det.label, (255, 255, 255, 255))
            x1, y1 = det.x, det.y
            x2, y2 = det.x + det.width, det.y + det.height
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            text = f"{det.label} {det.detection_confidence:.2f}"
            text_pos = (x1 + 2, y1 + 2)
            draw.text(text_pos, text, fill=color)
        output_path = output_dir / texture.path.name
        img.save(output_path)


# ---------------------------------------------------------------------------
# Auditing utilities
# ---------------------------------------------------------------------------


def audit_existing_dataset(csv_path: Path) -> None:
    if not csv_path.exists():
        print(f"⚠️  No se encontró {csv_path}, se omite auditoría.")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print("⚠️  data.csv está vacío, auditoría sin resultados.")
        csv_path.parent.joinpath("dataset_audit_report.csv").write_text("file_name,issue,details\n")
        return

    issues: List[Dict[str, str]] = []
    for file_name, group in df.groupby("file_name"):
        if len(group) > 10:
            issues.append(
                {
                    "file_name": file_name,
                    "issue": "over_detection",
                    "details": f"{len(group)} detecciones",
                }
            )
        reference_path = group.get("file_path")
        if reference_path is not None and not reference_path.isnull().all():
            sample_path = Path(str(reference_path.iloc[0]))
        else:
            sample_path = Path(file_name)
        info = classify_texture_name(sample_path)
        for _, row in group.iterrows():
            label = str(row.get("label", ""))
            if not label.startswith("auto_"):
                continue
            if label == "auto_glow" and info.is_magical:
                continue
            if info.is_mob:
                continue
            issues.append(
                {
                    "file_name": file_name,
                    "issue": "auto_on_non_mob",
                    "details": label,
                    }
                )

    report_path = csv_path.parent.joinpath("dataset_audit_report.csv")
    pd.DataFrame(issues).to_csv(report_path, index=False)
    print(f"📝 Auditoría guardada en {report_path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def create_dataset(root: Path, include_input: bool, output_csv: Path) -> None:
    textures = discover_textures(root, include_input)
    if not textures:
        print("❌ No se encontraron texturas para procesar.")
        return

    all_records: List[Dict[str, float]] = []
    for texture in textures:
        detections = analyse_texture(texture)
        if not detections:
            continue
        draw_detections(texture, detections)
        for det in detections:
            all_records.append(det.as_record(texture))

    if not all_records:
        print("⚠️  No se generaron detecciones tras aplicar filtros conservadores.")
        return

    df = pd.DataFrame(all_records)
    df.sort_values(["file_name", "label", "detection_confidence"], ascending=[True, True, False], inplace=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"✅ Dataset guardado en {output_csv} ({len(df)} detecciones)")

    audit_existing_dataset(output_csv)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generar dataset conservador de texturas")
    parser.add_argument("--root", default="toLearn", help="Directorio raíz con texturas de entrenamiento")
    parser.add_argument("--no-input", action="store_true", help="Ignorar el directorio input/")
    parser.add_argument("--output", default="data.csv", help="Ruta de salida para el CSV")
    args = parser.parse_args()

    root = Path(args.root)
    include_input = not args.no_input
    output_csv = Path(args.output)

    create_dataset(root, include_input, output_csv)


if __name__ == "__main__":
    main()
