"""Script to generate a training dataset from Minecraft texture variants.

This script scans the `toLearn/` and `input/` directories for base
textures (e.g., ``mob.png``) and their variant counterparts
(``mob_e.png``, ``mob_eyes.png``, ``mob_heart.png``,
``mob_spots.png``, ``mob_crackiness.png``). For every detected
base/variant pair, it computes the precise bounding boxes of the
differences, extracts several statistical and geometric features, and
stores them in a CSV dataset split into train/test subsets.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageChops


# Directories that may contain textures to analyse.
TEXTURE_DIRECTORIES: Sequence[str] = ("toLearn", "input")

# Mapping between known suffixes and the labels to use in the dataset.
VARIANT_SUFFIXES: Dict[str, str] = {
    "_e": "e",
    "_eyes": "eyes",
    "_heart": "heart",
    "_spots": "spots",
    "_crackiness": "crackiness",
}


@dataclass
class VariantInfo:
    """Information required to process a variant texture."""

    base_path: Path
    variant_path: Path
    label: str


def collect_variants(directory: Path) -> Iterable[VariantInfo]:
    """Yield :class:`VariantInfo` entries detected in ``directory``.

    Parameters
    ----------
    directory:
        Directory to analyse.
    """

    if not directory.exists() or not directory.is_dir():
        return []

    base_files: Dict[str, Path] = {}
    variant_files: List[Tuple[str, str, Path]] = []

    for file_path in directory.glob("*.png"):
        stem = file_path.stem
        matched_suffix = None
        for suffix in VARIANT_SUFFIXES:
            if stem.endswith(suffix):
                base_stem = stem[: -len(suffix)]
                if base_stem:
                    variant_files.append((base_stem, suffix, file_path))
                matched_suffix = suffix
                break
        if matched_suffix is None:
            base_files[stem] = file_path

    variants: List[VariantInfo] = []
    for base_stem, suffix, variant_path in variant_files:
        base_path = base_files.get(base_stem)
        if base_path is None:
            continue
        label = VARIANT_SUFFIXES[suffix]
        variants.append(VariantInfo(base_path=base_path, variant_path=variant_path, label=label))

    return variants


def compute_regions(mask: np.ndarray) -> List[Tuple[List[Tuple[int, int]], Tuple[int, int, int, int]]]:
    """Return connected regions and their bounding boxes from ``mask``.

    Parameters
    ----------
    mask:
        Boolean array representing the binarised difference between two
        textures.
    """

    visited = np.zeros_like(mask, dtype=bool)
    regions: List[Tuple[List[Tuple[int, int]], Tuple[int, int, int, int]]] = []
    rows, cols = mask.shape
    neighbours = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )

    for r in range(rows):
        for c in range(cols):
            if not mask[r, c] or visited[r, c]:
                continue

            stack = [(r, c)]
            coords: List[Tuple[int, int]] = []
            min_r = max_r = r
            min_c = max_c = c

            while stack:
                cr, cc = stack.pop()
                if visited[cr, cc] or not mask[cr, cc]:
                    continue
                visited[cr, cc] = True
                coords.append((cr, cc))
                min_r = min(min_r, cr)
                max_r = max(max_r, cr)
                min_c = min(min_c, cc)
                max_c = max(max_c, cc)

                for dr, dc in neighbours:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc] and mask[nr, nc]:
                        stack.append((nr, nc))

            regions.append((coords, (min_r, min_c, max_r, max_c)))

    return regions


def extract_features(variant_info: VariantInfo) -> List[Dict[str, float]]:
    """Extract dataset-ready features for the provided variant texture."""

    with Image.open(variant_info.base_path) as base_img_raw, Image.open(
        variant_info.variant_path
    ) as variant_img_raw:
        base_img = base_img_raw.convert("RGBA")
        variant_img = variant_img_raw.convert("RGBA")

        if base_img.size != variant_img.size:
            return []

        diff = ImageChops.difference(base_img, variant_img)
        diff_gray = diff.convert("L")
        diff_array = np.array(diff_gray, dtype=np.uint8)

        max_diff = int(diff_array.max())
        if max_diff == 0:
            print(f"🔍 Analizando {variant_info.variant_path.name}: 0 regiones detectadas")
            return []

        threshold = max_diff * 0.3
        mask = diff_array > threshold

        if not mask.any():
            print(f"🔍 Analizando {variant_info.variant_path.name}: 0 regiones detectadas")
            return []

        regions = compute_regions(mask)

        print(
            f"🔍 Analizando {variant_info.variant_path.name}: {len(regions)} regiones detectadas"
        )

        variant_rgb = variant_img.convert("RGB")
        rgb_array = np.array(variant_rgb, dtype=np.uint8)
        gray_array = np.array(variant_rgb.convert("L"), dtype=float)
        gy, gx = np.gradient(gray_array)
        gradient_magnitude = np.sqrt(gx**2 + gy**2)

        img_width, img_height = variant_img.size
        total_area = float(img_width * img_height)

        records: List[Dict[str, float]] = []

        for coords, (min_r, min_c, max_r, max_c) in regions:
            if not coords:
                continue

            rows_coords = np.array([r for r, _ in coords])
            cols_coords = np.array([c for _, c in coords])

            r_values = rgb_array[rows_coords, cols_coords, 0].astype(float)
            g_values = rgb_array[rows_coords, cols_coords, 1].astype(float)
            b_values = rgb_array[rows_coords, cols_coords, 2].astype(float)

            brightness_values = 0.299 * r_values + 0.587 * g_values + 0.114 * b_values
            gradient_values = gradient_magnitude[rows_coords, cols_coords]

            width = max_c - min_c + 1
            height = max_r - min_r + 1
            area = float(len(coords))

            record = {
                "file_name": variant_info.variant_path.name,
                "x": float(min_c),
                "y": float(min_r),
                "width": float(width),
                "height": float(height),
                "label": variant_info.label,
                "r_mean": float(r_values.mean()),
                "g_mean": float(g_values.mean()),
                "b_mean": float(b_values.mean()),
                "r_std": float(r_values.std(ddof=0)),
                "g_std": float(g_values.std(ddof=0)),
                "b_std": float(b_values.std(ddof=0)),
                "brightness_mean": float(brightness_values.mean()),
                "brightness_std": float(brightness_values.std(ddof=0)),
                "gradient_magnitude": float(gradient_values.mean()),
                "aspect_ratio": float(width / height) if height else 0.0,
                "area_ratio": float(area / total_area) if total_area else 0.0,
            }

            records.append(record)

        return records


def stratified_split(records: List[Dict[str, float]], seed: int = 42) -> pd.DataFrame:
    """Return a DataFrame with stratified train/test splits."""

    if not records:
        columns = [
            "file_name",
            "x",
            "y",
            "width",
            "height",
            "label",
            "r_mean",
            "g_mean",
            "b_mean",
            "r_std",
            "g_std",
            "b_std",
            "brightness_mean",
            "brightness_std",
            "gradient_magnitude",
            "aspect_ratio",
            "area_ratio",
            "split",
        ]
        return pd.DataFrame(columns=columns)

    grouped: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    for record in records:
        grouped[str(record["label"])].append(record)

    rng = random.Random(seed)
    final_records: List[Dict[str, float]] = []

    for label, items in grouped.items():
        items_copy = list(items)
        rng.shuffle(items_copy)

        if len(items_copy) <= 1:
            train_items = items_copy
            test_items: List[Dict[str, float]] = []
        else:
            test_count = max(1, int(round(len(items_copy) * 0.2)))
            if test_count >= len(items_copy):
                test_count = len(items_copy) - 1

            test_items = items_copy[:test_count]
            train_items = items_copy[test_count:]

        for item in train_items:
            row = dict(item)
            row["split"] = "train"
            final_records.append(row)

        for item in test_items:
            row = dict(item)
            row["split"] = "test"
            final_records.append(row)

    df = pd.DataFrame(final_records)
    columns_order = [
        "file_name",
        "x",
        "y",
        "width",
        "height",
        "label",
        "r_mean",
        "g_mean",
        "b_mean",
        "r_std",
        "g_std",
        "b_std",
        "brightness_mean",
        "brightness_std",
        "gradient_magnitude",
        "aspect_ratio",
        "area_ratio",
        "split",
    ]

    df = df.reindex(columns=columns_order)
    return df


def main() -> None:
    all_records: List[Dict[str, float]] = []

    for directory_name in TEXTURE_DIRECTORIES:
        directory = Path(directory_name)
        for variant_info in collect_variants(directory):
            records = extract_features(variant_info)
            all_records.extend(records)

    dataset_df = stratified_split(all_records)

    train_count = int((dataset_df["split"] == "train").sum())
    test_count = int((dataset_df["split"] == "test").sum())

    print(f"📊 Dataset: {train_count} train, {test_count} test")

    output_path = Path("data.csv")
    dataset_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
