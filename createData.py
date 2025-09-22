"""High-quality dataset generation pipeline for Minecraft texture variants."""

from __future__ import annotations

"""Pipeline avanzado para generar data.csv de alta calidad."""

# DEPENDENCIAS EXACTAS
import numpy as np
import pandas as pd
from PIL import Image, ImageChops, ImageFilter
from pathlib import Path
from collections import defaultdict
import random
from typing import Dict, List, Tuple, Optional
import argparse
from scipy import ndimage
import math
from sklearn.model_selection import train_test_split

# CONFIGURACIÓN POR DEFECTO - AJUSTADA PARA MINECRAFT
CONFIG: Dict[str, Dict[str, Dict[str, float]]] = {
    'thresholds': {
        'area_min_pixels': 25,
        'area_max_ratio': 0.30,
        'iou_merge': 0.35,
        'iou_dedup': 0.70,
        'score_global_min': 0.60,
        'conf_min': 0.50,
        'sem_min': 0.40,
        'vis_min': 0.30,
        'diff_threshold_ratio': 0.25,
        'heuristic_threshold_ratio': 0.70,
    },
    'limits': {
        'max_per_class': 1000,
        'min_per_class': 50,
        'max_class_share': 0.40,
        'max_boxes_per_image': 40,
    },
    'morphology': {
        'opening_kernel': (2, 2),
        'closing_kernel': (3, 3),
        'dilation_kernel': (2, 2),
    },
}

# Mapeo de etiquetas para variantes emisivas
VARIANT_LABELS: Dict[str, str] = {
    'e': 'emissive',
    'eyes': 'emissive_eyes',
    'heart': 'emissive_heart',
    'spots': 'emissive_spots',
    'crackiness': 'emissive_cracks',
}

VARIANT_SUFFIXES: Tuple[Tuple[str, str], ...] = (
    ('_crackiness', 'crackiness'),
    ('_spots', 'spots'),
    ('_heart', 'heart'),
    ('_eyes', 'eyes'),
    ('_e', 'e'),
)

IMAGE_CACHE: Dict[Path, Dict[str, np.ndarray]] = {}
RNG = random.Random(42)


def analyze_filename_pattern(filename: str) -> Dict[str, any]:
    """Analiza patrones en nombres de archivo para clasificación inteligente."""

    stem = Path(filename).stem.lower()

    BLOCK_ITEMS = {'beam', 'shield', 'sword', 'bricks', 'wood', 'log', 'planks',
                   'door', 'bed', 'stone', 'ore', 'acacia', 'birch', 'spruce',
                   'glass', 'wool', 'concrete', 'terracotta', 'sandstone'}

    LARGE_MOBS = {'dragon', 'wither', 'warden', 'giant', 'golem', 'leviathan'}
    SMALL_MOBS = {'zombie', 'skeleton', 'creeper', 'spider', 'cow', 'pig', 'sheep'}

    is_block_item = any(keyword in stem for keyword in BLOCK_ITEMS)
    is_large_mob = any(keyword in stem for keyword in LARGE_MOBS)
    is_small_mob = any(keyword in stem for keyword in SMALL_MOBS)

    variant_suffixes = ['_e', '_eyes', '_heart', '_spots', '_crackiness']
    is_variant = any(stem.endswith(suffix) for suffix in variant_suffixes)
    base_name = None
    suffix_used = None

    if is_variant:
        for suffix in variant_suffixes:
            if stem.endswith(suffix):
                base_name = stem[:-len(suffix)]
                suffix_used = suffix
                break

    return {
        'is_variant': is_variant,
        'is_block_item': is_block_item,
        'is_large_mob': is_large_mob,
        'is_small_mob': is_small_mob,
        'base_name': base_name,
        'suffix': suffix_used,
        'requires_special_rules': is_large_mob or is_block_item,
    }


def apply_alpha_premultiplication(image: Image.Image) -> Image.Image:
    """Aplica premultiplicación de alpha para mejorar diferencias en transparencias."""

    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    arr = np.array(image, dtype=np.float32)
    alpha = arr[..., 3:4] / 255.0
    arr[..., :3] *= alpha
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode='RGBA')


def advanced_morphological_processing(binary_mask: np.ndarray, config: Dict) -> np.ndarray:
    """Post-procesamiento morfológico avanzado para casos complejos."""

    try:
        opened = ndimage.binary_opening(
            binary_mask,
            structure=np.ones(config['morphology']['opening_kernel'], dtype=bool),
        )
        closed = ndimage.binary_closing(
            opened,
            structure=np.ones(config['morphology']['closing_kernel'], dtype=bool),
        )
        dilated = ndimage.binary_dilation(
            closed,
            structure=np.ones(config['morphology']['dilation_kernel'], dtype=bool),
        )

        labeled, num_features = ndimage.label(dilated)
        component_sizes = np.bincount(labeled.ravel())
        min_size = config['thresholds']['area_min_pixels']
        cleaned_mask = np.zeros_like(binary_mask, dtype=bool)
        for i in range(1, num_features + 1):
            if component_sizes[i] >= min_size:
                cleaned_mask[labeled == i] = True
        return cleaned_mask
    except Exception as exc:  # pragma: no cover - fallback
        print(f"❌ Error en procesamiento morfológico: {exc}")
        return binary_mask


def find_connected_components_adaptive(binary_mask: np.ndarray, file_info: Dict, config: Dict) -> List[Dict]:
    """Encuentra componentes conectados aplicando restricciones adaptativas."""

    if not np.any(binary_mask):
        return []

    labeled, num_features = ndimage.label(binary_mask)
    slices = ndimage.find_objects(labeled)
    height, width = binary_mask.shape
    total_area = float(height * width)

    area_min = config['thresholds']['area_min_pixels']
    area_max_ratio = config['thresholds']['area_max_ratio']

    if file_info['is_large_mob']:
        area_max_ratio *= 1.4
    if file_info['is_block_item']:
        area_max_ratio *= 0.7

    regions: List[Dict] = []
    for idx, slc in enumerate(slices, start=1):
        if slc is None:
            continue
        region_mask = labeled[slc] == idx
        area = int(region_mask.sum())
        if area < area_min:
            continue
        area_ratio = area / total_area if total_area else 0.0
        if area_ratio > area_max_ratio:
            continue

        y0, y1 = slc[0].start, slc[0].stop
        x0, x1 = slc[1].start, slc[1].stop
        region_dict = {
            'x': int(x0),
            'y': int(y0),
            'width': int(x1 - x0),
            'height': int(y1 - y0),
            'area': area,
            'area_ratio': area_ratio,
            'mask': region_mask.astype(bool),
            'slice': slc,
        }
        regions.append(region_dict)

    return regions


def _region_iou(a: Dict, b: Dict) -> float:
    ax1, ay1 = a['x'], a['y']
    ax2, ay2 = ax1 + a['width'], ay1 + a['height']
    bx1, by1 = b['x'], b['y']
    bx2, by2 = bx1 + b['width'], by1 + b['height']

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = a['width'] * a['height']
    area_b = b['width'] * b['height']
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def merge_regions_intelligent(regions: List[Dict], diff_array: np.ndarray, file_info: Dict, config: Dict) -> List[Dict]:
    """Fusiones inteligentes de regiones con alta superposición."""

    if not regions:
        return []

    for region in regions:
        y0, x0 = region['y'], region['x']
        h, w = region['height'], region['width']
        patch = diff_array[y0:y0 + h, x0:x0 + w]
        mask = region.get('mask')
        if mask is not None and mask.shape == patch.shape:
            values = patch[mask]
        else:
            values = patch.reshape(-1)
        region['raw_score'] = float(np.mean(values)) / 255.0 if values.size else 0.0

    merged: List[Dict] = []
    for region in sorted(regions, key=lambda r: r['raw_score'], reverse=True):
        merged_into_existing = False
        for existing in merged:
            if _region_iou(region, existing) >= config['thresholds']['iou_merge']:
                x1 = min(existing['x'], region['x'])
                y1 = min(existing['y'], region['y'])
                x2 = max(existing['x'] + existing['width'], region['x'] + region['width'])
                y2 = max(existing['y'] + existing['height'], region['y'] + region['height'])
                existing.update({
                    'x': x1,
                    'y': y1,
                    'width': x2 - x1,
                    'height': y2 - y1,
                    'area': existing['width'] * existing['height'],
                })
                existing['raw_score'] = max(existing['raw_score'], region['raw_score'])
                merged_into_existing = True
                break
        if not merged_into_existing:
            merged.append(dict(region))

    deduped: List[Dict] = []
    for region in sorted(merged, key=lambda r: r['raw_score'], reverse=True):
        if all(_region_iou(region, other) <= config['thresholds']['iou_dedup'] for other in deduped):
            deduped.append(region)

    return deduped


def apply_region_limits(regions: List[Dict], diff_array: np.ndarray, file_info: Dict, config: Dict) -> List[Dict]:
    """Aplica límites de número de regiones por imagen con prioridad de score."""

    if not regions:
        return []

    for region in regions:
        y0, x0 = region['y'], region['x']
        h, w = region['height'], region['width']
        patch = diff_array[y0:y0 + h, x0:x0 + w]
        values = patch.reshape(-1)
        region['detection_confidence'] = float(np.clip(values.mean() / 255.0 if values.size else 0.0, 0.0, 1.0))
        suffix = file_info.get('suffix') or ''
        region['label'] = VARIANT_LABELS.get(suffix.strip('_') or 'e', 'emissive')
        region['is_heuristic'] = False
        region['source'] = 'difference'
        region.pop('mask', None)
        region.pop('slice', None)

    limit = config['limits']['max_boxes_per_image']
    if not file_info['is_large_mob']:
        limit = min(limit, 12 if file_info['is_block_item'] else 20)

    regions_sorted = sorted(regions, key=lambda r: r['detection_confidence'], reverse=True)
    return regions_sorted[:limit]


def process_variant_pair(base_path: Path, variant_path: Path, file_info: Dict, config: Dict) -> List[Dict]:
    """Pipeline 1: Procesamiento robusto para pares base-variante."""

    try:
        with Image.open(base_path) as base_raw, Image.open(variant_path) as variant_raw:
            base_img = base_raw.convert('RGBA')
            variant_img = variant_raw.convert('RGBA')
    except Exception as exc:
        print(f"❌ Error abriendo imágenes {variant_path.name}: {exc}")
        return []

    if base_img.size != variant_img.size:
        print(f"⚠️  Tamaños diferentes: {base_path.name} vs {variant_path.name}")
        return []

    base_premultiplied = apply_alpha_premultiplication(base_img)
    variant_premultiplied = apply_alpha_premultiplication(variant_img)

    diff_image = ImageChops.difference(base_premultiplied, variant_premultiplied)
    diff = np.array(diff_image.convert('L'), dtype=np.float32)
    max_diff = np.max(diff) if np.max(diff) > 0 else 1.0
    threshold = max(16, int(config['thresholds']['diff_threshold_ratio'] * max_diff))
    binary_mask = diff > threshold

    cleaned_mask = advanced_morphological_processing(binary_mask, config)
    regions = find_connected_components_adaptive(cleaned_mask, file_info, config)
    merged_regions = merge_regions_intelligent(regions, diff, file_info, config)
    final_regions = apply_region_limits(merged_regions, diff, file_info, config)

    print(f"🔍 {variant_path.name}: {len(regions)} → {len(merged_regions)} → {len(final_regions)} regiones")
    return final_regions


def heuristic_region_detection(image_path: Path, file_info: Dict, config: Dict) -> List[Dict]:
    """Pipeline 2: Heurística adaptativa basada en tipo de textura."""

    try:
        with Image.open(image_path) as pil_img:
            img = pil_img.convert('RGB')
            img = img.filter(ImageFilter.MedianFilter(size=3))
    except Exception as exc:
        print(f"❌ Error abriendo {image_path.name}: {exc}")
        return []

    img_array = np.array(img)
    gray = np.array(img.convert('L'), dtype=np.float32)

    max_val = np.max(gray) if np.max(gray) > 0 else 255
    threshold_ratio = config['thresholds']['heuristic_threshold_ratio']
    if file_info['is_block_item']:
        threshold_ratio *= 0.8
    elif file_info['is_large_mob']:
        threshold_ratio *= 1.2

    threshold = int(threshold_ratio * max_val)
    binary_mask = gray > threshold

    cleaned_mask = advanced_morphological_processing(binary_mask, config)
    regions = find_connected_components_adaptive(cleaned_mask, file_info, config)

    classified_regions: List[Dict] = []
    for region in regions:
        classified = classify_region_with_context(region.copy(), gray, img_array, file_info)
        if classified and classified['detection_confidence'] >= 0.4:
            classified_regions.append(classified)

    if file_info['is_block_item']:
        classified_regions = [r for r in classified_regions if r['label'] != 'auto_eyes']
        if len(classified_regions) > 3:
            classified_regions = sorted(classified_regions, key=lambda x: x['detection_confidence'], reverse=True)[:3]
    elif file_info['is_large_mob']:
        if len(classified_regions) > config['limits']['max_boxes_per_image']:
            classified_regions = sorted(
                classified_regions,
                key=lambda x: x['detection_confidence'],
                reverse=True,
            )[:config['limits']['max_boxes_per_image']]

    if not classified_regions:
        base_region = create_base_region_with_context(img.size, file_info)
        classified_regions.append(base_region)

    avg_confidence = np.mean([r['detection_confidence'] for r in classified_regions]) if classified_regions else 0
    print(f"🔍 [AUTO] {image_path.name}: {len(classified_regions)} regiones (conf: {avg_confidence:.2f})")
    return classified_regions


def classify_region_with_context(region: Dict, gray_image: np.ndarray, color_image: np.ndarray, file_info: Dict) -> Optional[Dict]:
    """Clasificación que considera el contexto específico de Minecraft."""

    x, y, w, h = region['x'], region['y'], region['width'], region['height']
    img_h, img_w = gray_image.shape
    if w <= 0 or h <= 0:
        return None

    subregion = gray_image[y:y + h, x:x + w] if y + h <= img_h and x + w <= img_w else np.array([])
    if subregion.size == 0:
        return None

    area = w * h
    aspect_ratio = w / h if h > 0 else 1.0
    y_relative = y / img_h if img_h > 0 else 0
    brightness_mean = float(np.mean(subregion))
    brightness_std = float(np.std(subregion))

    large_mob_bonus = 1.2 if file_info['is_large_mob'] else 1.0
    block_item_penalty = 0.6 if file_info['is_block_item'] else 1.0
    top_position_bonus = 1.5 if y_relative < 0.4 else 1.0

    label = 'unknown'
    confidence = 0.3

    if (
        area <= 100 * large_mob_bonus and
        brightness_mean > 200 * block_item_penalty and
        y_relative < 0.4 and
        0.5 <= aspect_ratio <= 2.0 and
        not file_info['is_block_item']
    ):
        label = 'auto_eyes'
        base_confidence = 0.8 * min(1.0, (brightness_mean - 200) / 55 if brightness_mean > 200 else 0.0)
        confidence = base_confidence * top_position_bonus * large_mob_bonus * block_item_penalty
    elif (
        brightness_mean > 180 and
        brightness_std > 30 and
        area <= 500 * large_mob_bonus and
        aspect_ratio <= 5.0
    ):
        label = 'auto_glow'
        confidence = 0.7 * large_mob_bonus * block_item_penalty
    elif (
        brightness_std > 50 and
        aspect_ratio > 3.0 and
        area <= 300 and
        not file_info['is_block_item']
    ):
        label = 'auto_cracks'
        confidence = 0.6

    min_confidence = 0.3 * block_item_penalty
    if confidence < min_confidence:
        return None

    region.update({
        'label': label,
        'detection_confidence': float(min(0.95, confidence)),
        'source': 'heuristic',
        'is_heuristic': True,
        'context_factors': {
            'large_mob_bonus': large_mob_bonus,
            'block_item_penalty': block_item_penalty,
            'top_position_bonus': top_position_bonus,
        },
    })
    region.pop('mask', None)
    region.pop('slice', None)
    return region


def create_base_region_with_context(image_size: Tuple[int, int], file_info: Dict) -> Dict:
    """Crea una región base cuando no se detectan regiones válidas."""

    width, height = image_size
    detection_conf = 0.55 if file_info['is_block_item'] else 0.62
    return {
        'x': 0,
        'y': 0,
        'width': width,
        'height': height,
        'area': width * height,
        'area_ratio': 1.0,
        'label': 'base_texture',
        'detection_confidence': detection_conf,
        'is_heuristic': True,
        'source': 'fallback',
    }


def _load_image_arrays(image_path: Path) -> Dict[str, np.ndarray]:
    if image_path in IMAGE_CACHE:
        return IMAGE_CACHE[image_path]

    with Image.open(image_path) as img:
        rgb = img.convert('RGB')
        rgb_array = np.array(rgb, dtype=np.float32)
        gray_array = np.array(rgb.convert('L'), dtype=np.float32)
        sobel_x = ndimage.sobel(gray_array, axis=1)
        sobel_y = ndimage.sobel(gray_array, axis=0)
        gradient = np.hypot(sobel_x, sobel_y)
        IMAGE_CACHE[image_path] = {
            'rgb': rgb_array,
            'gray': gray_array,
            'gradient': gradient,
        }
        return IMAGE_CACHE[image_path]


def extract_region_features(region: Dict, image_path: Path) -> Dict[str, float]:
    """Extrae características estadísticas de la región."""

    arrays = _load_image_arrays(image_path)
    rgb = arrays['rgb']
    gray = arrays['gray']
    gradient = arrays['gradient']

    x, y, w, h = region['x'], region['y'], region['width'], region['height']
    h = max(1, h)
    w = max(1, w)
    h_limit = min(y + h, rgb.shape[0])
    w_limit = min(x + w, rgb.shape[1])

    patch_rgb = rgb[y:h_limit, x:w_limit]
    patch_gray = gray[y:h_limit, x:w_limit]
    patch_gradient = gradient[y:h_limit, x:w_limit]

    r_mean = float(patch_rgb[..., 0].mean()) if patch_rgb.size else 0.0
    g_mean = float(patch_rgb[..., 1].mean()) if patch_rgb.size else 0.0
    b_mean = float(patch_rgb[..., 2].mean()) if patch_rgb.size else 0.0
    r_std = float(patch_rgb[..., 0].std()) if patch_rgb.size else 0.0
    g_std = float(patch_rgb[..., 1].std()) if patch_rgb.size else 0.0
    b_std = float(patch_rgb[..., 2].std()) if patch_rgb.size else 0.0

    brightness_mean = float(patch_gray.mean()) if patch_gray.size else 0.0
    brightness_std = float(patch_gray.std()) if patch_gray.size else 0.0
    gradient_mean = float(patch_gradient.mean()) if patch_gradient.size else 0.0

    image_area = float(rgb.shape[0] * rgb.shape[1]) if rgb.size else 1.0
    area = float(region['width'] * region['height'])
    area_ratio = min(1.0, area / image_area) if image_area else 0.0
    aspect_ratio = float(region['width'] / region['height']) if region['height'] else 0.0

    return {
        'r_mean': r_mean,
        'g_mean': g_mean,
        'b_mean': b_mean,
        'r_std': r_std,
        'g_std': g_std,
        'b_std': b_std,
        'brightness_mean': brightness_mean,
        'brightness_std': brightness_std,
        'gradient_magnitude': gradient_mean,
        'area': area,
        'area_ratio': area_ratio,
        'aspect_ratio': aspect_ratio,
        'image_area': image_area,
    }


def calculate_geometric_quality(region: Dict, file_info: Dict) -> float:
    area_ratio = region.get('area_ratio', 0.0)
    aspect_ratio = region.get('aspect_ratio', 1.0)

    min_ratio = CONFIG['thresholds']['area_min_pixels'] / max(region.get('image_area', 1.0), 1.0)
    max_ratio = CONFIG['thresholds']['area_max_ratio']
    if file_info['is_large_mob']:
        max_ratio *= 1.4
    if file_info['is_block_item']:
        max_ratio *= 0.8

    ratio_score = np.clip((area_ratio - min_ratio) / max(max_ratio - min_ratio, 1e-6), 0.0, 1.0)
    aspect_penalty = np.clip(1.0 - abs(math.log(max(aspect_ratio, 1e-3))), 0.0, 1.0)
    return float(0.7 * ratio_score + 0.3 * aspect_penalty)


def calculate_visual_quality_robust(region: Dict) -> float:
    brightness_std = region.get('brightness_std', 0.0)
    gradient = region.get('gradient_magnitude', 0.0)
    contrast_score = np.clip(brightness_std / 60.0, 0.0, 1.0)
    gradient_score = np.clip(gradient / 40.0, 0.0, 1.0)
    return float(0.6 * contrast_score + 0.4 * gradient_score)


def evaluate_semantic_coherence_adaptive(region: Dict, file_info: Dict) -> float:
    label = region.get('label', 'unknown')
    base_score = 0.4
    if label == 'auto_eyes' and not file_info['is_block_item']:
        base_score = 0.75
    elif label == 'auto_glow':
        base_score = 0.65
    elif label == 'auto_cracks':
        base_score = 0.55
    elif label.startswith('emissive'):
        base_score = 0.8
    elif label == 'base_texture':
        base_score = 0.6

    if file_info['is_large_mob'] and label in {'auto_eyes', 'auto_glow'}:
        base_score += 0.1
    if file_info['is_block_item'] and label == 'base_texture':
        base_score += 0.05
    return float(np.clip(base_score, 0.0, 1.0))


def predict_training_utility(region: Dict, file_info: Dict) -> float:
    conf = region.get('detection_confidence', 0.5)
    area_ratio = region.get('area_ratio', 0.0)
    utility = 0.6 * conf + 0.4 * np.clip(area_ratio / CONFIG['thresholds']['area_max_ratio'], 0.0, 1.0)
    if region.get('label') == 'base_texture' and file_info['is_block_item']:
        utility *= 0.9
    return float(np.clip(utility, 0.0, 1.0))


def comprehensive_quality_assessment(region: Dict, image_path: Path, file_info: Dict) -> Dict[str, float]:
    try:
        image_arrays = _load_image_arrays(image_path)
        geom_quality = calculate_geometric_quality(region, file_info)
        visual_quality = calculate_visual_quality_robust(region)
        semantic_quality = evaluate_semantic_coherence_adaptive(region, file_info)
        utility_score = predict_training_utility(region, file_info)

        weights = {
            'geometric': 0.20,
            'visual': 0.25,
            'semantic': 0.30,
            'utility': 0.15,
            'confidence': 0.10,
        }
        if file_info['is_large_mob']:
            weights['semantic'] = 0.35
            weights['visual'] = 0.20

        global_score = (
            weights['geometric'] * geom_quality
            + weights['visual'] * visual_quality
            + weights['semantic'] * semantic_quality
            + weights['utility'] * utility_score
            + weights['confidence'] * region.get('detection_confidence', 0.5)
        )
        return {
            'geometric_quality': geom_quality,
            'visual_quality': visual_quality,
            'semantic_coherence': semantic_quality,
            'training_utility': utility_score,
            'score_global': float(min(1.0, global_score)),
        }
    except Exception as exc:
        print(f"⚠️  Error en evaluación de calidad para {image_path.name}: {exc}")
        return {
            'geometric_quality': 0.3,
            'visual_quality': 0.3,
            'semantic_coherence': 0.3,
            'training_utility': 0.3,
            'score_global': 0.3,
        }


def apply_geometric_filters(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    if df.empty:
        return df
    area_min = config['thresholds']['area_min_pixels']
    area_max_ratio = config['thresholds']['area_max_ratio']
    mask_valid = (
        (df['width'] > 0)
        & (df['height'] > 0)
        & (df['area'] >= area_min)
        & ((df['area_ratio'] <= area_max_ratio * 1.2) | (df['label'] == 'base_texture'))
    )
    return df[mask_valid].reset_index(drop=True)


def apply_quality_filters(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    if df.empty:
        return df
    thresholds = config['thresholds']
    mask = (
        (df['detection_confidence'] >= thresholds['conf_min'])
        & (df['semantic_coherence'] >= thresholds['sem_min'])
        & (df['visual_quality'] >= thresholds['vis_min'])
        & (df['score_global'] >= thresholds['score_global_min'])
    )
    return df[mask].reset_index(drop=True)


def deduplicate_regions(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    if df.empty:
        return df

    df_sorted = df.sort_values('score_global', ascending=False).reset_index(drop=True)
    keep_indices: List[int] = []
    for idx, row in df_sorted.iterrows():
        bbox_row = {
            'x': row['x'],
            'y': row['y'],
            'width': row['width'],
            'height': row['height'],
        }
        if all(
            _region_iou(
                bbox_row,
                {
                    'x': df_sorted.loc[keep_idx, 'x'],
                    'y': df_sorted.loc[keep_idx, 'y'],
                    'width': df_sorted.loc[keep_idx, 'width'],
                    'height': df_sorted.loc[keep_idx, 'height'],
                },
            ) <= config['thresholds']['iou_dedup']
            for keep_idx in keep_indices
        ):
            keep_indices.append(idx)
    return df_sorted.loc[keep_indices].reset_index(drop=True)


def intelligent_balancing(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    if df.empty:
        return df

    limits = config['limits']
    grouped = []
    for label, data in df.groupby('label'):
        data_sorted = data.sort_values('score_global', ascending=False)
        original_count = len(data_sorted)
        if len(data_sorted) > limits['max_per_class']:
            data_sorted = data_sorted.head(limits['max_per_class'])
            print(f"📉 Submuestreando {label}: {original_count} → {len(data_sorted)}")
        if 0 < len(data_sorted) < limits['min_per_class']:
            repeat = int(math.ceil(limits['min_per_class'] / len(data_sorted)))
            oversampled = pd.concat([data_sorted] * repeat, ignore_index=True).head(limits['min_per_class'])
            data_sorted = oversampled
            print(f"📈 Sobremuestreando {label}: {original_count} → {len(data_sorted)}")
        grouped.append(data_sorted)

    balanced = pd.concat(grouped, ignore_index=True)
    total = len(balanced)
    final_groups = []
    for label, data in balanced.groupby('label'):
        max_share = int(limits['max_class_share'] * total)
        if max_share and len(data) > max_share:
            final_groups.append(data.head(max_share))
        else:
            final_groups.append(data)
    if final_groups:
        balanced = pd.concat(final_groups, ignore_index=True)
    return balanced.sort_values('score_global', ascending=False).reset_index(drop=True)


def enhanced_quality_validation(df: pd.DataFrame, config: Dict) -> Dict[str, any]:
    if len(df) == 0:
        return {
            'total_records': 0,
            'classes_detected': 0,
            'average_global_score': 0,
            'quality_distribution': {},
            'meets_standards': False,
            'alerts': ['Dataset vacío'],
            'recommendations': ['Revisar configuración de umbrales'],
        }

    quality_counts = defaultdict(int)
    quality_counts['excellent'] = len(df[df['score_global'] >= 0.8])
    quality_counts['good'] = len(df[(df['score_global'] >= 0.6) & (df['score_global'] < 0.8)])
    quality_counts['poor'] = len(df[df['score_global'] < 0.6])

    metrics = {
        'total_records': len(df),
        'classes_detected': df['label'].nunique(),
        'average_global_score': df['score_global'].mean(),
        'quality_distribution': dict(quality_counts),
        'meets_standards': True,
        'alerts': [],
        'recommendations': [],
    }

    quality_thresholds = {
        'min_total_records': 100,
        'min_global_score': config['thresholds']['score_global_min'],
        'min_class_examples': config['limits']['min_per_class'],
        'max_class_share': config['limits']['max_class_share'],
        'min_quality_ratio': 0.7,
    }

    if metrics['total_records'] < quality_thresholds['min_total_records']:
        metrics['alerts'].append(
            f"Dataset muy pequeño ({metrics['total_records']} < {quality_thresholds['min_total_records']})"
        )
        metrics['meets_standards'] = False

    if metrics['average_global_score'] < quality_thresholds['min_global_score']:
        metrics['alerts'].append(
            f"Score global bajo ({metrics['average_global_score']:.3f} < {quality_thresholds['min_global_score']})"
        )
        metrics['meets_standards'] = False

    quality_ratio = (
        metrics['quality_distribution']['excellent'] + metrics['quality_distribution']['good']
    ) / metrics['total_records']
    if quality_ratio < quality_thresholds['min_quality_ratio']:
        metrics['alerts'].append(
            f"Proporción de calidad insuficiente ({quality_ratio:.1%} < {quality_thresholds['min_quality_ratio']:.0%})"
        )
        metrics['meets_standards'] = False

    class_proportions = df['label'].value_counts(normalize=True)
    for class_name, proportion in class_proportions.items():
        if proportion > quality_thresholds['max_class_share']:
            metrics['alerts'].append(
                f"Clase '{class_name}' desbalanceada ({proportion:.1%} > {quality_thresholds['max_class_share']:.0%})"
            )
            metrics['meets_standards'] = False
        class_count = int(len(df[df['label'] == class_name]))
        if class_count < quality_thresholds['min_class_examples']:
            metrics['alerts'].append(
                f"Clase '{class_name}' insuficiente ({class_count} < {quality_thresholds['min_class_examples']})"
            )
            metrics['meets_standards'] = False

    if metrics['quality_distribution']['poor'] > metrics['total_records'] * 0.3:
        metrics['recommendations'].append("Considerar aumentar umbrales de calidad")
    if metrics['classes_detected'] < 3:
        metrics['recommendations'].append("Dataset puede estar muy sesgado - agregar más variedad")

    return metrics


def generate_comprehensive_report(results: Dict[str, any], df: pd.DataFrame) -> None:
    print("📊 Estadísticas:")
    print(f"   • Registros totales: {results['total_records']}")
    print(f"   • Clases detectadas: {results['classes_detected']}")
    print(f"   • Score global promedio: {results['average_global_score']:.3f}")
    if results['quality_distribution']:
        excellent = results['quality_distribution'].get('excellent', 0)
        good = results['quality_distribution'].get('good', 0)
        poor = results['quality_distribution'].get('poor', 0)
        total = max(results['total_records'], 1)
        print(
            f"   • Distribución de calidad: Excelente: {excellent/total:.0%}, "
            f"Bueno: {good/total:.0%}, Pobre: {poor/total:.0%}"
        )

    if not df.empty:
        print("\n🏷️ Distribución por clases:")
        for label, count in df['label'].value_counts().items():
            share = count / len(df)
            print(f"   • {label}: {count} registros ({share:.1%})")

    if results['alerts']:
        print("\n⚠️ Alertas:")
        for alert in results['alerts']:
            print(f"   • {alert}")

    if results['recommendations']:
        print("\n💡 Recomendaciones:")
        for recommendation in results['recommendations']:
            print(f"   • {recommendation}")

    if results['meets_standards']:
        print("\n✅ Cumple todos los estándares de calidad")
    else:
        print("\n❌ No cumple los estándares de calidad")


def save_validation_samples(df: pd.DataFrame, n_samples: int = 20) -> None:
    if df.empty:
        return
    output_dir = Path('validation_samples')
    output_dir.mkdir(exist_ok=True)

    ranked = df.sort_values('score_global', ascending=False)
    if len(ranked) > n_samples:
        indices = list(range(len(ranked)))
        RNG.shuffle(indices)
        selected_indices = sorted(indices[:n_samples])
        sample_rows = ranked.iloc[selected_indices]
    else:
        sample_rows = ranked

    for idx, row in sample_rows.iterrows():
        image_path = Path(row['file_path'])
        if not image_path.exists():
            continue
        try:
            with Image.open(image_path) as img:
                box = (
                    int(row['x']),
                    int(row['y']),
                    int(row['x'] + row['width']),
                    int(row['y'] + row['height']),
                )
                crop = img.crop(box)
                sample_name = f"{image_path.stem}_{idx}_{row['label']}.png"
                crop.save(output_dir / sample_name)
        except Exception:
            continue


def discover_and_classify_files(root: Path, include_input: bool) -> List[Dict]:
    directories: List[Path] = []
    if root.exists():
        directories.append(root)
    if include_input:
        input_dir = Path('input')
        if input_dir.exists():
            directories.append(input_dir)

    seen_paths: set[Path] = set()
    discovered: List[Dict] = []
    for directory in directories:
        for path in directory.rglob('*.png'):
            resolved = path.resolve()
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            info = analyze_filename_pattern(path.name)
            base_path: Optional[Path] = None
            if info['is_variant'] and info['base_name']:
                candidate = path.with_name(f"{info['base_name']}{path.suffix}")
                if candidate.exists():
                    base_path = candidate
                else:
                    for lookup_dir in directories:
                        alt = lookup_dir / f"{info['base_name']}{path.suffix}"
                        if alt.exists():
                            base_path = alt
                            break
            info.update({
                'path': path,
                'base_path': base_path,
                'directory': directory,
            })
            discovered.append(info)
    discovered.sort(key=lambda item: item['path'].name)
    return discovered


def apply_region_postprocessing(region: Dict, file_info: Dict, image_path: Path) -> Dict:
    features = extract_region_features(region, image_path)
    region.update(features)
    quality_scores = comprehensive_quality_assessment(region, image_path, file_info)
    region.update(quality_scores)
    region['file_name'] = image_path.name
    region['file_path'] = str(image_path)
    region.pop('context_factors', None)
    return region


def build_records_for_file(file_info: Dict, config: Dict) -> List[Dict]:
    path: Path = file_info['path']
    regions: List[Dict]

    if file_info['is_variant'] and file_info.get('base_path') and file_info['base_path'].exists():
        regions = process_variant_pair(file_info['base_path'], path, file_info, config)
    else:
        regions = heuristic_region_detection(path, file_info, config)

    records: List[Dict] = []
    for region in regions:
        try:
            processed = apply_region_postprocessing(region, file_info, path)
            records.append(processed)
        except Exception as exc:
            print(f"❌ Error procesando región en {path.name}: {exc}")
    return records


def main() -> None:
    print("🚀 INICIANDO GENERACIÓN DE DATASET CON CONTROL DE CALIDAD")
    print("=" * 70)

    parser = argparse.ArgumentParser(
        description='Generar dataset de texturas Minecraft con calidad garantizada'
    )
    parser.add_argument('--root', default='toLearn', help='Directorio raíz con texturas')
    parser.add_argument('--include-input', action='store_true', default=True, help='Incluir directorio input/')
    parser.add_argument('--quality-report', action='store_true', default=True, help='Generar reporte de calidad')
    parser.add_argument('--save-samples', action='store_true', default=True, help='Guardar muestras visuales')
    args = parser.parse_args()

    config = CONFIG

    print("🔍 Descubriendo y clasificando archivos...")
    all_files = discover_and_classify_files(Path(args.root), args.include_input)
    if not all_files:
        print("❌ No se encontraron archivos PNG válidos")
        return

    print(f"📁 Archivos encontrados: {len(all_files)}")
    print("🔄 Procesando archivos con pipelines inteligentes...")

    all_regions: List[Dict] = []
    for file_info in all_files:
        try:
            regions = build_records_for_file(file_info, config)
            all_regions.extend(regions)
        except Exception as exc:
            print(f"❌ Error procesando {file_info['path'].name}: {exc}")

    if not all_regions:
        print("❌ No se generaron regiones válidas")
        return

    df_initial = pd.DataFrame(all_regions)
    print(f"📦 Dataset inicial: {len(df_initial)} regiones")

    print("\n" + "=" * 70)
    print("🔬 APLICANDO CONTROL DE CALIDAD EN CASCADA")
    print("=" * 70)

    df_geometric = apply_geometric_filters(df_initial, config)
    print(f"📏 Filtro geométrico: {len(df_initial)} → {len(df_geometric)}")

    df_quality = apply_quality_filters(df_geometric, config)
    print(f"🎯 Filtro de calidad: {len(df_geometric)} → {len(df_quality)}")

    df_deduped = deduplicate_regions(df_quality, config)
    print(f"🧹 Eliminación de duplicados: {len(df_quality)} → {len(df_deduped)}")

    print("\n⚖️ APLICANDO BALANCEO INTELIGENTE")
    df_balanced = intelligent_balancing(df_deduped, config)
    print(f"📊 Dataset balanceado: {len(df_deduped)} → {len(df_balanced)}")

    print("\n📋 VALIDACIÓN FINAL DE CALIDAD")
    validation_results = enhanced_quality_validation(df_balanced, config)

    print("\n" + "=" * 70)
    print("📊 REPORTE FINAL DE CALIDAD")
    print("=" * 70)
    generate_comprehensive_report(validation_results, df_balanced)

    if validation_results['meets_standards']:
        train_df, test_df = train_test_split(
            df_balanced,
            test_size=0.2,
            random_state=42,
            stratify=df_balanced['label'] if df_balanced['label'].nunique() > 1 else None,
        )
        train_df = train_df.copy()
        test_df = test_df.copy()
        train_df['split'] = 'train'
        test_df['split'] = 'test'
        df_final = pd.concat([train_df, test_df], ignore_index=True)
        df_final.to_csv('data.csv', index=False)

        print("\n💾 DATASET GUARDADO EXITOSAMENTE")
        print(f"   • Registros totales: {len(df_final)}")
        print(f"   • Entrenamiento: {len(train_df)}")
        print(f"   • Prueba: {len(test_df)}")
        print(f"   • Score global promedio: {validation_results['average_global_score']:.3f}")

        if args.save_samples:
            save_validation_samples(df_final, n_samples=20)
            print("   🔍 Muestras de validación guardadas en 'validation_samples/'")
        print("✅ PROCESO COMPLETADO EXITOSAMENTE")
    else:
        print("\n❌ EL DATASET NO CUMPLE LOS ESTÁNDARES DE CALIDAD")
        print("   Alertas identificadas:")
        for alert in validation_results['alerts']:
            print(f"   • {alert}")
        if validation_results['recommendations']:
            print("\n   💡 Recomendaciones:")
            for recommendation in validation_results['recommendations']:
                print(f"   • {recommendation}")


if __name__ == '__main__':
    main()
