"""Advanced dataset creation pipeline for Minecraft texture variants."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageChops
from sklearn.model_selection import train_test_split

# Directories to scan recursively for PNG textures.
TEXTURE_DIRECTORIES: Sequence[Path] = (Path("toLearn"), Path("input"))

# Dataset column definition.
COLUMNAS_DATASET: List[str] = [
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
    "is_heuristic",
    "detection_confidence",
    "split",
]

# Mapping between suffixes/labels and their dataset-friendly names.
NUEVAS_ETIQUETAS: Dict[str, str] = {
    "e": "emissive",
    "eyes": "eyes",
    "heart": "heart",
    "spots": "spots",
    "crackiness": "cracks",
    "auto_eyes": "auto_detected_eyes",
    "auto_glow": "auto_detected_glow",
    "auto_cracks": "auto_detected_cracks",
    "base_texture": "base_texture",
    "unknown": "unknown",
}

# Ordered suffix definitions for variant textures (longest first to avoid ambiguity).
VARIANT_SUFFIXES: Sequence[Tuple[str, str]] = (
    ("_crackiness", "crackiness"),
    ("_spots", "spots"),
    ("_heart", "heart"),
    ("_eyes", "eyes"),
    ("_e", "e"),
)


@dataclass(frozen=True)
class VariantDetection:
    """Information about a detected variant region."""

    coords: np.ndarray
    bbox: Tuple[int, int, int, int]
    label_key: str
    detection_confidence: float
    is_heuristic: bool


def iter_texture_files() -> List[Path]:
    """Return all PNG files found recursively inside configured directories."""

    discovered: List[Path] = []
    seen: set[Path] = set()
    for base_dir in TEXTURE_DIRECTORIES:
        if not base_dir.exists():
            continue
        for path in base_dir.rglob("*.png"):
            # Avoid duplicates when directories overlap.
            if path not in seen:
                discovered.append(path)
                seen.add(path)
    return discovered


def detectar_sufijo(archivo_path: Path) -> Optional[Tuple[str, str]]:
    """Return the suffix tuple (suffix, label_key) if the file matches a known variant."""

    stem = archivo_path.stem
    for suffix, label_key in VARIANT_SUFFIXES:
        if stem.endswith(suffix) and len(stem) > len(suffix):
            return suffix, label_key
    return None


def obtener_base_para_variante(variant_path: Path) -> Optional[Path]:
    """Return the base texture path for the provided variant, if it exists."""

    detected = detectar_sufijo(variant_path)
    if detected is None:
        return None

    suffix, _ = detected
    base_stem = variant_path.stem[: -len(suffix)]
    candidate = variant_path.with_name(f"{base_stem}{variant_path.suffix}")
    if candidate.exists():
        return candidate

    # Fallback: search in all known directories by name.
    base_name = f"{base_stem}{variant_path.suffix}"
    for directory in TEXTURE_DIRECTORIES:
        candidate = directory / base_name
        if candidate.exists():
            return candidate
    return None


def compute_regions(mask: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """Compute connected components from a boolean mask."""

    visited = np.zeros_like(mask, dtype=bool)
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

    regions: List[Tuple[np.ndarray, Tuple[int, int, int, int]]] = []

    for r in range(rows):
        for c in range(cols):
            if not mask[r, c] or visited[r, c]:
                continue

            stack = [(r, c)]
            coords_list: List[Tuple[int, int]] = []
            min_r = max_r = r
            min_c = max_c = c

            while stack:
                cr, cc = stack.pop()
                if visited[cr, cc] or not mask[cr, cc]:
                    continue
                visited[cr, cc] = True
                coords_list.append((cr, cc))
                min_r = min(min_r, cr)
                max_r = max(max_r, cr)
                min_c = min(min_c, cc)
                max_c = max(max_c, cc)

                for dr, dc in neighbours:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc] and mask[nr, nc]:
                        stack.append((nr, nc))

            if not coords_list:
                continue

            coords_array = np.array(coords_list, dtype=int)
            bbox = (min_c, min_r, max_c - min_c + 1, max_r - min_r + 1)
            regions.append((coords_array, bbox))

    return regions


def encontrar_componentes_conectados(
    mask: np.ndarray, *, area_minima: int, area_maxima: float
) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """Find connected components filtered by area constraints."""

    regiones = compute_regions(mask)
    filtradas: List[Tuple[np.ndarray, Tuple[int, int, int, int]]] = []
    for coords, bbox in regiones:
        area = coords.shape[0]
        if area < area_minima:
            continue
        if area > area_maxima:
            continue
        filtradas.append((coords, bbox))
    return filtradas


def clasificar_region_heuristica(
    region: Tuple[int, int, int, int], gray_image: np.ndarray, color_image: Image.Image
) -> Tuple[str, float]:
    """Clasifica una región detectada usando reglas heurísticas."""

    x, y, w, h = region
    subregion = gray_image[y : y + h, x : x + w]

    brillo_promedio = float(subregion.mean()) if subregion.size else 0.0
    contraste = float(subregion.std()) if subregion.size else 0.0
    relacion_aspecto = float(w / h) if h else 1.0
    area = float(w * h)
    posicion_relativa_y = float(y / gray_image.shape[0]) if gray_image.shape[0] else 0.0

    if (
        area <= 100
        and brillo_promedio > 200
        and posicion_relativa_y < 0.4
        and 0.5 <= relacion_aspecto <= 2.0
    ):
        return "auto_eyes", 0.85
    if brillo_promedio > 180 and contraste > 30 and area <= 500:
        return "auto_glow", 0.8
    if contraste > 50 and relacion_aspecto > 3.0 and area <= 300:
        return "auto_cracks", 0.75
    if area > 1000:
        return "base_texture", 0.65
    return "unknown", 0.55


def build_dataset_record(
    *,
    file_name: str,
    bbox: Tuple[int, int, int, int],
    coords: np.ndarray,
    label_key: str,
    detection_confidence: float,
    is_heuristic: bool,
    color_array: np.ndarray,
    brightness_array: np.ndarray,
    gradient_magnitude: np.ndarray,
    total_area: float,
) -> Dict[str, object]:
    """Build a dataset record from region coordinates and image statistics."""

    if coords.size == 0:
        raise ValueError("Empty region coordinates are not allowed")

    rows = coords[:, 0]
    cols = coords[:, 1]

    r_values = color_array[rows, cols, 0].astype(float)
    g_values = color_array[rows, cols, 1].astype(float)
    b_values = color_array[rows, cols, 2].astype(float)

    brightness_values = brightness_array[rows, cols]
    gradient_values = gradient_magnitude[rows, cols]

    x, y, width, height = bbox
    area = float(coords.shape[0])
    area_ratio = float(area / total_area) if total_area else 0.0
    aspect_ratio = float(width / height) if height else 0.0

    label = NUEVAS_ETIQUETAS.get(label_key, label_key)

    return {
        "file_name": file_name,
        "x": float(x),
        "y": float(y),
        "width": float(width),
        "height": float(height),
        "label": label,
        "r_mean": float(r_values.mean()),
        "g_mean": float(g_values.mean()),
        "b_mean": float(b_values.mean()),
        "r_std": float(r_values.std(ddof=0)),
        "g_std": float(g_values.std(ddof=0)),
        "b_std": float(b_values.std(ddof=0)),
        "brightness_mean": float(brightness_values.mean()),
        "brightness_std": float(brightness_values.std(ddof=0)),
        "gradient_magnitude": float(gradient_values.mean()),
        "aspect_ratio": aspect_ratio,
        "area_ratio": area_ratio,
        "is_heuristic": bool(is_heuristic),
        "detection_confidence": float(np.clip(detection_confidence, 0.0, 1.0)),
        "split": "unassigned",
    }


def procesar_con_diferencia(variant_path: Path) -> List[Dict[str, object]]:
    """Process variant textures by computing the difference with their base texture."""

    base_path = obtener_base_para_variante(variant_path)
    detected = detectar_sufijo(variant_path)
    if detected is None:
        return []
    _, label_key = detected

    if base_path is None or not base_path.exists():
        # No base available: fallback handled by caller.
        return []

    with Image.open(base_path) as base_img_raw, Image.open(variant_path) as variant_img_raw:
        base_img = base_img_raw.convert("RGBA")
        variant_img = variant_img_raw.convert("RGBA")

        if base_img.size != variant_img.size:
            return []

        diff = ImageChops.difference(base_img, variant_img)
        diff_gray = diff.convert("L")
        diff_array = np.array(diff_gray, dtype=np.uint8)

        max_diff = int(diff_array.max())
        if max_diff == 0:
            print(f"🔍 Analizando {variant_path.name}: 0 regiones detectadas")
            return []

        threshold = max_diff * 0.3
        mask = diff_array > threshold

        if not mask.any():
            print(f"🔍 Analizando {variant_path.name}: 0 regiones detectadas")
            return []

        regions = compute_regions(mask)
        print(f"🔍 Analizando {variant_path.name}: {len(regions)} regiones detectadas")

        variant_rgb = variant_img.convert("RGB")
        color_array = np.array(variant_rgb, dtype=np.uint8)
        gray_array = np.array(variant_rgb.convert("L"), dtype=float)
        gy, gx = np.gradient(gray_array)
        gradient_magnitude = np.sqrt(gx**2 + gy**2)
        brightness_array = 0.299 * color_array[:, :, 0] + 0.587 * color_array[:, :, 1] + 0.114 * color_array[:, :, 2]

        total_area = float(variant_img.size[0] * variant_img.size[1])

        records: List[Dict[str, object]] = []
        for coords, bbox in regions:
            if coords.size == 0:
                continue
            rows = coords[:, 0]
            cols = coords[:, 1]
            diff_values = diff_array[rows, cols].astype(float) / 255.0
            detection_confidence = float(np.clip(diff_values.mean(), 0.0, 1.0))

            record = build_dataset_record(
                file_name=variant_path.name,
                bbox=bbox,
                coords=coords,
                label_key=label_key,
                detection_confidence=detection_confidence,
                is_heuristic=False,
                color_array=color_array,
                brightness_array=brightness_array,
                gradient_magnitude=gradient_magnitude,
                total_area=total_area,
            )
            records.append(record)

        return records


def procesar_archivo_base(imagen_path: Path) -> List[Dict[str, object]]:
    """Procesa archivos base sin sufijos emisivos usando detección heurística."""

    with Image.open(imagen_path) as img_raw:
        color_image = img_raw.convert("RGB")
        gray_image = np.array(color_image.convert("L"))

    img_array = np.array(color_image, dtype=np.uint8)
    gray_float = gray_image.astype(float)
    gy, gx = np.gradient(gray_float)
    gradient_magnitude = np.sqrt(gx**2 + gy**2)
    brightness_array = 0.299 * img_array[:, :, 0] + 0.587 * img_array[:, :, 1] + 0.114 * img_array[:, :, 2]

    max_val = float(gray_image.max())
    threshold = int(0.7 * max_val) if max_val > 0 else 128
    mask = gray_image > threshold

    total_area = float(gray_image.size)

    regiones = encontrar_componentes_conectados(
        mask, area_minima=5, area_maxima=max(1.0, 0.5 * total_area)
    )

    resultados: List[Dict[str, object]] = []
    detecciones: List[VariantDetection] = []

    for coords, bbox in regiones:
        etiqueta, confianza = clasificar_region_heuristica(bbox, gray_image, color_image)
        detecciones.append(
            VariantDetection(
                coords=coords,
                bbox=bbox,
                label_key=etiqueta,
                detection_confidence=confianza,
                is_heuristic=True,
            )
        )

    if detecciones:
        confianzas = [d.detection_confidence for d in detecciones]
        confianza_promedio = float(np.mean(confianzas)) if confianzas else 0.0
        print(
            f"🔍 [AUTO] Analizando {imagen_path.name}: {len(detecciones)} regiones heurísticas (confianza: {confianza_promedio:.2f})"
        )
    else:
        print(f"🔍 [BASE] Textura {imagen_path.name}: marcada como base completa")
        h, w = gray_image.shape
        bbox = (0, 0, w, h)
        coords = np.indices((h, w)).reshape(2, -1).T
        detecciones.append(
            VariantDetection(
                coords=coords,
                bbox=bbox,
                label_key="base_texture",
                detection_confidence=0.65,
                is_heuristic=True,
            )
        )

    for deteccion in detecciones:
        try:
            record = build_dataset_record(
                file_name=imagen_path.name,
                bbox=deteccion.bbox,
                coords=deteccion.coords,
                label_key=deteccion.label_key,
                detection_confidence=deteccion.detection_confidence,
                is_heuristic=deteccion.is_heuristic,
                color_array=img_array,
                brightness_array=brightness_array,
                gradient_magnitude=gradient_magnitude,
                total_area=total_area,
            )
            resultados.append(record)
        except ValueError:
            continue

    return resultados


def es_archivo_variante(archivo_path: Path) -> bool:
    """Return whether a file uses a known emissive suffix."""

    return detectar_sufijo(archivo_path) is not None


def procesar_archivo_seguro(archivo_path: Path) -> List[Dict[str, object]]:
    """Procesa un archivo con manejo robusto de errores."""

    try:
        if es_archivo_variante(archivo_path):
            registros = procesar_con_diferencia(archivo_path)
            if registros:
                return registros
            # Fallback to heuristics if no base data available or no regions detected.
            return procesar_archivo_base(archivo_path)
        return procesar_archivo_base(archivo_path)
    except Exception as exc:  # noqa: BLE001
        print(f"❌ Error procesando {archivo_path.name}: {exc}")
        return [
            {
                "file_name": archivo_path.name,
                "x": float("nan"),
                "y": float("nan"),
                "width": float("nan"),
                "height": float("nan"),
                "label": "error",
                "r_mean": float("nan"),
                "g_mean": float("nan"),
                "b_mean": float("nan"),
                "r_std": float("nan"),
                "g_std": float("nan"),
                "b_std": float("nan"),
                "brightness_mean": float("nan"),
                "brightness_std": float("nan"),
                "gradient_magnitude": float("nan"),
                "aspect_ratio": float("nan"),
                "area_ratio": float("nan"),
                "is_heuristic": True,
                "detection_confidence": 0.0,
                "split": "unassigned",
            }
        ]


def balancear_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Balancea el dataset limitando clases dominantes y ampliando clases pequeñas."""

    if df.empty:
        return df

    datasets_balanceados: List[pd.DataFrame] = []
    conteo_clases = Counter(df["label"])

    MAXIMO_POR_CLASE = 1000
    MINIMO_POR_CLASE = 50
    OBJETIVO_POR_CLASE = 200

    for clase, count in conteo_clases.items():
        datos_clase = df[df["label"] == clase].copy()

        if count > MAXIMO_POR_CLASE:
            datos_clase = datos_clase.sample(MAXIMO_POR_CLASE, random_state=42)
            print(f"📉 Submuestreando {clase}: {count} → {MAXIMO_POR_CLASE}")
        elif count < MINIMO_POR_CLASE:
            if count > 0:
                factor_sobre = MINIMO_POR_CLASE // count + 1
                datos_clase = pd.concat([datos_clase] * factor_sobre, ignore_index=True)
                datos_clase = datos_clase.head(MINIMO_POR_CLASE)
                print(f"📈 Sobremuestreando {clase}: {count} → {MINIMO_POR_CLASE}")

        # Ajustar hacia el objetivo ideal si es posible.
        if len(datos_clase) > OBJETIVO_POR_CLASE:
            datos_clase = datos_clase.sample(OBJETIVO_POR_CLASE, random_state=42)
        elif len(datos_clase) < OBJETIVO_POR_CLASE:
            repeticiones = int(math.ceil(OBJETIVO_POR_CLASE / max(len(datos_clase), 1)))
            datos_clase = pd.concat([datos_clase] * max(repeticiones, 1), ignore_index=True)
            datos_clase = datos_clase.head(OBJETIVO_POR_CLASE)

        datasets_balanceados.append(datos_clase)

    if not datasets_balanceados:
        return df

    return pd.concat(datasets_balanceados, ignore_index=True)


def validar_calidad_dataset(df: pd.DataFrame) -> bool:
    """Valida que el dataset cumple con los criterios de calidad."""

    if df.empty:
        print("=== VALIDACIÓN DE CALIDAD DEL DATASET ===")
        print("Dataset vacío: no se pueden validar criterios.")
        return False

    conteo = df["label"].value_counts()
    total = len(df)

    print("=== VALIDACIÓN DE CALIDAD DEL DATASET ===")
    print(f"Total de registros: {total}")
    print("Distribución por clases:")

    criterios_cumplidos = True

    for clase, count in conteo.items():
        porcentaje = (count / total) * 100
        print(f"  {clase}: {count} ejemplos ({porcentaje:.1f}%)")
        if porcentaje > 40:
            print("  ⚠️  DEMASIADO GRANDE (>40%)")
            criterios_cumplidos = False
        if count < 50:
            print("  ⚠️  MUY PEQUEÑO (<50 ejemplos)")
            criterios_cumplidos = False

    train_count = int((df["split"] == "train").sum())
    test_count = int((df["split"] == "test").sum())
    ratio_test = test_count / total if total else 0

    print(f"\nDivisión train/test: {train_count}/{test_count} ({ratio_test:.1%})")
    if not (0.15 <= ratio_test <= 0.25):
        print("⚠️  División train/test fuera del rango 15-25%")
        criterios_cumplidos = False

    detecciones_heuristicas = df[df["is_heuristic"]]
    if not detecciones_heuristicas.empty:
        confianza_media = float(detecciones_heuristicas["detection_confidence"].mean())
        print(f"Confianza promedio detecciones heurísticas: {confianza_media:.2f}")
        if confianza_media <= 0.5:
            print("⚠️  Confianza promedio heurística ≤ 0.5")
            criterios_cumplidos = False

    if criterios_cumplidos:
        print("✅ Dataset cumple todos los criterios de calidad")
    else:
        print("❌ Dataset necesita ajustes de balanceo")

    return criterios_cumplidos


def main() -> None:
    archivos_png = iter_texture_files()
    if not archivos_png:
        print("No se encontraron archivos PNG para procesar.")
        return

    registros: List[Dict[str, object]] = []

    for archivo in archivos_png:
        registros.extend(procesar_archivo_seguro(archivo))

    if not registros:
        print("No se generaron registros para el dataset.")
        return

    df = pd.DataFrame(registros, columns=COLUMNAS_DATASET)

    df_balanceado = balancear_dataset(df)

    if df_balanceado.empty:
        print("No se pudo balancear el dataset: sin datos disponibles.")
        return

    train_df, test_df = train_test_split(
        df_balanceado,
        test_size=0.2,
        random_state=42,
        stratify=df_balanceado["label"] if df_balanceado["label"].nunique() > 1 else None,
    )

    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["split"] = "train"
    test_df["split"] = "test"

    df_final = pd.concat([train_df, test_df], ignore_index=True)

    df_final.to_csv("data.csv", index=False)

    validar_calidad_dataset(df_final)

    print(f"🎯 Dataset final guardado: {len(df_final)} registros")


if __name__ == "__main__":
    main()
