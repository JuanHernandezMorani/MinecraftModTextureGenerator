import pandas as pd
from pathlib import Path

def load_csv(path: str) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path}")
    return pd.read_csv(path)

def compare_datasets(new_path="data.csv", old_path="data_old.csv"):
    # Cargar datasets
    new_df = load_csv(new_path)
    old_df = load_csv(old_path)

    print("📊 COMPARACIÓN DE DATASETS")
    print("=" * 60)

    # Tamaños
    print(f"🔹 Registros nuevos: {len(new_df)}")
    print(f"🔹 Registros antiguos: {len(old_df)}\n")

    # Columnas presentes
    print("🔹 Columnas en nuevo:", list(new_df.columns))
    print("🔹 Columnas en viejo:", list(old_df.columns), "\n")

    # Distribución por clase
    if "label" in new_df.columns and "label" in old_df.columns:
        new_counts = new_df["label"].value_counts(normalize=True) * 100
        old_counts = old_df["label"].value_counts(normalize=True) * 100
        compare_df = pd.DataFrame({"Nuevo %": new_counts, "Viejo %": old_counts}).fillna(0)
        print("📌 Distribución por clase (%):")
        print(compare_df.round(2), "\n")

    # Métricas de bounding boxes
    if set(["width", "height"]).issubset(new_df.columns) and set(["width", "height"]).issubset(old_df.columns):
        new_area = (new_df["width"] * new_df["height"]).mean()
        old_area = (old_df["width"] * old_df["height"]).mean()
        print(f"📏 Área promedio de bounding boxes:")
        print(f"   Nuevo: {new_area:.2f}")
        print(f"   Viejo: {old_area:.2f}\n")

    # Score/calidad si existen
    quality_cols = ["score_global", "detection_confidence"]
    for col in quality_cols:
        if col in new_df.columns and col in old_df.columns:
            new_mean = new_df[col].mean()
            old_mean = old_df[col].mean()
            print(f"🎯 {col}:")
            print(f"   Nuevo promedio: {new_mean:.3f}")
            print(f"   Viejo promedio: {old_mean:.3f}\n")

    # Diferencias de archivos procesados
    if "file_name" in new_df.columns and "file_name" in old_df.columns:
        new_files = set(new_df["file_name"].unique())
        old_files = set(old_df["file_name"].unique())
        only_new = new_files - old_files
        only_old = old_files - new_files

        print("📂 Diferencias en archivos procesados:")
        print(f"   Archivos solo en nuevo: {len(only_new)}")
        if only_new:
            print("   Ejemplos:", list(only_new)[:5])
        print(f"   Archivos solo en viejo: {len(only_old)}")
        if only_old:
            print("   Ejemplos:", list(only_old)[:5])

if __name__ == "__main__":
    compare_datasets()
