from pathlib import Path
import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.windows import Window
from tqdm import tqdm


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

raster_path = BASE_DIR / "data" / "raw" / "calidad_agua" / "ortomosaicos" / "Humedal_Cordoba.tif"
points_path = BASE_DIR / "data" / "raw" / "calidad_agua" / "puntos_muestreo" / "Puntos_Muestreo_F.shp"
excel_path = BASE_DIR / "data" / "raw" / "calidad_agua" / "laboratorio" / "Resultados_Calidad_Agua_HCordoba.xlsx"
output_path = BASE_DIR / "data" / "processed" / "calidad_agua" / "Puntos_Muestreo_Reflectancia_Indices_Excel_2.gpkg"

window_size = 11

band_names = ["Blue", "Green", "Pan", "Red", "RedEdge", "NIR"]


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def validar_archivo(path: Path, descripcion: str) -> None:
    """
    Verifica que un archivo exista.
    """
    if not path.exists():
        raise FileNotFoundError(f"No se encontró {descripcion}: {path}")


def validar_window_size(window_size: int) -> None:
    """
    Verifica que el tamaño de ventana sea impar y mayor que cero.
    """
    if not isinstance(window_size, int):
        raise TypeError("window_size debe ser un entero.")
    if window_size <= 0:
        raise ValueError("window_size debe ser mayor que cero.")
    if window_size % 2 == 0:
        raise ValueError("window_size debe ser impar.")


def cargar_datos(points_path: Path, excel_path: Path | None = None) -> tuple[gpd.GeoDataFrame, pd.DataFrame | None]:
    """
    Carga el shapefile de puntos y, si está disponible, el archivo Excel de laboratorio.

    Si el Excel no existe, se asume que los datos de laboratorio o atributos
    requeridos ya están incorporados en el shapefile de puntos.
    """
    gdf = gpd.read_file(points_path)

    if excel_path is not None and excel_path.exists():
        df_excel = pd.read_excel(excel_path)
        print("Archivo Excel encontrado. Se realizará unión con los atributos del shapefile.")
    else:
        df_excel = None
        print(
            "Archivo Excel no encontrado. "
            "Se asumirá que los atributos requeridos ya están incluidos en el shapefile de puntos."
        )

    return gdf, df_excel


def validar_geometria_puntos(gdf: gpd.GeoDataFrame) -> None:
    """
    Verifica que el GeoDataFrame tenga geometría válida de puntos.
    """
    if gdf.empty:
        raise ValueError("El shapefile de puntos está vacío.")

    if "geometry" not in gdf.columns:
        raise ValueError("El shapefile no contiene columna de geometría.")

    tipos_validos = gdf.geometry.geom_type.isin(["Point", "MultiPoint"])
    if not tipos_validos.all():
        raise ValueError("Todas las geometrías del shapefile deben ser de tipo Point o MultiPoint.")


def detectar_columna_id(gdf: gpd.GeoDataFrame, df_excel: pd.DataFrame) -> tuple[str, str]:
    """
    Detecta la columna de ID compatible entre el shapefile y el Excel.
    """
    if df_excel.empty:
        raise ValueError("El archivo Excel está vacío.")

    excel_id = df_excel.columns[0]

    if excel_id in gdf.columns:
        merge_col = excel_id
    elif "ID" in gdf.columns:
        merge_col = "ID"
    elif "Name" in gdf.columns:
        merge_col = "Name"
    else:
        raise ValueError(
            "No se encontró una columna de ID coincidente entre shapefile y Excel. "
            "Verifique la primera columna del Excel y las columnas del shapefile."
        )

    return merge_col, excel_id


def obtener_candidate_predictors():
    """
    Define las bandas e índices espectrales a calcular.
    """
    candidate_predictors = [
        ("Blue",      lambda a: a[:, 0]),
        ("Green",     lambda a: a[:, 1]),
        ("Pan",       lambda a: a[:, 2]),
        ("Red",       lambda a: a[:, 3]),
        ("RedEdge",   lambda a: a[:, 4]),
        ("NIR",       lambda a: a[:, 5]),
        ("NDVI",      lambda a: (a[:, 5] - a[:, 3]) / np.where((a[:, 5] + a[:, 3]) == 0, np.nan, a[:, 5] + a[:, 3])),
        ("NDWI",      lambda a: (a[:, 1] - a[:, 5]) / np.where((a[:, 1] + a[:, 5]) == 0, np.nan, a[:, 1] + a[:, 5])),
        ("EVI",       lambda a: 2.5 * (a[:, 5] - a[:, 3]) / np.where((a[:, 5] + 6 * a[:, 3] - 7.5 * a[:, 0] + 1) == 0, np.nan, a[:, 5] + 6 * a[:, 3] - 7.5 * a[:, 0] + 1)),
        ("SAVI",      lambda a: (1.5 * (a[:, 5] - a[:, 3])) / np.where((a[:, 5] + a[:, 3] + 0.5) == 0, np.nan, a[:, 5] + a[:, 3] + 0.5)),
        ("GNDVI",     lambda a: (a[:, 5] - a[:, 1]) / np.where((a[:, 5] + a[:, 1]) == 0, np.nan, a[:, 5] + a[:, 1])),
        ("VARI",      lambda a: (a[:, 1] - a[:, 3]) / np.where((a[:, 1] + a[:, 3] - a[:, 0]) == 0, np.nan, a[:, 1] + a[:, 3] - a[:, 0])),
        ("NDRE",      lambda a: (a[:, 5] - a[:, 4]) / np.where((a[:, 5] + a[:, 4]) == 0, np.nan, a[:, 5] + a[:, 4])),
        ("CIgreen",   lambda a: (a[:, 5] / np.where(a[:, 1] == 0, np.nan, a[:, 1])) - 1),
        ("CIRE",      lambda a: (a[:, 5] / np.where(a[:, 4] == 0, np.nan, a[:, 4])) - 1),
        ("ARI",       lambda a: (1 / np.where(a[:, 1] == 0, np.nan, a[:, 1])) - (1 / np.where(a[:, 4] == 0, np.nan, a[:, 4]))),
        ("RENDVI",    lambda a: (a[:, 5] - a[:, 4]) / np.where((a[:, 5] + a[:, 4]) == 0, np.nan, a[:, 5] + a[:, 4])),
    ]
    return candidate_predictors


def validar_bandas_raster(src, band_names: list[str]) -> list[str]:
    """
    Verifica la cantidad de bandas del raster y ajusta automáticamente los nombres si es necesario.
    """
    num_bandas = src.count

    if num_bandas < 6:
        raise ValueError(
            f"El raster tiene {num_bandas} bandas. Se requieren al menos 6 bandas para calcular "
            "las reflectancias base e índices definidos."
        )

    if len(band_names) != num_bandas:
        band_names = [f"Banda_{i+1}" for i in range(num_bandas)]
        print("⚠️ La cantidad de nombres de banda no coincide con el raster.")
        print("   Se ajustaron nombres de banda automáticamente.")

    return band_names


def reproyectar_puntos_si_es_necesario(gdf: gpd.GeoDataFrame, raster_crs) -> gpd.GeoDataFrame:
    """
    Reproyecta el shapefile de puntos al CRS del raster si es necesario.
    """
    if gdf.crs is None:
        raise ValueError("El shapefile de puntos no tiene CRS definido.")

    if raster_crs is None:
        raise ValueError("El raster no tiene CRS definido.")

    if gdf.crs != raster_crs:
        print("⚠️ El CRS de los puntos no coincide con el del raster.")
        print("   Se reproyectarán los puntos al CRS del raster.")
        gdf = gdf.to_crs(raster_crs)

    return gdf


def extraer_valores_por_punto(gdf: gpd.GeoDataFrame, src, window_size: int, candidate_predictors: list) -> dict:
    """
    Extrae reflectancias promedio e índices espectrales para cada punto.
    """
    offset = window_size // 2
    resultados = {label: [] for label, _ in candidate_predictors}

    for geom in tqdm(gdf.geometry, total=len(gdf), desc="Extrayendo reflectancias e índices"):
        if geom is None or geom.is_empty:
            for label in resultados.keys():
                resultados[label].append(np.nan)
            continue

        x, y = geom.x, geom.y
        row, col = src.index(x, y)

        window = Window(col - offset, row - offset, window_size, window_size)

        arr = np.array([
            src.read(b, window=window, boundless=True, fill_value=np.nan).flatten()
            for b in range(1, src.count + 1)
        ]).T

        for label, func in candidate_predictors:
            try:
                val = np.nanmean(func(arr))
            except Exception:
                val = np.nan
            resultados[label].append(val)

    return resultados


def agregar_resultados_al_gdf(gdf: gpd.GeoDataFrame, resultados: dict) -> gpd.GeoDataFrame:
    """
    Agrega al GeoDataFrame las columnas calculadas de reflectancia e índices.
    """

    print("\nColumnas del GeoDataFrame antes de guardar:")
    print(gdf.columns.tolist())
    
    for label, values in resultados.items():
        gdf[label] = values
    return gdf


def unir_con_excel(gdf: gpd.GeoDataFrame, df_excel: pd.DataFrame, merge_col: str, excel_id: str) -> gpd.GeoDataFrame:
    """
    Une los resultados espaciales con los datos del Excel.
    """
    gdf = gdf.merge(df_excel, left_on=merge_col, right_on=excel_id, how="left")
    return gdf

'''
def guardar_resultado(gdf: gpd.GeoDataFrame, output_path: Path) -> None:
    """
    Guarda el archivo espacial de salida.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".gpkg":
        if output_path.exists():
            output_path.unlink()

        gdf.to_file(output_path, driver="GPKG", layer="puntos_reflectancia_indices")

    else:
        gdf.to_file(output_path)

    print("\n✅ Archivo generado con éxito:")
    print(f"   {output_path}")
    print("   Contiene reflectancias promedio, índices espectrales y parámetros de calidad de agua.")
'''

def guardar_resultado(gdf: gpd.GeoDataFrame, output_path: Path) -> None:
    """
    Guarda el archivo espacial de salida.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".gpkg":
        if output_path.exists():
            output_path.unlink()

        gdf.to_file(
            output_path,
            driver="GPKG",
            layer="puntos_reflectancia_indices"
        )

    else:
        gdf.to_file(output_path)

    print("\n✅ Archivo generado con éxito:")
    print(f"   {output_path}")
    print("   Contiene reflectancias promedio, índices espectrales y parámetros de calidad de agua.")


# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================

def main():
    print("Cargando archivos...")

    validar_archivo(raster_path, "el raster multibanda")
    validar_archivo(points_path, "el shapefile de puntos")
    validar_window_size(window_size)

    gdf, df_excel = cargar_datos(points_path, excel_path)
    validar_geometria_puntos(gdf)

    if df_excel is not None:
        merge_col, excel_id = detectar_columna_id(gdf, df_excel)
    else:
        merge_col, excel_id = None, None

    candidate_predictors = obtener_candidate_predictors()

    with rasterio.open(raster_path) as src:
        band_names_ajustadas = validar_bandas_raster(src, band_names)
        _ = band_names_ajustadas

        print(f"Raster con {src.count} bandas detectadas.")
        print(f"Calculando reflectancia promedio e índices (ventana {window_size}x{window_size})...")

        gdf = reproyectar_puntos_si_es_necesario(gdf, src.crs)

        resultados = extraer_valores_por_punto(
            gdf=gdf,
            src=src,
            window_size=window_size,
            candidate_predictors=candidate_predictors
        )

    gdf = agregar_resultados_al_gdf(gdf, resultados)

    if df_excel is not None:
        print("\nUniendo datos del Excel de laboratorio con los puntos...")
        gdf = unir_con_excel(gdf, df_excel, merge_col, excel_id)
    else:
        print(
            "\nNo se realizó unión con Excel. "
            "Se conservarán los atributos originales del shapefile de puntos junto con las reflectancias e índices calculados."
        )

    guardar_resultado(gdf, output_path)

if __name__ == "__main__":
    main()