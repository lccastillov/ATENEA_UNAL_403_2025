from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import joblib

from sklearn.model_selection import train_test_split


# ============================================================
# CARGA Y VALIDACIÓN DE PUNTOS
# ============================================================

def cargar_puntos_entrenamiento(
    ruta_puntos: Path,
    columna_clase: str = "Clase_N1",
    columna_clase_nombre: str = "d_nivel_1_"
) -> gpd.GeoDataFrame:
    """
    Carga los puntos de entrenamiento y valida las columnas de clase.

    Parameters
    ----------
    ruta_puntos : Path
        Ruta del archivo vectorial con puntos de entrenamiento.

    columna_clase : str
        Campo con el código numérico de clase.

    columna_clase_nombre : str
        Campo con la etiqueta descriptiva de clase.
    """
    ruta_puntos = Path(ruta_puntos)

    if not ruta_puntos.exists():
        raise FileNotFoundError(f"No se encontró la capa de puntos: {ruta_puntos}")

    gdf = gpd.read_file(ruta_puntos)

    if gdf.empty:
        raise ValueError("La capa de puntos de entrenamiento está vacía.")

    if gdf.crs is None:
        raise ValueError("La capa de puntos de entrenamiento no tiene CRS definido.")

    columnas_requeridas = [columna_clase, columna_clase_nombre]

    faltantes = [col for col in columnas_requeridas if col not in gdf.columns]

    if faltantes:
        raise ValueError(
            "La capa de puntos no contiene las siguientes columnas requeridas: "
            f"{faltantes}"
        )

    return gdf


def reproyectar_puntos_a_raster(
    gdf_puntos: gpd.GeoDataFrame,
    ruta_raster: Path
) -> gpd.GeoDataFrame:
    """
    Reproyecta los puntos al CRS del raster si es necesario.
    """
    ruta_raster = Path(ruta_raster)

    if not ruta_raster.exists():
        raise FileNotFoundError(f"No se encontró el raster: {ruta_raster}")

    with rasterio.open(ruta_raster) as src:
        raster_crs = src.crs

    if gdf_puntos.crs != raster_crs:
        print(f"Reproyectando puntos de {gdf_puntos.crs} a {raster_crs}...")
        gdf_puntos = gdf_puntos.to_crs(raster_crs)

    return gdf_puntos


# ============================================================
# DISTRIBUCIÓN DE CLASES
# ============================================================

def contar_clases(
    gdf_puntos: gpd.GeoDataFrame,
    columna_clase: str = "Clase_N1",
    columna_clase_nombre: str = "d_nivel_1_"
) -> pd.DataFrame:
    """
    Genera una tabla de conteo de puntos por clase.
    """
    tabla = (
        gdf_puntos
        .groupby(columna_clase_nombre)
        .agg(
            conteo=(columna_clase_nombre, "size"),
            codigo_clase=(columna_clase, "mean")
        )
        .reset_index()
        .sort_values("conteo", ascending=False)
    )

    return tabla


def graficar_distribucion_clases(
    tabla_clases: pd.DataFrame,
    ruta_figura: Path | None = None,
    columna_nombre: str = "d_nivel_1_",
    columna_conteo: str = "conteo",
    titulo: str = "Distribución de puntos por clase",
    dpi: int = 300
) -> None:
    """
    Grafica la distribución de puntos por clase.
    """
    import matplotlib.pyplot as plt

    tabla_plot = tabla_clases.sort_values(columna_conteo, ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(5, 0.45 * len(tabla_plot))))

    ax.barh(
        tabla_plot[columna_nombre],
        tabla_plot[columna_conteo]
    )

    ax.set_xlabel("Número de puntos")
    ax.set_ylabel("Clase")
    ax.set_title(titulo, fontweight="bold")

    for i, valor in enumerate(tabla_plot[columna_conteo]):
        ax.text(valor, i, f" {valor}", va="center", fontsize=9)

    plt.tight_layout()

    if ruta_figura is not None:
        ruta_figura = Path(ruta_figura)
        ruta_figura.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(ruta_figura, dpi=dpi, bbox_inches="tight")
        print("Figura guardada en:")
        print(ruta_figura)

    plt.show()


# ============================================================
# NORMALIZACIÓN Y EXTRACCIÓN DE VALORES
# ============================================================

def minmax_normalize(
    arr: np.ndarray
) -> np.ndarray:
    """
    Normaliza un arreglo mediante min-max.
    """
    arr = arr.astype(float)

    min_val = np.nanmin(arr)
    max_val = np.nanmax(arr)

    return (arr - min_val) / (max_val - min_val + 1e-10)


def cargar_raster_normalizado(
    ruta_raster: Path,
    normalizar: bool = True
) -> dict:
    """
    Carga un raster multibanda y opcionalmente normaliza cada banda con min-max.

    Returns
    -------
    dict
        Diccionario con raster normalizado, perfil y metadatos de forma.
    """
    ruta_raster = Path(ruta_raster)

    if not ruta_raster.exists():
        raise FileNotFoundError(f"No se encontró el raster: {ruta_raster}")

    with rasterio.open(ruta_raster) as src:
        arr = src.read().astype(float)
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
        nodata = src.nodata

    if normalizar:
        arr_proc = np.array([
            minmax_normalize(arr[i])
            for i in range(arr.shape[0])
        ])
    else:
        arr_proc = arr

    x_image = np.moveaxis(arr_proc, 0, -1)

    return {
        "arr": arr,
        "arr_proc": arr_proc,
        "x_image": x_image,
        "profile": profile,
        "transform": transform,
        "crs": crs,
        "nodata": nodata,
        "n_bandas": arr.shape[0],
        "filas": arr.shape[1],
        "columnas": arr.shape[2]
    }


def extraer_valores_raster_en_puntos(
    ruta_raster: Path,
    gdf_puntos: gpd.GeoDataFrame,
    columna_clase: str = "Clase_N1",
    variables_modelo: list[str] | None = None,
    normalizar: bool = True
) -> tuple[np.ndarray, np.ndarray, gpd.GeoDataFrame, pd.DataFrame]:
    """
    Extrae valores del raster en los puntos de entrenamiento.

    Para cada punto se obtiene el valor de todas las bandas del raster en el píxel
    correspondiente.

    Returns
    -------
    X_points : np.ndarray
        Matriz de variables predictoras.

    y_points : np.ndarray
        Vector de clases.

    gdf_validos : gpd.GeoDataFrame
        Puntos que pudieron asociarse a píxeles válidos.

    df_muestras : pd.DataFrame
        Tabla con variables extraídas y clase.
    """
    info = cargar_raster_normalizado(
        ruta_raster=ruta_raster,
        normalizar=normalizar
    )

    gdf_puntos = reproyectar_puntos_a_raster(
        gdf_puntos=gdf_puntos,
        ruta_raster=ruta_raster
    )

    x_image = info["x_image"]
    transform = info["transform"]

    xs = gdf_puntos.geometry.x.values
    ys = gdf_puntos.geometry.y.values

    rows, cols = rasterio.transform.rowcol(transform, xs, ys)

    X_points = []
    valid_idx = []

    for i, (row, col) in enumerate(zip(rows, cols)):
        if 0 <= row < x_image.shape[0] and 0 <= col < x_image.shape[1]:
            valores = x_image[row, col, :]

            if np.all(np.isfinite(valores)):
                X_points.append(valores)
                valid_idx.append(i)

    if not X_points:
        raise ValueError("No se extrajeron puntos válidos del raster.")

    X_points = np.vstack(X_points)
    gdf_validos = gdf_puntos.iloc[valid_idx].copy()
    y_points = gdf_validos[columna_clase].to_numpy()

    if variables_modelo is None:
        variables_modelo = [f"b{i + 1}" for i in range(X_points.shape[1])]

    if len(variables_modelo) != X_points.shape[1]:
        variables_modelo = [f"b{i + 1}" for i in range(X_points.shape[1])]

    df_muestras = pd.DataFrame(
        X_points,
        columns=variables_modelo
    )

    df_muestras[columna_clase] = y_points

    print("Puntos originales:", len(gdf_puntos))
    print("Puntos válidos:", len(gdf_validos))
    print("Puntos descartados:", len(gdf_puntos) - len(gdf_validos))
    print("Shape X_points:", X_points.shape)
    print("Shape y_points:", y_points.shape)

    return X_points, y_points, gdf_validos, df_muestras


def guardar_muestras(
    df_muestras: pd.DataFrame,
    ruta_salida: Path
) -> None:
    """
    Guarda la tabla de muestras extraídas en CSV.
    """
    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    df_muestras.to_csv(
        ruta_salida,
        index=False,
        encoding="utf-8-sig"
    )

    print("Muestras guardadas en:")
    print(ruta_salida)


# ============================================================
# SPLIT ESTRATIFICADO
# ============================================================

def crear_split_estratificado(
    X_points: np.ndarray,
    y_points: np.ndarray,
    test_size: float = 0.3,
    random_state: int = 42
):
    """
    Crea un split entrenamiento/prueba estratificado por clase.
    """
    return train_test_split(
        X_points,
        y_points,
        test_size=test_size,
        stratify=y_points,
        random_state=random_state
    )


def guardar_split_clasificacion(
    ruta_split: Path,
    X_train,
    X_test,
    y_train,
    y_test,
    variables_modelo: list[str],
    columna_clase: str,
    columna_clase_nombre: str,
    ruta_raster: Path,
    ruta_puntos: Path,
    ruta_muestras: Path,
    test_size: float,
    random_state: int,
    normalizar: bool
) -> None:
    """
    Guarda el split de clasificación y metadatos asociados.
    """
    ruta_split = Path(ruta_split)
    ruta_split.parent.mkdir(parents=True, exist_ok=True)

    split_data = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "variables_modelo": variables_modelo,
        "columna_clase": columna_clase,
        "columna_clase_nombre": columna_clase_nombre,
        "ruta_raster": str(ruta_raster),
        "ruta_puntos": str(ruta_puntos),
        "ruta_muestras": str(ruta_muestras),
        "test_size": test_size,
        "random_state": random_state,
        "normalizar": normalizar,
        "clases": sorted(np.unique(y_train).tolist())
    }

    joblib.dump(split_data, ruta_split)

    print("Split guardado en:")
    print(ruta_split)


def cargar_split_clasificacion(
    ruta_split: Path
) -> dict:
    """
    Carga un split de clasificación guardado en formato joblib.
    """
    ruta_split = Path(ruta_split)

    if not ruta_split.exists():
        raise FileNotFoundError(f"No se encontró el split: {ruta_split}")

    return joblib.load(ruta_split)


def preparar_muestras_y_split(
    ruta_raster: Path,
    ruta_puntos: Path,
    ruta_muestras: Path,
    ruta_split: Path,
    variables_modelo: list[str],
    columna_clase: str = "Clase_N1",
    columna_clase_nombre: str = "d_nivel_1_",
    normalizar: bool = True,
    test_size: float = 0.3,
    random_state: int = 42
) -> dict:
    """
    Ejecuta la preparación de muestras y genera el split estratificado.
    """
    gdf_puntos = cargar_puntos_entrenamiento(
        ruta_puntos=ruta_puntos,
        columna_clase=columna_clase,
        columna_clase_nombre=columna_clase_nombre
    )

    X_points, y_points, gdf_validos, df_muestras = extraer_valores_raster_en_puntos(
        ruta_raster=ruta_raster,
        gdf_puntos=gdf_puntos,
        columna_clase=columna_clase,
        variables_modelo=variables_modelo,
        normalizar=normalizar
    )

    guardar_muestras(
        df_muestras=df_muestras,
        ruta_salida=ruta_muestras
    )

    X_train, X_test, y_train, y_test = crear_split_estratificado(
        X_points=X_points,
        y_points=y_points,
        test_size=test_size,
        random_state=random_state
    )

    guardar_split_clasificacion(
        ruta_split=ruta_split,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        variables_modelo=variables_modelo,
        columna_clase=columna_clase,
        columna_clase_nombre=columna_clase_nombre,
        ruta_raster=ruta_raster,
        ruta_puntos=ruta_puntos,
        ruta_muestras=ruta_muestras,
        test_size=test_size,
        random_state=random_state,
        normalizar=normalizar
    )

    return {
        "gdf_puntos": gdf_puntos,
        "gdf_validos": gdf_validos,
        "X_points": X_points,
        "y_points": y_points,
        "df_muestras": df_muestras,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }
