from pathlib import Path
import json

import numpy as np
import pandas as pd
import rasterio


# ============================================================
# DEFINICIÓN DE VARIABLES
# ============================================================

def obtener_feature_names_clasificacion() -> list[str]:
    """
    Retorna los nombres esperados de bandas, índices y texturas del stack Sentinel-2.
    """
    return [
        # Bandas Sentinel-2
        "B2", "B3", "B4", "B8",
        "B5", "B6", "B7", "B8A", "B11", "B12",

        # Índices espectrales
        "NDVI", "EVI", "SAVI", "MNDWI", "NDMI", "NDBI",

        # Texturas GLCM B2
        "TEX_B2_contrast", "TEX_B2_diss", "TEX_B2_ent", "TEX_B2_asm",
        "TEX_B2_idm", "TEX_B2_corr", "TEX_B2_var", "TEX_B2_sent",

        # Texturas GLCM B3
        "TEX_B3_contrast", "TEX_B3_diss", "TEX_B3_ent", "TEX_B3_asm",
        "TEX_B3_idm", "TEX_B3_corr", "TEX_B3_var", "TEX_B3_sent",

        # Texturas GLCM B4
        "TEX_B4_contrast", "TEX_B4_diss", "TEX_B4_ent", "TEX_B4_asm",
        "TEX_B4_idm", "TEX_B4_corr", "TEX_B4_var", "TEX_B4_sent",

        # Texturas GLCM B8
        "TEX_B8_contrast", "TEX_B8_diss", "TEX_B8_ent", "TEX_B8_asm",
        "TEX_B8_idm", "TEX_B8_corr", "TEX_B8_var", "TEX_B8_sent",
    ]


def obtener_grupos_bandas() -> tuple[dict, dict]:
    """
    Retorna diccionarios de grupos y colores para variables del stack.
    """
    feature_names = obtener_feature_names_clasificacion()

    grupo_bandas = {}

    for name in feature_names:
        if name.startswith("TEX_"):
            grupo_bandas[name] = "S2 GLCM textures"
        elif name in ["NDVI", "EVI", "SAVI", "MNDWI", "NDMI", "NDBI"]:
            grupo_bandas[name] = "S2 VIs"
        else:
            grupo_bandas[name] = "S2 bands"

    colores_grupo = {
        "S2 bands": "red",
        "S2 VIs": "orange",
        "S2 GLCM textures": "purple"
    }

    return grupo_bandas, colores_grupo


def resolver_nombres_bandas(
    n_bandas: int,
    feature_names: list[str] | None = None
) -> list[str]:
    """
    Retorna nombres reales de bandas si coinciden con el raster; de lo contrario,
    genera nombres genéricos b1, b2, ...
    """
    if feature_names is not None and len(feature_names) == n_bandas:
        return list(feature_names)

    return [f"b{i + 1}" for i in range(n_bandas)]


# ============================================================
# LECTURA Y CONVERSIÓN DE RASTER
# ============================================================

def cargar_raster_multibanda(ruta_raster: Path) -> tuple[np.ndarray, dict]:
    """
    Carga un raster multibanda como arreglo NumPy y retorna también su perfil.
    """
    ruta_raster = Path(ruta_raster)

    if not ruta_raster.exists():
        raise FileNotFoundError(f"No se encontró el raster: {ruta_raster}")

    with rasterio.open(ruta_raster) as src:
        arr = src.read().astype(np.float32)
        profile = src.profile.copy()

    return arr, profile


def raster_a_matriz_valida(
    arr: np.ndarray,
    max_pixeles: int = 100_000,
    random_state: int = 42,
    nodata_val: float | int | None = None,
    excluir_fondo_cero: bool = True
) -> np.ndarray:
    """
    Convierte un raster (bandas, filas, columnas) en matriz tabular
    (píxeles, bandas) y toma una muestra de píxeles válidos.

    Se excluyen:
    - filas con NaN o infinitos;
    - filas con NoData;
    - filas completamente iguales a cero, si excluir_fondo_cero=True.

    Esta depuración evita que el fondo del raster recortado domine el cálculo
    de correlación.
    """
    if arr.ndim != 3:
        raise ValueError("El raster debe tener forma (bandas, filas, columnas).")

    X = arr.reshape(arr.shape[0], -1).T

    mask_valid = np.isfinite(X).all(axis=1)

    if nodata_val is not None:
        mask_valid &= ~(X == nodata_val).any(axis=1)

    if excluir_fondo_cero:
        mask_valid &= ~(np.all(X == 0, axis=1))

    X_valid = X[mask_valid]

    if X_valid.shape[0] == 0:
        raise ValueError("No se encontraron píxeles válidos en el raster.")

    n = min(max_pixeles, X_valid.shape[0])

    rng = np.random.default_rng(random_state)
    idx = rng.choice(X_valid.shape[0], n, replace=False)

    print("Píxeles totales:", X.shape[0])
    print("Píxeles válidos usados para correlación:", X_valid.shape[0])
    print("Píxeles muestreados:", n)

    return X_valid[idx]


def crear_dataframe_muestra(
    arr: np.ndarray,
    feature_names: list[str] | None = None,
    max_pixeles: int = 100_000,
    random_state: int = 42,
    nodata_val: float | int | None = None,
    excluir_fondo_cero: bool = True
) -> pd.DataFrame:
    """
    Crea un DataFrame de muestra a partir de un raster multibanda.
    """
    Xs = raster_a_matriz_valida(
        arr=arr,
        max_pixeles=max_pixeles,
        random_state=random_state,
        nodata_val=nodata_val,
        excluir_fondo_cero=excluir_fondo_cero
    )

    band_names = resolver_nombres_bandas(
        n_bandas=Xs.shape[1],
        feature_names=feature_names
    )

    return pd.DataFrame(Xs, columns=band_names)


# ============================================================
# CORRELACIÓN Y SELECCIÓN DE VARIABLES
# ============================================================

def calcular_correlaciones(
    df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calcula matrices de correlación Pearson y Spearman.
    """
    corr_pearson = df.corr(method="pearson")
    corr_spearman = df.corr(method="spearman")

    return corr_pearson, corr_spearman


def obtener_pares_alta_correlacion(
    corr: pd.DataFrame,
    threshold: float = 0.90
) -> pd.DataFrame:
    """
    Retorna pares de variables con correlación absoluta mayor o igual al umbral.
    """
    pares = (
        corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        .stack()
        .reset_index()
    )

    pares.columns = ["band1", "band2", "r"]

    pares = pares[pares["r"].abs() >= threshold]
    pares = pares.sort_values("r", key=np.abs, ascending=False)

    return pares.reset_index(drop=True)


def seleccionar_variables_por_correlacion(
    corr: pd.DataFrame,
    threshold: float = 0.95
) -> tuple[list[str], list[str]]:
    """
    Selecciona variables eliminando columnas con correlación absoluta mayor
    que el umbral en la parte superior de la matriz.

    Returns
    -------
    variables_finales : list[str]
        Variables seleccionadas.

    variables_eliminadas : list[str]
        Variables eliminadas por alta correlación.
    """
    corr_abs = corr.abs()

    upper = corr_abs.where(
        np.triu(np.ones(corr_abs.shape), k=1).astype(bool)
    )

    variables_eliminadas = [
        col for col in upper.columns
        if any(upper[col] > threshold)
    ]

    variables_finales = [
        col for col in corr.columns
        if col not in variables_eliminadas
    ]

    return variables_finales, variables_eliminadas


def guardar_dataframe(
    df: pd.DataFrame,
    ruta_salida: Path
) -> None:
    """
    Guarda un DataFrame en CSV.
    """
    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(ruta_salida, index=False, encoding="utf-8-sig")

    print("Tabla guardada en:")
    print(ruta_salida)


def guardar_lista_json(
    valores: list,
    ruta_salida: Path
) -> None:
    """
    Guarda una lista en formato JSON.
    """
    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    with open(ruta_salida, "w", encoding="utf-8") as f:
        json.dump(valores, f, indent=4, ensure_ascii=False)

    print("Archivo JSON guardado en:")
    print(ruta_salida)


def cargar_lista_json(ruta_json: Path) -> list:
    """
    Carga una lista desde un archivo JSON.
    """
    ruta_json = Path(ruta_json)

    if not ruta_json.exists():
        raise FileNotFoundError(f"No se encontró el archivo JSON: {ruta_json}")

    with open(ruta_json, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# RASTER REDUCIDO
# ============================================================

def generar_raster_reducido(
    ruta_raster: Path,
    variables_finales: list[str],
    band_names: list[str],
    ruta_salida_tif: Path,
    ruta_salida_json: Path | None = None
) -> tuple[np.ndarray, dict, list[int]]:
    """
    Genera un raster reducido usando una lista de variables seleccionadas.

    Parameters
    ----------
    ruta_raster : Path
        Raster multibanda original.

    variables_finales : list[str]
        Variables que se conservarán.

    band_names : list[str]
        Nombres de bandas del raster original.

    ruta_salida_tif : Path
        Ruta del GeoTIFF reducido.

    ruta_salida_json : Path | None
        Ruta para guardar la lista de variables finales.

    Returns
    -------
    arr_reducido : np.ndarray
        Raster reducido.

    profile : dict
        Perfil actualizado.

    indices : list[int]
        Índices de bandas seleccionadas.
    """
    arr, profile = cargar_raster_multibanda(ruta_raster)

    faltantes = [v for v in variables_finales if v not in band_names]

    if faltantes:
        raise ValueError(
            "Las siguientes variables no existen en band_names: "
            + ", ".join(faltantes)
        )

    indices = [band_names.index(v) for v in variables_finales]

    arr_reducido = arr[indices, :, :]

    profile.update(count=len(indices))

    ruta_salida_tif = Path(ruta_salida_tif)
    ruta_salida_tif.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(ruta_salida_tif, "w", **profile) as dst:
        dst.write(arr_reducido)

    print("Raster reducido guardado en:")
    print(ruta_salida_tif)

    if ruta_salida_json is not None:
        guardar_lista_json(
            valores=variables_finales,
            ruta_salida=ruta_salida_json
        )

    return arr_reducido, profile, indices


# ============================================================
# PREPROCESAMIENTO PARA MACHINE LEARNING
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


def preprocesar_raster(
    ruta_raster: Path,
    usar_scaler: bool = True
) -> dict:
    """
    Preprocesa un raster para uso en modelos de machine learning.

    Pasos:
    - lectura del raster;
    - normalización min-max por banda;
    - reorganización a matriz píxeles × variables;
    - escalamiento opcional con StandardScaler.
    """
    from sklearn.preprocessing import StandardScaler

    ruta_raster = Path(ruta_raster)

    if not ruta_raster.exists():
        raise FileNotFoundError(f"No se encontró el raster: {ruta_raster}")

    with rasterio.open(ruta_raster) as src:
        arr = src.read().astype(float)
        profile = src.profile.copy()
        n_bandas, filas, columnas = arr.shape

    arr_norm = np.array([
        minmax_normalize(arr[i])
        for i in range(n_bandas)
    ])

    x = np.moveaxis(arr_norm, 0, -1)
    X_data = x.reshape(-1, n_bandas)

    if usar_scaler:
        scaler = StandardScaler().fit(X_data)
        X = scaler.transform(X_data)
    else:
        scaler = None
        X = X_data

    return {
        "arr_norm": arr_norm,
        "X": X,
        "n_bandas": n_bandas,
        "filas": filas,
        "columnas": columnas,
        "scaler": scaler,
        "profile": profile
    }


# ============================================================
# PIPELINE DE PREPARACIÓN DE RASTER
# ============================================================

def pipeline_preparacion_raster(
    ruta_raster: Path,
    feature_names: list[str],
    carpeta_tablas: Path,
    carpeta_figuras: Path,
    ruta_raster_reducido: Path,
    ruta_bandas_json: Path,
    max_pixeles: int = 100_000,
    random_state: int = 42,
    threshold_pares: float = 0.90,
    threshold_seleccion: float = 0.95,
    nodata_val: float | int | None = -9999,
    excluir_fondo_cero: bool = True
) -> dict:
    """
    Ejecuta la preparación del raster:

    - carga raster;
    - crea muestra tabular;
    - calcula Pearson y Spearman;
    - detecta pares altamente correlacionados;
    - selecciona variables por correlación;
    - guarda tablas;
    - genera raster reducido.
    """
    arr, profile = cargar_raster_multibanda(ruta_raster)

    band_names = resolver_nombres_bandas(
        n_bandas=arr.shape[0],
        feature_names=feature_names
    )

    nodata_raster = profile.get("nodata", None)
    
    if nodata_val is None:
        nodata_usado = nodata_raster
    else:
        nodata_usado = nodata_val
    
    df_muestra = crear_dataframe_muestra(
        arr=arr,
        feature_names=band_names,
        max_pixeles=max_pixeles,
        random_state=random_state,
        nodata_val=nodata_usado,
        excluir_fondo_cero=excluir_fondo_cero
    )

    corr_pearson, corr_spearman = calcular_correlaciones(df_muestra)

    pares_pearson = obtener_pares_alta_correlacion(
        corr=corr_pearson,
        threshold=threshold_pares
    )

    pares_spearman = obtener_pares_alta_correlacion(
        corr=corr_spearman,
        threshold=threshold_pares
    )

    variables_finales, variables_eliminadas = seleccionar_variables_por_correlacion(
        corr=corr_pearson,
        threshold=threshold_seleccion
    )

    carpeta_tablas = Path(carpeta_tablas)
    carpeta_tablas.mkdir(parents=True, exist_ok=True)

    guardar_dataframe(
        pares_pearson,
        carpeta_tablas / "pares_correlacion_pearson.csv"
    )

    guardar_dataframe(
        pares_spearman,
        carpeta_tablas / "pares_correlacion_spearman.csv"
    )

    df_variables = pd.DataFrame({
        "variable": band_names,
        "seleccionada": [v in variables_finales for v in band_names],
        "eliminada": [v in variables_eliminadas for v in band_names]
    })

    guardar_dataframe(
        df_variables,
        carpeta_tablas / "variables_seleccionadas_pearson.csv"
    )

    arr_reducido, profile_reducido, indices = generar_raster_reducido(
        ruta_raster=ruta_raster,
        variables_finales=variables_finales,
        band_names=band_names,
        ruta_salida_tif=ruta_raster_reducido,
        ruta_salida_json=ruta_bandas_json
    )

    return {
        "arr": arr,
        "profile": profile,
        "band_names": band_names,
        "df_muestra": df_muestra,
        "corr_pearson": corr_pearson,
        "corr_spearman": corr_spearman,
        "pares_pearson": pares_pearson,
        "pares_spearman": pares_spearman,
        "variables_finales": variables_finales,
        "variables_eliminadas": variables_eliminadas,
        "df_variables": df_variables,
        "arr_reducido": arr_reducido,
        "profile_reducido": profile_reducido,
        "indices": indices
    }
