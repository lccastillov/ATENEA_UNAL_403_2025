from pathlib import Path
import os
import tempfile

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
import joblib


# ============================================================
# CARGA DE MODELOS Y METADATOS
# ============================================================

def cargar_modelo(ruta_modelo: Path):
    """
    Carga un modelo entrenado en formato .joblib.
    """
    if not ruta_modelo.exists():
        raise FileNotFoundError(f"No se encontró el modelo: {ruta_modelo}")

    return joblib.load(ruta_modelo)


def cargar_modelos(rutas_modelos: dict) -> dict:
    """
    Carga múltiples modelos entrenados.

    Parameters
    ----------
    rutas_modelos : dict
        Diccionario con estructura:
        {
            "SVR": ruta_modelo_svr,
            "GBR": ruta_modelo_gbr,
            "RFR": ruta_modelo_rfr
        }

    Returns
    -------
    dict
        Diccionario de modelos cargados.
    """
    modelos = {}

    for nombre_modelo, ruta_modelo in rutas_modelos.items():
        modelos[nombre_modelo] = cargar_modelo(ruta_modelo)

    return modelos


def cargar_split(ruta_split: Path) -> dict:
    """
    Carga el archivo .joblib con la partición y metadatos del entrenamiento.
    """
    if not ruta_split.exists():
        raise FileNotFoundError(f"No se encontró el archivo de split: {ruta_split}")

    return joblib.load(ruta_split)


# ============================================================
# OPERACIONES AUXILIARES
# ============================================================

def safe_div(numerador, denominador):
    """
    Realiza una división segura evitando divisiones por cero.
    """
    return np.where(denominador != 0, numerador / denominador, np.nan)


def validar_raster_multibanda(ruta_raster: Path, min_bandas: int = 6) -> None:
    """
    Verifica que el raster exista y tenga el número mínimo de bandas requerido.
    """
    if not ruta_raster.exists():
        raise FileNotFoundError(f"No se encontró el raster: {ruta_raster}")

    with rasterio.open(ruta_raster) as src:
        if src.count < min_bandas:
            raise ValueError(
                f"El raster tiene {src.count} bandas. "
                f"Se requieren al menos {min_bandas} bandas."
            )


def leer_raster_multibanda(ruta_raster: Path):
    """
    Lee un raster multibanda y retorna los datos y el perfil espacial.
    """
    validar_raster_multibanda(ruta_raster)

    with rasterio.open(ruta_raster) as src:
        data = src.read()
        profile = src.profile.copy()

    return data, profile


# ============================================================
# CÁLCULO DE VARIABLES ESPECTRALES
# ============================================================

def separar_bandas_rededgep(data: np.ndarray) -> dict:
    """
    Separa las bandas de un raster MicaSense RedEdge-P.

    Orden esperado:
    1. Blue
    2. Green
    3. Pan
    4. Red
    5. RedEdge
    6. NIR
    """
    if data.shape[0] < 6:
        raise ValueError(
            f"El raster tiene {data.shape[0]} bandas. "
            "Se requieren al menos 6 bandas en el orden Blue, Green, Pan, Red, RedEdge, NIR."
        )

    bandas = {
        "Blue": data[0].astype("float32"),
        "Green": data[1].astype("float32"),
        "Pan": data[2].astype("float32"),
        "Red": data[3].astype("float32"),
        "RedEdge": data[4].astype("float32"),
        "NIR": data[5].astype("float32")
    }

    return bandas


def calcular_indices_espectrales(bandas: dict) -> dict:
    """
    Calcula índices espectrales a partir de las bandas del sensor RedEdge-P.
    """
    blue = bandas["Blue"]
    green = bandas["Green"]
    red = bandas["Red"]
    rededge = bandas["RedEdge"]
    nir = bandas["NIR"]

    indices = {
        "NDVI": safe_div((nir - red), (nir + red)),
        "NDWI": safe_div((green - nir), (green + nir)),
        "EVI": safe_div(2.5 * (nir - red), (nir + 6 * red - 7.5 * blue + 1)),
        "SAVI": safe_div(1.5 * (nir - red), (nir + red + 0.5)),
        "GNDVI": safe_div((nir - green), (nir + green)),
        "VARI": safe_div((green - red), (green + red - blue)),
        "NDRE": safe_div((nir - rededge), (nir + rededge)),
        "CIgreen": safe_div(nir, green) - 1,
        "CIRE": safe_div(nir, rededge) - 1,
        "ARI": safe_div(1, green) - safe_div(1, rededge),
        "RENDVI": safe_div((nir - rededge), (nir + rededge))
    }

    return indices


def construir_variables_disponibles(data: np.ndarray) -> dict:
    """
    Construye el diccionario completo de bandas e índices disponibles.
    """
    bandas = separar_bandas_rededgep(data)
    indices = calcular_indices_espectrales(bandas)

    variables_disponibles = {}
    variables_disponibles.update(bandas)
    variables_disponibles.update(indices)

    return variables_disponibles


def validar_variables_requeridas(variables_disponibles: dict, vars_pred: list[str]) -> None:
    """
    Verifica que todas las variables usadas en entrenamiento estén disponibles para predicción.
    """
    variables_faltantes = [var for var in vars_pred if var not in variables_disponibles]

    if variables_faltantes:
        raise ValueError(
            "Las siguientes variables usadas en el entrenamiento no están disponibles "
            f"para la predicción raster: {variables_faltantes}"
        )


def construir_stack_predictivo(
    variables_disponibles: dict,
    vars_pred: list[str]
) -> np.ndarray:
    """
    Construye el stack predictivo en el mismo orden de variables usado durante el entrenamiento.
    """
    validar_variables_requeridas(
        variables_disponibles=variables_disponibles,
        vars_pred=vars_pred
    )

    stack_final = np.stack(
        [variables_disponibles[var] for var in vars_pred],
        axis=0
    )

    return stack_final


# ============================================================
# PREDICCIÓN
# ============================================================

def preparar_matriz_prediccion(stack_final: np.ndarray):
    """
    Convierte el stack raster a matriz 2D para entrada de los modelos.

    Returns
    -------
    matriz : np.ndarray
        Matriz completa de predicción con forma (n_pixeles, n_variables).

    matriz_valida : np.ndarray
        Matriz solo con píxeles válidos.

    mask_invalid : np.ndarray
        Máscara booleana de píxeles inválidos.

    height : int
        Alto del raster.

    width : int
        Ancho del raster.
    """
    n_features, height, width = stack_final.shape

    matriz = stack_final.reshape(n_features, -1).T
    matriz = matriz.astype("float32")

    mask_invalid = ~np.isfinite(matriz).all(axis=1)
    matriz_valida = matriz[~mask_invalid]

    return matriz, matriz_valida, mask_invalid, height, width


def predecir_stack(modelo, stack_final: np.ndarray) -> np.ndarray:
    """
    Aplica un modelo sobre el stack predictivo y reconstruye la imagen de predicción.
    """
    matriz, matriz_valida, mask_invalid, height, width = preparar_matriz_prediccion(stack_final)

    pred_flat = np.full(matriz.shape[0], np.nan, dtype="float32")
    pred_flat[~mask_invalid] = modelo.predict(matriz_valida).astype("float32")

    pred_image = pred_flat.reshape(height, width)

    return pred_image


def predecir_stack_modelos(
    modelos: dict,
    stack_final: np.ndarray
) -> dict:
    """
    Aplica múltiples modelos sobre el mismo stack predictivo.

    Parameters
    ----------
    modelos : dict
        Diccionario con modelos entrenados.

    stack_final : np.ndarray
        Stack predictivo con forma (n_variables, filas, columnas).

    Returns
    -------
    dict
        Diccionario de predicciones raster por modelo.
    """
    predicciones = {}

    for nombre_modelo, modelo in modelos.items():
        print(f"Generando predicción con modelo {nombre_modelo}...")

        predicciones[nombre_modelo] = predecir_stack(
            modelo=modelo,
            stack_final=stack_final
        )

    return predicciones


def convertir_prediccion_escala_original(
    pred_image: np.ndarray,
    target_modelo: str,
    aplicar_log: bool,
    prefijo_log: str = "ln_"
) -> np.ndarray:
    """
    Convierte la predicción a escala original si el modelo fue entrenado
    con una variable objetivo transformada mediante logaritmo natural.

    La condición principal es que el target_modelo comience con el prefijo ln_.
    Ejemplo:
    - ln_DQO -> se aplica np.exp()
    - pH -> no se transforma
    - DQO -> no se transforma
    """
    if aplicar_log and target_modelo.startswith(prefijo_log):
        return np.exp(pred_image)

    return pred_image


def convertir_predicciones_escala_original(
    predicciones: dict,
    target_modelo: str,
    aplicar_log: bool,
    prefijo_log: str = "ln_"
) -> dict:
    """
    Convierte a escala original las predicciones de múltiples modelos cuando aplica.
    """
    predicciones_convertidas = {}

    for nombre_modelo, pred_image in predicciones.items():
        predicciones_convertidas[nombre_modelo] = convertir_prediccion_escala_original(
            pred_image=pred_image,
            target_modelo=target_modelo,
            aplicar_log=aplicar_log,
            prefijo_log=prefijo_log
        )

    return predicciones_convertidas


# ============================================================
# RECORTE Y EXPORTACIÓN
# ============================================================

def cargar_limite_recorte(ruta_limite: Path, raster_crs):
    """
    Carga el límite de recorte y lo reproyecta al CRS del raster si es necesario.
    """
    if not ruta_limite.exists():
        raise FileNotFoundError(f"No se encontró el archivo de límite: {ruta_limite}")

    gdf_limite = gpd.read_file(ruta_limite)

    if gdf_limite.empty:
        raise ValueError("El archivo de límite está vacío.")

    if gdf_limite.crs is None:
        raise ValueError("El archivo de límite no tiene CRS definido.")

    if gdf_limite.crs != raster_crs:
        gdf_limite = gdf_limite.to_crs(raster_crs)

    return gdf_limite


def exportar_raster_prediccion(
    pred_image: np.ndarray,
    profile: dict,
    ruta_salida: Path,
    ruta_limite: Path | None = None
) -> None:
    """
    Exporta una predicción como GeoTIFF.

    Si se proporciona ruta_limite, recorta el raster con dicho límite.
    """
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    profile_pred = profile.copy()
    profile_pred.update(
        count=1,
        dtype="float32",
        nodata=np.nan
    )

    if ruta_limite is None:
        with rasterio.open(ruta_salida, "w", **profile_pred) as dst:
            dst.write(pred_image.astype("float32"), 1)

        print("Raster de predicción exportado correctamente en:")
        print(ruta_salida)
        return

    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmpfile:
        temp_path = tmpfile.name

    try:
        with rasterio.open(temp_path, "w", **profile_pred) as tmp:
            tmp.write(pred_image.astype("float32"), 1)

        gdf_limite = cargar_limite_recorte(ruta_limite, profile_pred["crs"])
        geoms = gdf_limite.geometry.values

        with rasterio.open(temp_path) as src_pred:
            clipped, clipped_transform = mask(src_pred, geoms, crop=True)
            clipped_profile = src_pred.profile.copy()
            clipped_profile.update({
                "height": clipped.shape[1],
                "width": clipped.shape[2],
                "transform": clipped_transform
            })

        with rasterio.open(ruta_salida, "w", **clipped_profile) as dst:
            dst.write(clipped.astype("float32"))

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    print("Raster de predicción exportado correctamente en:")
    print(ruta_salida)


def exportar_predicciones_modelos(
    predicciones: dict,
    profile: dict,
    carpeta_salida: Path,
    target_salida: str,
    ruta_limite: Path | None = None
) -> dict:
    """
    Exporta un raster GeoTIFF por cada modelo.

    Los archivos se guardan con nombres:

    {target_salida}_{modelo}.tif

    Ejemplo:
    DQO_SVR.tif
    DQO_GBR.tif
    DQO_RFR.tif
    """
    carpeta_salida.mkdir(parents=True, exist_ok=True)

    rutas_salida = {}

    for nombre_modelo, pred_image in predicciones.items():
        ruta_salida = carpeta_salida / f"{target_salida}_{nombre_modelo}.tif"

        exportar_raster_prediccion(
            pred_image=pred_image,
            profile=profile,
            ruta_salida=ruta_salida,
            ruta_limite=ruta_limite
        )

        rutas_salida[nombre_modelo] = ruta_salida

    return rutas_salida


# ============================================================
# PIPELINE COMPLETO DE INFERENCIA MULTIMODELO
# ============================================================

def pipeline_prediccion_multimodelo(
    rutas_modelos: dict,
    ruta_split: Path,
    ruta_raster: Path,
    carpeta_salida: Path,
    ruta_limite: Path | None = None
) -> tuple[dict, dict]:
    """
    Ejecuta el flujo completo de predicción espacial con múltiples modelos.

    Parameters
    ----------
    rutas_modelos : dict
        Diccionario de rutas de modelos entrenados.
        Ejemplo:
        {
            "SVR": Path(".../SVR.joblib"),
            "GBR": Path(".../GBR.joblib"),
            "RFR": Path(".../RFR.joblib")
        }

    ruta_split : Path
        Ruta del archivo .joblib con split y metadatos.

    ruta_raster : Path
        Ruta del ortomosaico multibanda recortado o completo.

    carpeta_salida : Path
        Carpeta donde se guardarán los raster de predicción.

    ruta_limite : Path | None
        Ruta del límite espacial de recorte. Si es None, no se recorta.

    Returns
    -------
    tuple
        predicciones, rutas_salida
    """
    modelos = cargar_modelos(rutas_modelos)
    split_data = cargar_split(ruta_split)

    target_modelo = split_data.get("target_modelo", split_data["target"])
    target_salida = split_data.get("target_salida", target_modelo.replace("ln_", "", 1))
    vars_pred = split_data["vars_pred"]
    aplicar_log = split_data.get("aplicar_log", target_modelo.startswith("ln_"))

    data, profile = leer_raster_multibanda(ruta_raster)

    variables_disponibles = construir_variables_disponibles(data)

    stack_final = construir_stack_predictivo(
        variables_disponibles=variables_disponibles,
        vars_pred=vars_pred
    )

    predicciones = predecir_stack_modelos(
        modelos=modelos,
        stack_final=stack_final
    )

    predicciones = convertir_predicciones_escala_original(
        predicciones=predicciones,
        target_modelo=target_modelo,
        aplicar_log=aplicar_log
    )

    rutas_salida = exportar_predicciones_modelos(
        predicciones=predicciones,
        profile=profile,
        carpeta_salida=carpeta_salida,
        target_salida=target_salida,
        ruta_limite=ruta_limite
    )

    return predicciones, rutas_salida

# ============================================================
# VISUALIZACIÓN DE PREDICCIONES RASTER
# ============================================================

def leer_raster_prediccion(ruta_raster: Path) -> np.ndarray:
    """
    Lee un raster de predicción y retorna la banda principal como arreglo NumPy.

    Los valores NoData se convierten a NaN para facilitar la visualización.
    """
    if not ruta_raster.exists():
        raise FileNotFoundError(f"No se encontró el raster de predicción: {ruta_raster}")

    with rasterio.open(ruta_raster) as src:
        arr = src.read(1).astype("float32")

        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan

    return arr


def visualizar_rasters_prediccion(
    rutas_salida: dict,
    target_salida: str,
    ruta_figura: Path | None = None,
    cmap: str = "RdYlBu_r",
    usar_percentiles: bool = True,
    pmin: float = 2,
    pmax: float = 98,
    figsize_base: tuple[int, int] = (7, 5),
    dpi: int = 300
) -> dict:
    """
    Visualiza los raster de predicción generados por los modelos.

    Parameters
    ----------
    rutas_salida : dict
        Diccionario con las rutas de los raster generados por modelo.
        Ejemplo:
        {
            "SVR": Path(".../DQO_SVR.tif"),
            "GBR": Path(".../DQO_GBR.tif"),
            "RFR": Path(".../DQO_RFR.tif")
        }

    target_salida : str
        Nombre limpio del parámetro usado para títulos y salidas.
        Ejemplo: DQO.

    ruta_figura : Path | None
        Ruta donde se guardará la figura. Si es None, no se guarda.

    cmap : str
        Paleta de colores de Matplotlib.

    usar_percentiles : bool
        Si True, usa percentiles para definir el rango de color.

    pmin, pmax : float
        Percentiles usados para definir el rango de visualización.

    figsize_base : tuple[int, int]
        Tamaño base por raster. El alto se multiplica por el número de modelos.

    dpi : int
        Resolución de guardado de la figura.

    Returns
    -------
    dict
        Diccionario con los arreglos raster leídos por modelo.
    """
    import matplotlib.pyplot as plt

    if not rutas_salida:
        raise ValueError("El diccionario rutas_salida está vacío.")

    rasters = {}

    for nombre_modelo, ruta_raster in rutas_salida.items():
        rasters[nombre_modelo] = leer_raster_prediccion(Path(ruta_raster))

    valores_validos = [
        arr[np.isfinite(arr)].ravel()
        for arr in rasters.values()
        if np.isfinite(arr).any()
    ]

    if not valores_validos:
        raise ValueError("No se encontraron valores válidos para visualizar en los raster.")

    valores = np.concatenate(valores_validos)

    if usar_percentiles:
        vmin, vmax = np.nanpercentile(valores, [pmin, pmax])
    else:
        vmin, vmax = np.nanmin(valores), np.nanmax(valores)

    n_modelos = len(rasters)

    fig, axes = plt.subplots(
        nrows=n_modelos,
        ncols=1,
        figsize=(figsize_base[0], figsize_base[1] * n_modelos)
    )

    if n_modelos == 1:
        axes = [axes]

    for ax, (nombre_modelo, arr) in zip(axes, rasters.items()):
        img = ax.imshow(
            arr,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax
        )

        ax.set_title(f"{target_salida} - {nombre_modelo}", fontsize=14)
        ax.axis("off")

        cbar = plt.colorbar(img, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label("Valor del parámetro")

    plt.tight_layout()

    if ruta_figura is not None:
        ruta_figura.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(ruta_figura, dpi=dpi, bbox_inches="tight")

        print("Figura guardada en:")
        print(ruta_figura)

    plt.show()

    return rasters


