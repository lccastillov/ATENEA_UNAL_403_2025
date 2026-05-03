from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage


# ============================================================
# LECTURA Y PREPROCESAMIENTO
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


def cargar_raster_para_clasificacion(
    ruta_raster: Path,
    normalizar: bool = True
) -> dict:
    """
    Carga y prepara un raster multibanda para clasificación espacial.

    El raster se transforma de:

    bandas × filas × columnas

    a:

    píxeles × variables
    """
    ruta_raster = Path(ruta_raster)

    if not ruta_raster.exists():
        raise FileNotFoundError(f"No se encontró el raster: {ruta_raster}")

    with rasterio.open(ruta_raster) as src:
        arr = src.read().astype(float)
        profile = src.profile.copy()
        nodata = src.nodata

    if normalizar:
        arr_proc = np.array([
            minmax_normalize(arr[i])
            for i in range(arr.shape[0])
        ])
    else:
        arr_proc = arr

    n_bandas, filas, columnas = arr_proc.shape

    X = np.moveaxis(arr_proc, 0, -1).reshape(-1, n_bandas)

    mascara_valida = np.isfinite(X).all(axis=1)

    if nodata is not None:
        arr_original = np.moveaxis(arr, 0, -1).reshape(-1, n_bandas)
        mascara_valida &= ~(arr_original == nodata).any(axis=1)

    return {
        "arr": arr,
        "arr_proc": arr_proc,
        "X": X,
        "mascara_valida": mascara_valida,
        "profile": profile,
        "n_bandas": n_bandas,
        "filas": filas,
        "columnas": columnas,
        "nodata": nodata
    }


# ============================================================
# CLASIFICACIÓN ESPACIAL
# ============================================================

def clasificar_raster(
    modelo,
    ruta_raster: Path,
    normalizar: bool = True,
    nodata_salida: int = 0
) -> tuple[np.ndarray, dict]:
    """
    Aplica un modelo de clasificación sobre todo el raster.

    Returns
    -------
    mapa_clasificado : np.ndarray
        Raster clasificado en 2D.

    info : dict
        Diccionario con perfil y metadatos.
    """
    info = cargar_raster_para_clasificacion(
        ruta_raster=ruta_raster,
        normalizar=normalizar
    )

    X = info["X"]
    mascara_valida = info["mascara_valida"]

    pred_flat = np.full(
        X.shape[0],
        nodata_salida,
        dtype=np.int16
    )

    pred_flat[mascara_valida] = modelo.predict(X[mascara_valida]).astype(np.int16)

    mapa_clasificado = pred_flat.reshape(
        info["filas"],
        info["columnas"]
    )

    return mapa_clasificado, info


def majority_filter(
    mapa: np.ndarray,
    size: int = 3,
    nodata: int = 0
) -> np.ndarray:
    """
    Aplica filtro de mayoría a un mapa clasificado.

    El valor NoData se mantiene sin modificar.
    """
    def moda_local(values):
        values = values.astype(np.int64)
        values = values[values != nodata]

        if values.size == 0:
            return nodata

        counts = np.bincount(values)
        return np.argmax(counts)

    mapa_filtrado = ndimage.generic_filter(
        mapa,
        moda_local,
        size=size,
        mode="nearest"
    )

    mapa_filtrado[mapa == nodata] = nodata

    return mapa_filtrado.astype(np.int16)


# ============================================================
# EXPORTACIÓN
# ============================================================

def guardar_mapa_clasificado(
    mapa: np.ndarray,
    profile: dict,
    ruta_salida: Path,
    nodata: int = 0
) -> None:
    """
    Guarda un mapa clasificado como GeoTIFF.
    """
    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    profile_out = profile.copy()
    profile_out.update({
        "count": 1,
        "dtype": "int16",
        "nodata": nodata
    })

    with rasterio.open(ruta_salida, "w", **profile_out) as dst:
        dst.write(mapa.astype(np.int16), 1)

    print("Raster guardado en:")
    print(ruta_salida)


# ============================================================
# VISUALIZACIÓN
# ============================================================

def visualizar_mapa_clasificado(
    mapa: np.ndarray,
    mapa_clases: dict | None = None,
    ruta_figura: Path | None = None,
    titulo: str = "Mapa clasificado",
    cmap: str = "tab20",
    nodata: int = 0,
    dpi: int = 300
) -> None:
    """
    Visualiza un mapa clasificado y guarda la figura si se indica ruta.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    mapa_plot = mapa.astype(float)
    mapa_plot[mapa == nodata] = np.nan

    fig, ax = plt.subplots(figsize=(10, 8))

    img = ax.imshow(
        mapa_plot,
        cmap=cmap,
        interpolation="nearest"
    )

    ax.set_title(titulo, fontweight="bold")
    ax.axis("off")

    if mapa_clases is not None:
        clases_presentes = sorted([
            int(c) for c in np.unique(mapa)
            if c != nodata
        ])

        cmap_obj = plt.get_cmap(cmap)

        patches = []

        for i, clase in enumerate(clases_presentes):
            nombre = mapa_clases.get(clase, str(clase))
            color = cmap_obj(i / max(1, len(clases_presentes) - 1))
            patches.append(
                mpatches.Patch(color=color, label=f"{clase} - {nombre}")
            )

        ax.legend(
            handles=patches,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=2,
            frameon=False
        )

    plt.tight_layout()

    if ruta_figura is not None:
        ruta_figura = Path(ruta_figura)
        ruta_figura.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(ruta_figura, dpi=dpi, bbox_inches="tight")
        print("Figura guardada en:")
        print(ruta_figura)

    plt.show()


# ============================================================
# PIPELINE
# ============================================================

def pipeline_clasificacion_espacial(
    modelo,
    ruta_raster: Path,
    carpeta_rasters_salida: Path,
    carpeta_figuras: Path,
    nombre_salida: str,
    mapa_clases: dict | None = None,
    normalizar: bool = True,
    aplicar_filtro_mayoria: bool = True,
    majority_size: int = 3,
    nodata_salida: int = 0
) -> dict:
    """
    Ejecuta clasificación espacial, filtro de mayoría, exportación y visualización.
    """
    mapa_clasificado, info = clasificar_raster(
        modelo=modelo,
        ruta_raster=ruta_raster,
        normalizar=normalizar,
        nodata_salida=nodata_salida
    )

    carpeta_rasters_salida = Path(carpeta_rasters_salida)
    carpeta_figuras = Path(carpeta_figuras)

    ruta_mapa = carpeta_rasters_salida / f"{nombre_salida}_clasificado.tif"
    ruta_figura_mapa = carpeta_figuras / f"{nombre_salida}_clasificado.png"

    guardar_mapa_clasificado(
        mapa=mapa_clasificado,
        profile=info["profile"],
        ruta_salida=ruta_mapa,
        nodata=nodata_salida
    )

    visualizar_mapa_clasificado(
        mapa=mapa_clasificado,
        mapa_clases=mapa_clases,
        ruta_figura=ruta_figura_mapa,
        titulo="Mapa clasificado - Modelo optimizado",
        nodata=nodata_salida
    )

    resultados = {
        "mapa_clasificado": mapa_clasificado,
        "ruta_mapa_clasificado": ruta_mapa
    }

    if aplicar_filtro_mayoria:
        mapa_filtrado = majority_filter(
            mapa=mapa_clasificado,
            size=majority_size,
            nodata=nodata_salida
        )

        ruta_mapa_filtrado = carpeta_rasters_salida / f"{nombre_salida}_clasificado_filtrado.tif"
        ruta_figura_filtrado = carpeta_figuras / f"{nombre_salida}_clasificado_filtrado.png"

        guardar_mapa_clasificado(
            mapa=mapa_filtrado,
            profile=info["profile"],
            ruta_salida=ruta_mapa_filtrado,
            nodata=nodata_salida
        )

        visualizar_mapa_clasificado(
            mapa=mapa_filtrado,
            mapa_clases=mapa_clases,
            ruta_figura=ruta_figura_filtrado,
            titulo=f"Mapa clasificado con filtro de mayoría {majority_size}x{majority_size}",
            nodata=nodata_salida
        )

        resultados["mapa_filtrado"] = mapa_filtrado
        resultados["ruta_mapa_filtrado"] = ruta_mapa_filtrado

    return resultados
