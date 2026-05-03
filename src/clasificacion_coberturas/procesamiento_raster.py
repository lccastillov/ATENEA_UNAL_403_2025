from pathlib import Path
from datetime import datetime
import re

from unidecode import unidecode
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.features import rasterize
from shapely.geometry import mapping


# ============================================================
# UTILIDADES
# ============================================================

def slugify(texto: str) -> str:
    """
    Convierte un texto en un nombre seguro para archivos.

    Ejemplo:
    'Humedal Córdoba' -> 'humedal_cordoba'
    """
    texto = unidecode(str(texto)).lower()
    texto = re.sub(r"[^a-z0-9]+", "_", texto).strip("_")

    return texto or "sin_nombre"


def crear_timestamp() -> str:
    """
    Genera una marca de tiempo para nombrar archivos de salida sin sobrescribir.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ============================================================
# CARGA Y VALIDACIÓN DE INSUMOS
# ============================================================

def cargar_humedales(
    ruta_humedales: Path,
    campo_nombre: str = "nombre_ap"
) -> gpd.GeoDataFrame:
    """
    Carga la capa vectorial de humedales y valida el campo usado para identificar
    cada humedal.

    Parameters
    ----------
    ruta_humedales : Path
        Ruta de la capa vectorial de humedales.

    campo_nombre : str
        Nombre del campo que contiene el nombre o identificador del humedal.
        Este campo se usa para disolver geometrías y nombrar los archivos de salida.
    """
    if not ruta_humedales.exists():
        raise FileNotFoundError(f"No se encontró la capa de humedales: {ruta_humedales}")

    gdf = gpd.read_file(ruta_humedales)

    if gdf.empty:
        raise ValueError("La capa de humedales está vacía.")

    if gdf.crs is None:
        raise ValueError("La capa de humedales no tiene CRS definido.")

    if campo_nombre not in gdf.columns:
        raise ValueError(
            f"La capa de humedales no contiene el campo '{campo_nombre}'. "
            "Este campo es necesario para identificar y nombrar cada humedal."
        )

    return gdf


def validar_raster(ruta_raster: Path) -> None:
    """
    Valida que exista el raster de entrada.
    """
    if not ruta_raster.exists():
        raise FileNotFoundError(f"No se encontró el raster de entrada: {ruta_raster}")


def alinear_crs_humedales_raster(
    gdf_humedales: gpd.GeoDataFrame,
    raster_crs
) -> gpd.GeoDataFrame:
    """
    Reproyecta la capa de humedales al CRS del raster si es necesario.
    """
    if gdf_humedales.crs != raster_crs:
        print(f"Reproyectando humedales de {gdf_humedales.crs} a {raster_crs}...")
        gdf_humedales = gdf_humedales.to_crs(raster_crs)

    return gdf_humedales


def disolver_humedales(
    gdf_humedales: gpd.GeoDataFrame,
    campo_nombre: str = "nombre_ap"
) -> gpd.GeoDataFrame:
    """
    Disuelve geometrías por humedal usando el campo de nombre definido.
    """
    gdf_diss = gdf_humedales.dissolve(
        by=campo_nombre,
        as_index=False
    )

    return gdf_diss


# ============================================================
# RECORTE Y MÁSCARA
# ============================================================

def recortar_raster_por_geometria(
    src,
    geometria,
    nodata_val: float | int,
    all_touched: bool = False
):
    """
    Recorta un raster usando una geometría.

    Parameters
    ----------
    src : rasterio.DatasetReader
        Raster abierto con rasterio.

    geometria : shapely geometry
        Geometría usada para el recorte.

    nodata_val : float | int
        Valor NoData de salida.

    all_touched : bool
        Si False, sólo se incluyen píxeles cuyo centro cae dentro del polígono.
        Si True, se incluyen todos los píxeles tocados por la geometría.
    """
    out_image, out_transform = rio_mask(
        src,
        [mapping(geometria)],
        crop=True,
        nodata=nodata_val,
        all_touched=all_touched
    )

    out_meta = src.meta.copy()
    out_meta.update({
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform,
        "nodata": nodata_val
    })

    return out_image, out_transform, out_meta


def crear_mascara_binaria(
    geometria,
    out_shape: tuple[int, int],
    transform,
    all_touched: bool = False
):
    """
    Crea una máscara binaria alineada al raster recortado.

    Valores:
    - 1: dentro del humedal
    - 0: fuera del humedal
    """
    mask_array = rasterize(
        [(geometria, 1)],
        out_shape=out_shape,
        transform=transform,
        fill=0,
        all_touched=all_touched,
        dtype="uint8"
    )

    return mask_array


def guardar_raster(
    array,
    meta: dict,
    ruta_salida: Path
) -> None:
    """
    Guarda un raster en disco.
    """
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(ruta_salida, "w", **meta) as dst:
        dst.write(array)


def guardar_mascara(
    mask_array,
    transform,
    crs,
    ruta_salida: Path
) -> None:
    """
    Guarda una máscara binaria de una banda.
    """
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    meta = {
        "driver": "GTiff",
        "height": mask_array.shape[0],
        "width": mask_array.shape[1],
        "count": 1,
        "dtype": "uint8",
        "crs": crs,
        "transform": transform,
        "nodata": 0
    }

    with rasterio.open(ruta_salida, "w", **meta) as dst:
        dst.write(mask_array, 1)


# ============================================================
# PROCESAMIENTO PRINCIPAL
# ============================================================

def procesar_recortes_humedales(
    ruta_raster: Path,
    ruta_humedales: Path,
    carpeta_rasters_recortados: Path,
    carpeta_mascaras: Path,
    carpeta_resumen: Path,
    year: int,
    campo_nombre: str = "nombre_ap",
    nodata_val: float | int = -9999,
    all_touched: bool = False
) -> pd.DataFrame:
    """
    Recorta un raster multibanda por humedal y genera máscaras binarias alineadas.

    Parameters
    ----------
    ruta_raster : Path
        Ruta del raster multibanda de entrada.

    ruta_humedales : Path
        Ruta de la capa vectorial de humedales.

    carpeta_rasters_recortados : Path
        Carpeta donde se guardarán los raster recortados.

    carpeta_mascaras : Path
        Carpeta donde se guardarán las máscaras binarias.

    carpeta_resumen : Path
        Carpeta donde se guardará el CSV resumen.

    year : int
        Año asociado al raster procesado. Se usa para nombrar el resumen.

    campo_nombre : str
        Campo de la capa vectorial usado para identificar cada humedal.

    nodata_val : float | int
        Valor NoData asignado a los raster recortados.

    all_touched : bool
        Si False, sólo se incluyen píxeles cuyo centro cae dentro del polígono.
        Si True, se incluyen todos los píxeles tocados por la geometría.

    Returns
    -------
    pd.DataFrame
        Tabla resumen con rutas de productos generados.
    """
    validar_raster(ruta_raster)

    gdf_humedales = cargar_humedales(
        ruta_humedales=ruta_humedales,
        campo_nombre=campo_nombre
    )

    registros = []

    with rasterio.open(ruta_raster) as src:
        gdf_humedales = alinear_crs_humedales_raster(
            gdf_humedales=gdf_humedales,
            raster_crs=src.crs
        )

        gdf_diss = disolver_humedales(
            gdf_humedales=gdf_humedales,
            campo_nombre=campo_nombre
        )

        for _, row in gdf_diss.iterrows():
            nombre_humedal = row[campo_nombre]
            slug = slugify(nombre_humedal)
            geom = row.geometry

            out_image, out_transform, out_meta = recortar_raster_por_geometria(
                src=src,
                geometria=geom,
                nodata_val=nodata_val,
                all_touched=all_touched
            )

            ruta_clip = carpeta_rasters_recortados / f"{slug}_clip.tif"
            ruta_mask = carpeta_mascaras / f"{slug}_mask.tif"

            guardar_raster(
                array=out_image,
                meta=out_meta,
                ruta_salida=ruta_clip
            )

            mask_array = crear_mascara_binaria(
                geometria=geom,
                out_shape=(out_image.shape[1], out_image.shape[2]),
                transform=out_transform,
                all_touched=all_touched
            )

            guardar_mascara(
                mask_array=mask_array,
                transform=out_transform,
                crs=src.crs,
                ruta_salida=ruta_mask
            )

            registros.append({
                "year": year,
                "nombre_humedal": nombre_humedal,
                "slug": slug,
                "ruta_raster_recortado": str(ruta_clip),
                "ruta_mascara": str(ruta_mask),
                "all_touched": all_touched,
                "nodata_val": nodata_val
            })

            print(f"Procesado: {nombre_humedal}")

    df_resumen = pd.DataFrame(registros)

    timestamp = crear_timestamp()
    ruta_resumen = carpeta_resumen / f"resumen_recortes_humedales_{year}_{timestamp}.csv"
    carpeta_resumen.mkdir(parents=True, exist_ok=True)

    df_resumen.to_csv(
        ruta_resumen,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nResumen de recortes guardado en:")
    print(ruta_resumen)

    return df_resumen