from pathlib import Path

import ee
import geemap
import geopandas as gpd
import pandas as pd


# ============================================================
# INICIALIZACIÓN Y ROI
# ============================================================

def inicializar_gee(project: str | None = None) -> None:
    """
    Inicializa Google Earth Engine.

    Parameters
    ----------
    project : str | None
        ID del proyecto de Google Earth Engine. Si es None, se intenta inicializar
        con la configuración por defecto del usuario autenticado.
    """
    if project:
        ee.Initialize(project=project)
    else:
        ee.Initialize()

    print("Google Earth Engine inicializado correctamente.")


def cargar_roi_local(ruta_roi: Path, epsg_salida: int = 4326) -> gpd.GeoDataFrame:
    """
    Carga un archivo vectorial local como GeoDataFrame y lo reproyecta a EPSG:4326.

    Earth Engine trabaja con geometrías en coordenadas geográficas, por lo que
    se recomienda convertir el ROI a EPSG:4326 antes de enviarlo a GEE.
    """
    if not ruta_roi.exists():
        raise FileNotFoundError(f"No se encontró el archivo ROI: {ruta_roi}")

    gdf = gpd.read_file(ruta_roi)

    if gdf.empty:
        raise ValueError("El archivo ROI está vacío.")

    if gdf.crs is None:
        raise ValueError("El archivo ROI no tiene CRS definido.")

    if gdf.crs.to_epsg() != epsg_salida:
        gdf = gdf.to_crs(epsg=epsg_salida)

    return gdf


def convertir_roi_a_ee(gdf_roi: gpd.GeoDataFrame):
    """
    Convierte un GeoDataFrame local a objeto de Earth Engine.
    """
    roi_ee = geemap.gdf_to_ee(gdf_roi)
    region = roi_ee.geometry()

    return roi_ee, region


def cargar_roi_como_ee(ruta_roi: Path, epsg_salida: int = 4326):
    """
    Carga un ROI local y lo convierte a objeto de Earth Engine.
    """
    gdf_roi = cargar_roi_local(
        ruta_roi=ruta_roi,
        epsg_salida=epsg_salida
    )

    roi_ee, region = convertir_roi_a_ee(gdf_roi)

    return gdf_roi, roi_ee, region


# ============================================================
# PROCESAMIENTO SENTINEL-2
# ============================================================

def mask_s2_clouds(image):
    """
    Aplica máscara de nubes y cirros usando la banda QA60 de Sentinel-2 SR.

    Bits utilizados:
    - bit 10: nubes
    - bit 11: cirros
    """
    qa = image.select("QA60")

    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11

    mask = (
        qa.bitwiseAnd(cloud_bit_mask).eq(0)
        .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    )

    return image.updateMask(mask)


def add_indices(img):
    """
    Calcula índices espectrales a partir de bandas Sentinel-2.

    Índices calculados:
    - NDVI
    - EVI
    - SAVI
    - MNDWI
    - NDMI
    - NDBI
    """
    ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")

    mndwi = img.normalizedDifference(["B3", "B11"]).rename("MNDWI")

    ndmi = img.normalizedDifference(["B8", "B11"]).rename("NDMI")

    ndbi = img.normalizedDifference(["B11", "B8"]).rename("NDBI")

    evi = img.expression(
        "2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))",
        {
            "NIR": img.select("B8"),
            "RED": img.select("B4"),
            "BLUE": img.select("B2")
        }
    ).rename("EVI")

    savi = img.expression(
        "1.5 * ((NIR - RED) / (NIR + RED + 0.5))",
        {
            "NIR": img.select("B8"),
            "RED": img.select("B4")
        }
    ).rename("SAVI")

    return img.addBands([
        ndvi,
        evi,
        savi,
        mndwi,
        ndmi,
        ndbi
    ])


def glcm_for_band(img, band_name: str, size: int = 3):
    """
    Calcula texturas GLCM para una banda de Sentinel-2.

    Para calcular texturas GLCM en Earth Engine, la banda debe estar en formato entero.
    Por esta razón, la banda se convierte a Uint16 antes del cálculo.

    Parameters
    ----------
    img : ee.Image
        Imagen base.

    band_name : str
        Nombre de la banda sobre la cual calcular texturas.

    size : int
        Tamaño de ventana para el cálculo de texturas.

    Returns
    -------
    ee.Image
        Imagen con bandas de textura renombradas con prefijo TEX_.
    """
    band_int = img.select(band_name).toUint16()

    textures = band_int.glcmTexture(size=size)

    metricas = [
        "contrast",
        "diss",
        "ent",
        "asm",
        "idm",
        "corr",
        "var",
        "sent"
    ]

    bandas_originales = [
        f"{band_name}_{metrica}"
        for metrica in metricas
    ]

    bandas_nuevas = [
        f"TEX_{band_name}_{metrica}"
        for metrica in metricas
    ]

    textures = textures.select(
        bandas_originales,
        bandas_nuevas
    )

    return textures


def crear_stack_sentinel2(
    region,
    year: int = 2018,
    cloud_pct: int = 15,
    glcm_size: int = 3,
    tex_bands: list[str] | None = None
):
    """
    Crea un stack Sentinel-2 con bandas espectrales, índices y texturas GLCM.

    Parameters
    ----------
    region : ee.Geometry
        Región de interés.

    year : int
        Año de análisis.

    cloud_pct : int
        Porcentaje máximo de nubosidad permitido.

    glcm_size : int
        Tamaño de ventana para texturas GLCM.

    tex_bands : list[str] | None
        Bandas sobre las cuales calcular texturas.

    Returns
    -------
    ee.Image
        Stack multibanda final.
    """
    if tex_bands is None:
        tex_bands = ["B2", "B3", "B4", "B8"]

    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))
        .map(mask_s2_clouds)
    )

    mosaic = collection.median().toFloat().clip(region)

    bandas_base = [
        "B2",
        "B3",
        "B4",
        "B8",
        "B5",
        "B6",
        "B7",
        "B8A",
        "B11",
        "B12"
    ]

    base_stack = mosaic.select(bandas_base)

    stack_indices = add_indices(base_stack)

    textures = []

    for band in tex_bands:
        textures.append(
            glcm_for_band(
                img=base_stack,
                band_name=band,
                size=glcm_size
            )
        )

    texture_stack = ee.Image.cat(textures)

    stack_final = stack_indices.addBands(texture_stack).toFloat().clip(region)

    return stack_final


# ============================================================
# VISUALIZACIÓN
# ============================================================

def visualizar_stack_gee(
    stack,
    roi_ee,
    region,
    year: int,
    vis_bands: list[str] | None = None,
    vis_min: float = 0,
    vis_max: float = 3000,
    zoom: int = 12
):
    """
    Visualiza el stack Sentinel-2 en un mapa interactivo con geemap.
    """
    if vis_bands is None:
        vis_bands = ["B4", "B3", "B2"]

    vis_params = {
        "bands": vis_bands,
        "min": vis_min,
        "max": vis_max
    }

    mapa = geemap.Map()
    mapa.centerObject(region, zoom)
    mapa.addLayer(stack, vis_params, f"Sentinel-2 Stack {year}")
    mapa.addLayer(roi_ee, {"color": "red"}, "ROI Humedales")

    return mapa


# ============================================================
# EXPORTACIÓN
# ============================================================

def exportar_stack_drive(
    stack,
    region,
    description: str,
    folder: str,
    file_name_prefix: str,
    scale: int = 10,
    max_pixels: float = 1e13
):
    """
    Exporta el stack a Google Drive como GeoTIFF.
    """
    task = ee.batch.Export.image.toDrive(
        image=stack,
        description=description,
        folder=folder,
        fileNamePrefix=file_name_prefix,
        scale=scale,
        region=region,
        fileFormat="GeoTIFF",
        maxPixels=max_pixels
    )

    task.start()

    print("Exportación a Google Drive iniciada.")
    print(f"Descripción de tarea: {description}")
    print(f"Carpeta en Drive: {folder}")
    print(f"Nombre de archivo: {file_name_prefix}")

    return task


def exportar_stack_asset(
    stack,
    region,
    description: str,
    asset_id: str,
    scale: int = 10,
    max_pixels: float = 1e13
):
    """
    Exporta el stack a un Asset de Google Earth Engine.

    Requiere que el usuario tenga permisos de escritura sobre el asset destino.
    """
    task = ee.batch.Export.image.toAsset(
        image=stack,
        description=description,
        assetId=asset_id,
        scale=scale,
        region=region,
        maxPixels=max_pixels
    )

    task.start()

    print("Exportación a Earth Engine Asset iniciada.")
    print(f"Descripción de tarea: {description}")
    print(f"Asset destino: {asset_id}")

    return task


def exportar_stack_cloud_storage(
    stack,
    region,
    description: str,
    bucket: str,
    file_name_prefix: str,
    scale: int = 10,
    max_pixels: float = 1e13
):
    """
    Exporta el stack a Google Cloud Storage.

    Requiere que el usuario tenga un bucket configurado y permisos de escritura.
    """
    task = ee.batch.Export.image.toCloudStorage(
        image=stack,
        description=description,
        bucket=bucket,
        fileNamePrefix=file_name_prefix,
        scale=scale,
        region=region,
        fileFormat="GeoTIFF",
        maxPixels=max_pixels
    )

    task.start()

    print("Exportación a Google Cloud Storage iniciada.")
    print(f"Descripción de tarea: {description}")
    print(f"Bucket: {bucket}")
    print(f"Nombre de archivo: {file_name_prefix}")

    return task


def exportar_stack(
    stack,
    region,
    modo_exportacion: str,
    description: str,
    scale: int = 10,
    drive_folder: str | None = None,
    file_name_prefix: str | None = None,
    asset_id: str | None = None,
    cloud_bucket: str | None = None,
    max_pixels: float = 1e13
):
    """
    Ejecuta la exportación del stack según el modo seleccionado.

    Modos disponibles:
    - drive
    - asset
    - cloud_storage
    """
    modo_exportacion = modo_exportacion.lower().strip()

    if modo_exportacion == "drive":
        if drive_folder is None or file_name_prefix is None:
            raise ValueError("Para exportar a Drive debe definir drive_folder y file_name_prefix.")

        return exportar_stack_drive(
            stack=stack,
            region=region,
            description=description,
            folder=drive_folder,
            file_name_prefix=file_name_prefix,
            scale=scale,
            max_pixels=max_pixels
        )

    if modo_exportacion == "asset":
        if asset_id is None:
            raise ValueError("Para exportar a Asset debe definir asset_id.")

        return exportar_stack_asset(
            stack=stack,
            region=region,
            description=description,
            asset_id=asset_id,
            scale=scale,
            max_pixels=max_pixels
        )

    if modo_exportacion == "cloud_storage":
        if cloud_bucket is None or file_name_prefix is None:
            raise ValueError("Para exportar a Cloud Storage debe definir cloud_bucket y file_name_prefix.")

        return exportar_stack_cloud_storage(
            stack=stack,
            region=region,
            description=description,
            bucket=cloud_bucket,
            file_name_prefix=file_name_prefix,
            scale=scale,
            max_pixels=max_pixels
        )

    raise ValueError(
        "modo_exportacion no reconocido. Use: 'drive', 'asset' o 'cloud_storage'."
    )


# ============================================================
# TRAZABILIDAD
# ============================================================

def obtener_lista_bandas(stack) -> list[str]:
    """
    Obtiene la lista de bandas del stack desde Earth Engine.
    """
    return stack.bandNames().getInfo()


def guardar_lista_bandas(
    bandas: list[str],
    ruta_salida: Path
) -> pd.DataFrame:
    """
    Guarda la lista de bandas del stack en un archivo CSV local.
    """
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    df_bandas = pd.DataFrame({
        "orden": range(1, len(bandas) + 1),
        "banda": bandas
    })

    df_bandas.to_csv(
        ruta_salida,
        index=False,
        encoding="utf-8-sig"
    )

    print("Lista de bandas guardada en:")
    print(ruta_salida)

    return df_bandas