from pathlib import Path
import math
import re

import numpy as np
import matplotlib.pyplot as plt
import rasterio


# ============================================================
# VISUALIZACIÓN DE RASTERS
# ============================================================

def stretch_percentile(
    band: np.ndarray,
    pmin: float = 2,
    pmax: float = 98,
    fondo_cero: bool = True
) -> np.ndarray:
    """
    Aplica realce por percentiles a una banda raster.

    Parameters
    ----------
    band : np.ndarray
        Banda de entrada.

    pmin, pmax : float
        Percentiles usados para realzar contraste.

    fondo_cero : bool
        Si True, conserva en cero los píxeles con valor original igual a cero.

    Returns
    -------
    np.ndarray
        Banda normalizada entre 0 y 1.
    """
    band = band.astype(float)

    if fondo_cero:
        valid = band[band > 0]
    else:
        valid = band[np.isfinite(band)]

    if valid.size == 0:
        return np.zeros_like(band, dtype=float)

    min_val, max_val = np.percentile(valid, (pmin, pmax))

    if max_val - min_val == 0:
        return np.zeros_like(band, dtype=float)

    stretched = np.clip((band - min_val) / (max_val - min_val), 0, 1)

    if fondo_cero:
        stretched[band == 0] = 0

    return stretched


def limpiar_nombre_raster(ruta: Path) -> str:
    """
    Limpia el nombre de un raster para usarlo como título de figura.
    """
    nombre = Path(ruta).stem
    nombre = nombre.replace("_clip", "")
    nombre = re.sub(r"^(pdem|rdh)_", "", nombre)
    nombre = nombre.replace("_", " ").title()

    return nombre


def cargar_rgb(
    ruta_raster: Path,
    bandas_rgb: tuple[int, int, int] = (3, 2, 1),
    aplicar_stretch: bool = True
) -> np.ndarray:
    """
    Carga una composición RGB desde un raster multibanda Sentinel-2.

    Por defecto se asume el orden:
    - banda 1: B2 Azul
    - banda 2: B3 Verde
    - banda 3: B4 Rojo

    Parameters
    ----------
    ruta_raster : Path
        Ruta del raster multibanda.

    bandas_rgb : tuple[int, int, int]
        Índices de bandas rasterio para Rojo, Verde y Azul.

    aplicar_stretch : bool
        Si True, aplica realce por percentiles a cada canal.
    """
    ruta_raster = Path(ruta_raster)

    if not ruta_raster.exists():
        raise FileNotFoundError(f"No se encontró el raster: {ruta_raster}")

    with rasterio.open(ruta_raster) as src:
        r = src.read(bandas_rgb[0]).astype(float)
        g = src.read(bandas_rgb[1]).astype(float)
        b = src.read(bandas_rgb[2]).astype(float)

    if aplicar_stretch:
        r = stretch_percentile(r)
        g = stretch_percentile(g)
        b = stretch_percentile(b)

    return np.dstack((r, g, b))


def visualizar_mosaico_rasters(
    rutas_raster: list[Path],
    ruta_figura: Path | None = None,
    cols: int = 4,
    figsize_col: float = 4.5,
    figsize_row: float = 4.0,
    bandas_rgb: tuple[int, int, int] = (3, 2, 1),
    aplicar_stretch: bool = True,
    titulo: str = "Mosaico de raster recortados",
    dpi: int = 300
) -> None:
    """
    Visualiza un mosaico RGB de varios raster recortados.

    La figura se muestra en el notebook y, si se suministra ruta_figura,
    también se guarda en disco.
    """
    rutas_validas = [Path(r) for r in rutas_raster if Path(r).exists()]

    if not rutas_validas:
        raise FileNotFoundError("No se encontraron raster válidos para visualizar.")

    n = len(rutas_validas)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(figsize_col * cols, figsize_row * rows)
    )

    axes = np.array(axes).reshape(-1)

    for i, ruta in enumerate(rutas_validas):
        ax = axes[i]

        try:
            rgb = cargar_rgb(
                ruta_raster=ruta,
                bandas_rgb=bandas_rgb,
                aplicar_stretch=aplicar_stretch
            )

            ax.imshow(rgb)
            ax.set_title(limpiar_nombre_raster(ruta), fontsize=9)
            ax.axis("off")

        except Exception as exc:
            ax.set_title(f"Error: {Path(ruta).name}", fontsize=8)
            ax.text(0.5, 0.5, str(exc), ha="center", va="center", fontsize=7)
            ax.axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(titulo, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if ruta_figura is not None:
        ruta_figura = Path(ruta_figura)
        ruta_figura.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(ruta_figura, dpi=dpi, bbox_inches="tight")
        print("Figura guardada en:")
        print(ruta_figura)

    plt.show()


def visualizar_mosaico_bandas(
    ruta_raster: Path,
    feature_names: list[str] | None = None,
    ruta_figura: Path | None = None,
    stretch: bool = False,
    cols: int = 4,
    cmap: str = "gray",
    dpi: int = 300
) -> None:
    """
    Visualiza todas las bandas de un raster en un mosaico.

    Parameters
    ----------
    ruta_raster : Path
        Raster multibanda a visualizar.

    feature_names : list[str] | None
        Nombres de bandas. Si coincide con el número de bandas, se usan
        como títulos.

    ruta_figura : Path | None
        Ruta para guardar la figura.

    stretch : bool
        Si True, aplica realce por percentiles por banda.
    """
    ruta_raster = Path(ruta_raster)

    if not ruta_raster.exists():
        raise FileNotFoundError(f"No se encontró el raster: {ruta_raster}")

    with rasterio.open(ruta_raster) as src:
        n_bandas = src.count
        rows = math.ceil(n_bandas / cols)

        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(4.2 * cols, 3.8 * rows)
        )

        axes = np.array(axes).reshape(-1)

        for i in range(n_bandas):
            band = src.read(i + 1)

            if stretch:
                band = stretch_percentile(band)

            if feature_names is not None and len(feature_names) == n_bandas:
                titulo_banda = feature_names[i]
            else:
                titulo_banda = f"Banda {i + 1}"

            axes[i].imshow(band, cmap=cmap)
            axes[i].set_title(titulo_banda, fontsize=8)
            axes[i].axis("off")

        for j in range(n_bandas, len(axes)):
            axes[j].axis("off")

    fig.suptitle(f"Bandas de {ruta_raster.name}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if ruta_figura is not None:
        ruta_figura = Path(ruta_figura)
        ruta_figura.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(ruta_figura, dpi=dpi, bbox_inches="tight")
        print("Figura guardada en:")
        print(ruta_figura)

    plt.show()


def graficar_matriz_correlacion(
    matriz_corr,
    titulo: str,
    ruta_figura: Path | None = None,
    cmap: str = "coolwarm",
    annot: bool = False,
    figsize: tuple[int, int] = (14, 12),
    dpi: int = 300
) -> None:
    """
    Grafica una matriz de correlación con seaborn.
    """
    import seaborn as sns

    plt.figure(figsize=figsize)

    sns.heatmap(
        matriz_corr,
        cmap=cmap,
        center=0,
        annot=annot,
        fmt=".2f",
        square=True,
        linewidths=0.2,
        cbar_kws={"label": "Coeficiente de correlación"}
    )

    plt.title(titulo, fontsize=14, fontweight="bold")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()

    if ruta_figura is not None:
        ruta_figura = Path(ruta_figura)
        ruta_figura.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(ruta_figura, dpi=dpi, bbox_inches="tight")
        print("Figura guardada en:")
        print(ruta_figura)

    plt.show()
