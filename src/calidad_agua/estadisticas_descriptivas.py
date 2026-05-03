from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# CONFIGURACIÓN DE ESTILO
# ============================================================

def configurar_estilo_graficas():
    """
    Configura el estilo global de las figuras.
    """
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "axes.titlesize": 22,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"]
    })


# ============================================================
# DICCIONARIOS AUXILIARES
# ============================================================

def obtener_traduccion_estadisticas():
    """
    Retorna el diccionario de traducción de nombres estadísticos.
    """
    return {
        "mean": "Media",
        "50%": "Mediana",
        "std": "Desv. Estándar",
        "min": "Mínimo",
        "max": "Máximo"
    }


def obtener_parametros():
    """
    Retorna el diccionario de nombres y unidades de los parámetros.
    """
    return {
        "DQO": {"nombre": "DQO", "unidad": "mg/L O₂"},
        "pH": {"nombre": "pH", "unidad": "Unidades de pH"},
        "Fosfatos": {"nombre": "Fosfatos", "unidad": "mg/L"},
        "CE": {"nombre": "Conductividad Eléctrica", "unidad": "µS/cm"},
        "Turbidez": {"nombre": "Turbidez", "unidad": "NTU"},
        "Chl": {"nombre": "Clorofila-a", "unidad": "µg/L"},
        "ficocianina": {"nombre": "Ficocianina", "unidad": "µg/L"},
        "Nitratos ": {"nombre": "Nitratos", "unidad": "mg/L"},
        "Sulfatos": {"nombre": "Sulfatos", "unidad": "mg/L"}
    }


# ============================================================
# LECTURA Y VALIDACIÓN
# ============================================================

def validar_excel(path: Path) -> None:
    """
    Verifica que el archivo Excel exista.
    """
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo Excel: {path}")


def cargar_datos_excel(archivo: Path) -> pd.DataFrame:
    """
    Carga el archivo Excel y fija la primera columna como índice.
    """
    df = pd.read_excel(archivo)
    if df.empty:
        raise ValueError("El archivo Excel está vacío.")
    df = df.set_index(df.columns[0])
    return df


# ============================================================
# CÁLCULO DE ESTADÍSTICAS
# ============================================================

def calcular_estadisticas_generales(serie: pd.Series, trad_stats: dict) -> pd.Series:
    """
    Calcula estadísticas descriptivas generales para una serie numérica.
    """
    stats = serie.describe()[["mean", "50%", "std", "min", "max"]]
    stats = stats.rename(index=trad_stats)
    return stats


def calcular_estadisticas_ph(serie: pd.Series) -> dict:
    """
    Calcula estadísticas especiales para pH trabajando primero sobre [H+].
    """
    H = 10 ** (-serie.values)

    mean_H = H.mean()
    median_H = np.median(H)
    std_H = H.std(ddof=1)
    min_H = H.min()
    max_H = H.max()

    mean_pH = -np.log10(mean_H)
    median_pH = -np.log10(median_H)
    min_pH = -np.log10(max_H)
    max_pH = -np.log10(min_H)

    sigma_pH = (1.0 / np.log(10)) * (std_H / mean_H)

    stats_display = {
        "Media": mean_pH,
        "Mediana": median_pH,
        "Desv. Estándar": sigma_pH,
        "Mínimo": min_pH,
        "Máximo": max_pH
    }

    return stats_display


# ============================================================
# GRAFICACIÓN
# ============================================================

def construir_texto_estadisticas(stats: dict | pd.Series, unidad: str) -> str:
    """
    Construye el texto del recuadro de estadísticas.
    """
    lineas = []
    for k, v in stats.items():
        if unidad:
            lineas.append(f"{k}: {v:.2f} {unidad}")
        else:
            lineas.append(f"{k}: {v:.2f}")
    return "\n".join(lineas)


def graficar_boxplot(serie: pd.Series, nombre: str, unidad: str, textstr: str, salida_archivo: Path) -> None:
    """
    Genera y guarda un boxplot con puntos y recuadro de estadísticas.
    """
    plt.figure(figsize=(9, 7))
    sns.boxplot(y=serie, color="skyblue", width=0.4)
    sns.stripplot(y=serie, color="black", alpha=0.6, jitter=True)

    plt.title(nombre, fontsize=22, fontweight="bold", pad=20)
    plt.ylabel(unidad if unidad else "", fontsize=18, fontweight="bold")
    plt.xlabel("", fontsize=16, fontweight="bold")
    plt.xticks(fontsize=16, fontweight="bold")
    plt.yticks(fontsize=16, fontweight="bold")

    plt.gca().text(
        1.05,
        0.5,
        textstr,
        transform=plt.gca().transAxes,
        fontsize=14,
        fontweight="bold",
        verticalalignment="center",
        bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", ec="black", lw=1.5)
    )

    plt.tight_layout()
    salida_archivo.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(salida_archivo, dpi=400, bbox_inches="tight")
    plt.close()


# ============================================================
# FUNCIÓN PRINCIPAL DE PROCESAMIENTO
# ============================================================

def generar_estadisticas_y_graficas(archivo_excel: Path, salida_dir: Path) -> None:
    """
    Genera estadísticas descriptivas y boxplots para todas las columnas numéricas del Excel.
    """
    validar_excel(archivo_excel)
    configurar_estilo_graficas()

    trad_stats = obtener_traduccion_estadisticas()
    parametros = obtener_parametros()

    df = cargar_datos_excel(archivo_excel)

    for col in df.columns:
        serie = pd.to_numeric(df[col], errors="coerce").dropna()

        if serie.empty:
            print(f"⚠️ Columna '{col}' no tiene valores numéricos. Se omitirá.")
            continue

        nombre = parametros.get(col, {}).get("nombre", col)
        unidad = parametros.get(col, {}).get("unidad", "")

        if col.strip().lower() == "ph" or col == "pH":
            stats_display = calcular_estadisticas_ph(serie)
            textstr = construir_texto_estadisticas(stats_display, unidad)
        else:
            stats = calcular_estadisticas_generales(serie, trad_stats)
            textstr = construir_texto_estadisticas(stats, unidad)

        salida_archivo = salida_dir / f"{nombre}_boxplot.png"
        graficar_boxplot(serie, nombre, unidad, textstr, salida_archivo)

    print(f"✅ Gráficas tipo boxplot generadas en la carpeta: {salida_dir}")


# ============================================================
# EJECUCIÓN DIRECTA OPCIONAL
# ============================================================

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parents[2]

    archivo_excel = BASE_DIR / "data" / "raw" / "calidad_agua" / "laboratorio" / "Resultados_Calidad_Agua_HCordoba.xlsx"
    salida_dir = BASE_DIR / "outputs" / "figures" / "calidad_agua" / "estadisticas_descriptivas"

    generar_estadisticas_y_graficas(archivo_excel, salida_dir)