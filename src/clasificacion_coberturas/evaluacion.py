from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.inspection import permutation_importance


# ============================================================
# CARGA DE MODELOS Y METADATOS
# ============================================================

def cargar_modelo(ruta_modelo: Path):
    """
    Carga un modelo entrenado en formato joblib.
    """
    ruta_modelo = Path(ruta_modelo)

    if not ruta_modelo.exists():
        raise FileNotFoundError(f"No se encontró el modelo: {ruta_modelo}")

    return joblib.load(ruta_modelo)


def obtener_mapa_clases_desde_puntos(
    ruta_puntos: Path,
    columna_clase: str = "Clase_N1",
    columna_clase_nombre: str = "d_nivel_1_"
) -> dict:
    """
    Construye un diccionario código-clase a partir de la capa de puntos.

    Returns
    -------
    dict
        Diccionario con código numérico de clase como llave y nombre descriptivo
        como valor.
    """
    ruta_puntos = Path(ruta_puntos)

    if not ruta_puntos.exists():
        raise FileNotFoundError(f"No se encontró la capa de puntos: {ruta_puntos}")

    gdf = gpd.read_file(ruta_puntos)

    columnas_requeridas = [columna_clase, columna_clase_nombre]
    faltantes = [col for col in columnas_requeridas if col not in gdf.columns]

    if faltantes:
        raise ValueError(
            "La capa de puntos no contiene las columnas requeridas: "
            f"{faltantes}"
        )

    tabla = (
        gdf[[columna_clase, columna_clase_nombre]]
        .drop_duplicates()
        .sort_values(columna_clase)
    )

    return dict(zip(tabla[columna_clase], tabla[columna_clase_nombre]))


# ============================================================
# MÉTRICAS Y REPORTES
# ============================================================

def calcular_metricas_clasificacion(
    y_true,
    y_pred
) -> dict:
    """
    Calcula métricas principales para clasificación multiclase.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),
        "recall_macro": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),
        "f1_macro": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )
    }


def evaluar_modelo_final(
    modelo,
    X_test,
    y_test
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Evalúa el modelo final sobre el conjunto de prueba.

    Returns
    -------
    y_pred : np.ndarray
        Predicciones del modelo.

    df_metricas : pd.DataFrame
        Tabla con métricas globales.

    df_reporte : pd.DataFrame
        Reporte de clasificación por clase.

    matriz_confusion : np.ndarray
        Matriz de confusión.
    """
    y_pred = modelo.predict(X_test)

    metricas = calcular_metricas_clasificacion(
        y_true=y_test,
        y_pred=y_pred
    )

    df_metricas = pd.DataFrame([metricas])

    clases = sorted(np.unique(np.concatenate([y_test, y_pred])))

    matriz = confusion_matrix(
        y_test,
        y_pred,
        labels=clases
    )

    reporte = classification_report(
        y_test,
        y_pred,
        labels=clases,
        output_dict=True,
        zero_division=0
    )

    df_reporte = pd.DataFrame(reporte).T

    return y_pred, df_metricas, df_reporte, matriz


def guardar_dataframe(
    df: pd.DataFrame,
    ruta_salida: Path,
    index: bool = True
) -> None:
    """
    Guarda un DataFrame en CSV.
    """
    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        ruta_salida,
        index=index,
        encoding="utf-8-sig"
    )

    print("Tabla guardada en:")
    print(ruta_salida)


# ============================================================
# VISUALIZACIÓN DE EVALUACIÓN
# ============================================================

def graficar_matriz_confusion(
    matriz_confusion: np.ndarray,
    clases: list,
    mapa_clases: dict | None = None,
    ruta_figura: Path | None = None,
    titulo: str = "Matriz de confusión",
    cmap: str = "Blues",
    dpi: int = 300
) -> None:
    """
    Grafica y guarda la matriz de confusión.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    if mapa_clases is not None:
        etiquetas = [
            mapa_clases.get(clase, str(clase))
            for clase in clases
        ]
    else:
        etiquetas = [str(clase) for clase in clases]

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        matriz_confusion,
        annot=True,
        fmt="d",
        cmap=cmap,
        xticklabels=etiquetas,
        yticklabels=etiquetas,
        cbar_kws={"label": "Número de muestras"}
    )

    plt.xlabel("Clase estimada")
    plt.ylabel("Clase observada")
    plt.title(titulo, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if ruta_figura is not None:
        ruta_figura = Path(ruta_figura)
        ruta_figura.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(ruta_figura, dpi=dpi, bbox_inches="tight")
        print("Figura guardada en:")
        print(ruta_figura)

    plt.show()


def calcular_importancia_permutacion(
    modelo,
    X_test,
    y_test,
    variables_modelo: list[str],
    scoring: str = "f1_macro",
    n_repeats: int = 10,
    random_state: int = 42,
    n_jobs: int = -1
) -> pd.DataFrame:
    """
    Calcula importancia de variables por permutación.
    """
    resultado = permutation_importance(
        estimator=modelo,
        X=X_test,
        y=y_test,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=n_jobs
    )

    df_importancia = pd.DataFrame({
        "variable": variables_modelo,
        "importancia_media": resultado.importances_mean,
        "importancia_std": resultado.importances_std
    })

    df_importancia = df_importancia.sort_values(
        "importancia_media",
        ascending=False
    ).reset_index(drop=True)

    return df_importancia


def graficar_importancia_permutacion(
    df_importancia: pd.DataFrame,
    ruta_figura: Path | None = None,
    top_n: int = 20,
    titulo: str = "Importancia por permutación",
    dpi: int = 300
) -> None:
    """
    Grafica importancia de variables por permutación.
    """
    import matplotlib.pyplot as plt

    df_plot = df_importancia.head(top_n).copy()
    df_plot = df_plot.sort_values("importancia_media", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(6, 0.45 * len(df_plot))))

    ax.barh(
        df_plot["variable"],
        df_plot["importancia_media"],
        xerr=df_plot["importancia_std"]
    )

    ax.set_xlabel("Disminución media del score")
    ax.set_ylabel("Variable")
    ax.set_title(titulo, fontweight="bold")

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

def pipeline_evaluacion_clasificacion(
    modelo,
    X_test,
    y_test,
    variables_modelo: list[str],
    carpeta_tablas: Path,
    carpeta_figuras: Path,
    mapa_clases: dict | None = None,
    scoring_importancia: str = "f1_macro",
    n_repeats_importancia: int = 10,
    random_state: int = 42,
    n_jobs: int = -1
) -> dict:
    """
    Ejecuta la evaluación del modelo optimizado y guarda tablas y figuras.
    """
    carpeta_tablas = Path(carpeta_tablas)
    carpeta_figuras = Path(carpeta_figuras)

    y_pred, df_metricas, df_reporte, matriz = evaluar_modelo_final(
        modelo=modelo,
        X_test=X_test,
        y_test=y_test
    )

    clases = sorted(np.unique(np.concatenate([y_test, y_pred])))

    df_matriz = pd.DataFrame(
        matriz,
        index=clases,
        columns=clases
    )

    guardar_dataframe(
        df=df_metricas,
        ruta_salida=carpeta_tablas / "metricas_modelo_final.csv",
        index=False
    )

    guardar_dataframe(
        df=df_reporte,
        ruta_salida=carpeta_tablas / "classification_report.csv",
        index=True
    )

    guardar_dataframe(
        df=df_matriz,
        ruta_salida=carpeta_tablas / "matriz_confusion.csv",
        index=True
    )

    graficar_matriz_confusion(
        matriz_confusion=matriz,
        clases=clases,
        mapa_clases=mapa_clases,
        ruta_figura=carpeta_figuras / "matriz_confusion.png",
        titulo="Matriz de confusión - Modelo optimizado"
    )

    df_importancia = calcular_importancia_permutacion(
        modelo=modelo,
        X_test=X_test,
        y_test=y_test,
        variables_modelo=variables_modelo,
        scoring=scoring_importancia,
        n_repeats=n_repeats_importancia,
        random_state=random_state,
        n_jobs=n_jobs
    )

    guardar_dataframe(
        df=df_importancia,
        ruta_salida=carpeta_tablas / "importancia_permutacion.csv",
        index=False
    )

    graficar_importancia_permutacion(
        df_importancia=df_importancia,
        ruta_figura=carpeta_figuras / "importancia_permutacion.png",
        top_n=20,
        titulo="Importancia por permutación - Modelo optimizado"
    )

    return {
        "y_pred": y_pred,
        "df_metricas": df_metricas,
        "df_reporte": df_reporte,
        "matriz_confusion": matriz,
        "df_importancia": df_importancia
    }
