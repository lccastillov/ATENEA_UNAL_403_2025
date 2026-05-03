from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# VALIDACIONES
# ============================================================

def modelo_tiene_importancias(modelo) -> bool:
    """
    Verifica si un modelo contiene el atributo feature_importances_.

    Este atributo está disponible en modelos basados en árboles, como:

    - RandomForestRegressor
    - GradientBoostingRegressor

    No está disponible directamente en modelos como SVR con kernel.
    """
    return hasattr(modelo, "feature_importances_")


def validar_importancias_modelo(modelo, vars_pred: list[str], nombre_modelo: str) -> None:
    """
    Valida que el modelo tenga importancias de variables y que la cantidad
    coincida con el número de variables predictoras.
    """
    if not modelo_tiene_importancias(modelo):
        raise AttributeError(
            f"El modelo {nombre_modelo} no contiene el atributo 'feature_importances_'. "
            "Este tipo de importancia aplica directamente a modelos basados en árboles. "
            "Para modelos como SVR se requieren otros enfoques interpretativos, "
            "como permutation importance."
        )

    importancias = modelo.feature_importances_

    if len(importancias) != len(vars_pred):
        raise ValueError(
            f"La cantidad de importancias del modelo {nombre_modelo} "
            "no coincide con la cantidad de variables predictoras."
        )


# ============================================================
# IMPORTANCIA DE VARIABLES
# ============================================================

def obtener_importancia_variables(
    modelo,
    vars_pred: list[str],
    nombre_modelo: str
) -> pd.DataFrame:
    """
    Obtiene la importancia relativa de variables para un modelo compatible.

    Parameters
    ----------
    modelo : object
        Modelo entrenado compatible con feature_importances_.

    vars_pred : list[str]
        Lista de variables predictoras usadas durante el entrenamiento.

    nombre_modelo : str
        Nombre del modelo evaluado.

    Returns
    -------
    pd.DataFrame
        Tabla con variables e importancias ordenadas de mayor a menor.
    """
    validar_importancias_modelo(
        modelo=modelo,
        vars_pred=vars_pred,
        nombre_modelo=nombre_modelo
    )

    df_importancia = pd.DataFrame({
        "modelo": nombre_modelo,
        "variable": vars_pred,
        "importancia": modelo.feature_importances_
    })

    df_importancia = df_importancia.sort_values(
        by="importancia",
        ascending=False
    ).reset_index(drop=True)

    return df_importancia


def obtener_importancias_modelos(
    modelos: dict,
    vars_pred: list[str],
    omitir_no_compatibles: bool = True
) -> dict:
    """
    Obtiene la importancia de variables para todos los modelos compatibles.

    Parameters
    ----------
    modelos : dict
        Diccionario de modelos entrenados. Ejemplo:
        {
            "SVR": modelo_svr,
            "GBR": modelo_gbr,
            "RFR": modelo_rfr
        }

    vars_pred : list[str]
        Lista de variables predictoras usadas durante el entrenamiento.

    omitir_no_compatibles : bool
        Si es True, omite modelos sin feature_importances_.
        Si es False, lanza error cuando un modelo no sea compatible.

    Returns
    -------
    dict
        Diccionario con tablas de importancia por modelo.
    """
    resultados_importancia = {}

    for nombre_modelo, modelo in modelos.items():
        if not modelo_tiene_importancias(modelo):
            mensaje = (
                f"Modelo {nombre_modelo} omitido: no contiene feature_importances_. "
                "Esto es esperado en modelos como SVR."
            )

            if omitir_no_compatibles:
                print(f"⚠️ {mensaje}")
                continue

            raise AttributeError(mensaje)

        df_importancia = obtener_importancia_variables(
            modelo=modelo,
            vars_pred=vars_pred,
            nombre_modelo=nombre_modelo
        )

        resultados_importancia[nombre_modelo] = df_importancia

    return resultados_importancia


# ============================================================
# EXPORTACIÓN DE TABLAS
# ============================================================

def guardar_tabla_importancia(
    df_importancia: pd.DataFrame,
    ruta_salida: Path
) -> None:
    """
    Guarda una tabla de importancia de variables en formato CSV.
    """
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    df_importancia.to_csv(ruta_salida, index=False, encoding="utf-8-sig")

    print("Tabla de importancia guardada en:")
    print(ruta_salida)


def guardar_tablas_importancia(
    resultados_importancia: dict,
    carpeta_salida: Path
) -> dict:
    """
    Guarda tablas de importancia para múltiples modelos.

    Returns
    -------
    dict
        Diccionario con las rutas de salida por modelo.
    """
    carpeta_salida.mkdir(parents=True, exist_ok=True)

    rutas_tablas = {}

    for nombre_modelo, df_importancia in resultados_importancia.items():
        ruta_salida = carpeta_salida / f"importancia_variables_{nombre_modelo}.csv"

        guardar_tabla_importancia(
            df_importancia=df_importancia,
            ruta_salida=ruta_salida
        )

        rutas_tablas[nombre_modelo] = ruta_salida

    return rutas_tablas


# ============================================================
# GRÁFICAS DE IMPORTANCIA
# ============================================================

def graficar_importancia_variables(
    df_importancia: pd.DataFrame,
    nombre_modelo: str,
    target: str,
    ruta_salida: Path | None = None,
    top_n: int | None = None
) -> None:
    """
    Genera una gráfica de barras horizontales con la importancia de variables.
    """
    df_plot = df_importancia.copy()

    if top_n is not None:
        df_plot = df_plot.head(top_n)

    df_plot = df_plot.sort_values("importancia", ascending=True)

    plt.figure(figsize=(10, 7))

    plt.barh(
        df_plot["variable"],
        df_plot["importancia"]
    )

    plt.xlabel("Importancia relativa")
    plt.ylabel("Variable predictora")
    plt.title(f"Importancia de variables - {nombre_modelo} ({target})")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    if ruta_salida is not None:
        ruta_salida.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(ruta_salida, dpi=300, bbox_inches="tight")

        print("Figura de importancia guardada en:")
        print(ruta_salida)

    plt.show()


def graficar_importancias_modelos(
    resultados_importancia: dict,
    target: str,
    carpeta_salida: Path,
    top_n: int | None = None
) -> dict:
    """
    Genera y guarda gráficas de importancia para múltiples modelos.

    Returns
    -------
    dict
        Diccionario con rutas de figuras por modelo.
    """
    carpeta_salida.mkdir(parents=True, exist_ok=True)

    rutas_figuras = {}

    for nombre_modelo, df_importancia in resultados_importancia.items():
        ruta_salida = carpeta_salida / f"importancia_variables_{nombre_modelo}.png"

        graficar_importancia_variables(
            df_importancia=df_importancia,
            nombre_modelo=nombre_modelo,
            target=target,
            ruta_salida=ruta_salida,
            top_n=top_n
        )

        rutas_figuras[nombre_modelo] = ruta_salida

    return rutas_figuras


# ============================================================
# TABLA CONSOLIDADA
# ============================================================

def construir_tabla_importancia_consolidada(
    resultados_importancia: dict
) -> pd.DataFrame:
    """
    Construye una tabla consolidada con las importancias de todos los modelos compatibles.
    """
    tablas = []

    for nombre_modelo, df_importancia in resultados_importancia.items():
        df_tmp = df_importancia.copy()
        df_tmp["modelo"] = nombre_modelo
        tablas.append(df_tmp)

    if not tablas:
        return pd.DataFrame(columns=["modelo", "variable", "importancia"])

    df_consolidado = pd.concat(tablas, ignore_index=True)

    return df_consolidado


def guardar_tabla_importancia_consolidada(
    resultados_importancia: dict,
    ruta_salida: Path
) -> pd.DataFrame:
    """
    Construye y guarda una tabla consolidada de importancia de variables.
    """
    df_consolidado = construir_tabla_importancia_consolidada(
        resultados_importancia=resultados_importancia
    )

    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    df_consolidado.to_csv(ruta_salida, index=False, encoding="utf-8-sig")

    print("Tabla consolidada de importancia guardada en:")
    print(ruta_salida)

    return df_consolidado


# ============================================================
# PIPELINE COMPLETO DE INTERPRETACIÓN
# ============================================================

def pipeline_importancia_variables(
    modelos: dict,
    vars_pred: list[str],
    target: str,
    carpeta_tablas: Path,
    carpeta_figuras: Path,
    top_n: int | None = None,
    guardar_consolidado: bool = True
) -> tuple[dict, dict, dict, pd.DataFrame | None]:
    """
    Ejecuta el flujo completo de análisis de importancia de variables.

    El análisis se aplica únicamente a modelos compatibles con feature_importances_,
    como GradientBoostingRegressor y RandomForestRegressor.

    Modelos como SVR se omiten automáticamente porque no tienen importancia de
    variables directa bajo este atributo.

    Parameters
    ----------
    modelos : dict
        Diccionario de modelos entrenados.

    vars_pred : list[str]
        Variables predictoras usadas durante el entrenamiento.

    target : str
        Variable objetivo modelada.

    carpeta_tablas : Path
        Carpeta donde se guardarán las tablas CSV.

    carpeta_figuras : Path
        Carpeta donde se guardarán las figuras PNG.

    top_n : int | None
        Número máximo de variables a graficar. Si es None, grafica todas.

    guardar_consolidado : bool
        Si True, guarda una tabla consolidada con todos los modelos compatibles.

    Returns
    -------
    tuple
        resultados_importancia, rutas_tablas, rutas_figuras, df_consolidado
    """
    resultados_importancia = obtener_importancias_modelos(
        modelos=modelos,
        vars_pred=vars_pred,
        omitir_no_compatibles=True
    )

    rutas_tablas = guardar_tablas_importancia(
        resultados_importancia=resultados_importancia,
        carpeta_salida=carpeta_tablas
    )

    rutas_figuras = graficar_importancias_modelos(
        resultados_importancia=resultados_importancia,
        target=target,
        carpeta_salida=carpeta_figuras,
        top_n=top_n
    )

    df_consolidado = None

    if guardar_consolidado:
        ruta_consolidado = carpeta_tablas / "importancia_variables_consolidado.csv"

        df_consolidado = guardar_tabla_importancia_consolidada(
            resultados_importancia=resultados_importancia,
            ruta_salida=ruta_consolidado
        )

    return resultados_importancia, rutas_tablas, rutas_figuras, df_consolidado