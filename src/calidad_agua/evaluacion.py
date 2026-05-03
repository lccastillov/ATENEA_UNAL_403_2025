from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# ============================================================
# MÉTRICAS BASE
# ============================================================

def calcular_metricas(y_true, y_pred) -> dict:
    """
    Calcula métricas de evaluación para un modelo de regresión.

    Métricas calculadas:
    - R2
    - RMSE
    - MAE
    """
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    return {
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae
    }


def predecir_modelo(modelo, X):
    """
    Genera predicciones para un modelo dado.
    """
    return modelo.predict(X)


# ============================================================
# EVALUACIÓN INDIVIDUAL Y MULTIMODELO
# ============================================================

def evaluar_modelo_individual(modelo, X_train, X_test, y_train, y_test) -> dict:
    """
    Evalúa un modelo individual sobre entrenamiento y prueba.
    """
    y_pred_train = predecir_modelo(modelo, X_train)
    y_pred_test = predecir_modelo(modelo, X_test)

    metricas_train = calcular_metricas(y_train, y_pred_train)
    metricas_test = calcular_metricas(y_test, y_pred_test)

    return {
        "metricas_train": metricas_train,
        "metricas_test": metricas_test,
        "y_pred_train": y_pred_train,
        "y_pred_test": y_pred_test
    }


def evaluar_modelos(modelos: dict, X_train, X_test, y_train, y_test) -> dict:
    """
    Evalúa múltiples modelos.

    Parameters
    ----------
    modelos : dict
        Diccionario con estructura:
        {
            "SVR": modelo_svr,
            "GBR": modelo_gbr,
            "RFR": modelo_rfr
        }

    Returns
    -------
    dict
        Diccionario con métricas y predicciones por modelo.
    """
    resultados = {}

    for nombre_modelo, modelo in modelos.items():
        print(f"Evaluando modelo: {nombre_modelo}")

        resultados[nombre_modelo] = evaluar_modelo_individual(
            modelo=modelo,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test
        )

    return resultados


# ============================================================
# TABLA DE MÉTRICAS
# ============================================================

def construir_tabla_metricas(resultados: dict) -> pd.DataFrame:
    """
    Construye una tabla comparativa de métricas para todos los modelos.
    """
    registros = []

    for nombre_modelo, res in resultados.items():
        for conjunto in ["train", "test"]:
            metricas = res[f"metricas_{conjunto}"]

            registro = {
                "modelo": nombre_modelo,
                "conjunto": conjunto,
                "R2": metricas["R2"],
                "RMSE": metricas["RMSE"],
                "MAE": metricas["MAE"]
            }

            registros.append(registro)

    df_metricas = pd.DataFrame(registros)

    return df_metricas


def guardar_tabla_metricas(df_metricas: pd.DataFrame, ruta_salida: Path) -> None:
    """
    Guarda la tabla de métricas en formato CSV.
    """
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    df_metricas.to_csv(ruta_salida, index=False, encoding="utf-8-sig")

    print("Tabla de métricas guardada en:")
    print(ruta_salida)


# ============================================================
# GRÁFICAS: OBSERVADO VS ESTIMADO
# ============================================================

def graficar_observado_vs_estimado(
    y_true,
    y_pred,
    nombre_modelo: str,
    target: str,
    ruta_salida: Path | None = None
) -> None:
    """
    Genera una gráfica de observado vs estimado para un modelo.
    """
    plt.figure(figsize=(7, 7))

    plt.scatter(y_true, y_pred, alpha=0.7)

    min_val = min(np.nanmin(y_true), np.nanmin(y_pred))
    max_val = max(np.nanmax(y_true), np.nanmax(y_pred))

    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle="--"
    )

    plt.xlabel("Observado")
    plt.ylabel("Estimado")
    plt.title(f"Observado vs estimado - {nombre_modelo} ({target})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if ruta_salida is not None:
        ruta_salida.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(ruta_salida, dpi=300, bbox_inches="tight")
        print("Figura guardada en:")
        print(ruta_salida)

    plt.show()


def graficar_observado_vs_estimado_modelos(
    resultados: dict,
    y_test,
    target: str,
    carpeta_salida: Path | None = None
) -> None:
    """
    Genera gráficas observado vs estimado para todos los modelos.
    """
    for nombre_modelo, res in resultados.items():
        ruta_salida = None

        if carpeta_salida is not None:
            ruta_salida = carpeta_salida / f"observado_estimado_{nombre_modelo}.png"

        graficar_observado_vs_estimado(
            y_true=y_test,
            y_pred=res["y_pred_test"],
            nombre_modelo=nombre_modelo,
            target=target,
            ruta_salida=ruta_salida
        )


# ============================================================
# GRÁFICA: COMPARACIÓN RMSE
# ============================================================

def graficar_comparacion_rmse(
    df_metricas: pd.DataFrame,
    target: str,
    ruta_salida: Path | None = None,
    conjunto: str = "test"
) -> None:
    """
    Genera una gráfica de comparación de RMSE entre modelos.
    """
    df_plot = df_metricas[df_metricas["conjunto"] == conjunto].copy()

    plt.figure(figsize=(8, 6))
    plt.bar(df_plot["modelo"], df_plot["RMSE"])

    plt.xlabel("Modelo")
    plt.ylabel("RMSE")
    plt.title(f"Comparación de RMSE entre modelos - {target} ({conjunto})")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if ruta_salida is not None:
        ruta_salida.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(ruta_salida, dpi=300, bbox_inches="tight")
        print("Figura guardada en:")
        print(ruta_salida)

    plt.show()


# ============================================================
# GRÁFICA: COMPARACIÓN GENERAL DE MÉTRICAS
# ============================================================

def graficar_comparacion_metricas(
    df_metricas: pd.DataFrame,
    target: str,
    ruta_salida: Path | None = None,
    conjunto: str = "test"
) -> None:
    """
    Genera una gráfica comparativa de R2, RMSE y MAE entre modelos.

    Nota:
    Las métricas tienen escalas distintas, por lo que esta gráfica debe interpretarse
    como una comparación visual general. Para análisis detallado, consultar la tabla CSV.
    """
    df_plot = df_metricas[df_metricas["conjunto"] == conjunto].copy()
    metricas = ["R2", "RMSE", "MAE"]

    x = np.arange(len(df_plot["modelo"]))
    width = 0.25

    plt.figure(figsize=(10, 6))

    for i, metrica in enumerate(metricas):
        plt.bar(
            x + i * width,
            df_plot[metrica],
            width,
            label=metrica
        )

    plt.xticks(x + width, df_plot["modelo"])
    plt.xlabel("Modelo")
    plt.ylabel("Valor de la métrica")
    plt.title(f"Comparación de métricas entre modelos - {target} ({conjunto})")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if ruta_salida is not None:
        ruta_salida.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(ruta_salida, dpi=300, bbox_inches="tight")
        print("Figura guardada en:")
        print(ruta_salida)

    plt.show()


# ============================================================
# CURVA DE DEVIANCE - GRADIENT BOOSTING
# ============================================================

def calcular_deviance_gbr(modelo_gbr, X_test, y_test) -> list[float]:
    """
    Calcula la curva de deviance del modelo Gradient Boosting Regressor.

    En regresión, se usa el error cuadrático medio por etapa como aproximación
    de la deviance durante el proceso aditivo del Gradient Boosting.
    """
    if not hasattr(modelo_gbr, "staged_predict"):
        raise AttributeError(
            "El modelo proporcionado no tiene el método 'staged_predict'. "
            "Verifique que corresponda a un GradientBoostingRegressor."
        )

    test_deviance = []

    for y_pred in modelo_gbr.staged_predict(X_test):
        mse = mean_squared_error(y_test, y_pred)
        test_deviance.append(mse)

    return test_deviance


def graficar_deviance_gbr(
    modelo_gbr,
    X_test,
    y_test,
    target: str,
    ruta_salida: Path | None = None
) -> None:
    """
    Genera la curva de deviance del modelo Gradient Boosting Regressor.
    """
    test_deviance = calcular_deviance_gbr(
        modelo_gbr=modelo_gbr,
        X_test=X_test,
        y_test=y_test
    )

    plt.figure(figsize=(8, 6))

    plt.plot(
        np.arange(1, len(test_deviance) + 1),
        test_deviance
    )

    plt.xlabel("Número de árboles")
    plt.ylabel("Deviance / MSE")
    plt.title(f"Curva de deviance - Gradient Boosting Regressor ({target})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if ruta_salida is not None:
        ruta_salida.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(ruta_salida, dpi=300, bbox_inches="tight")
        print("Figura guardada en:")
        print(ruta_salida)

    plt.show()


# ============================================================
# PIPELINE COMPLETO DE EVALUACIÓN
# ============================================================

def pipeline_evaluacion_modelos(
    modelos: dict,
    X_train,
    X_test,
    y_train,
    y_test,
    target: str,
    ruta_metricas: Path | None = None,
    carpeta_figuras: Path | None = None
) -> tuple[dict, pd.DataFrame]:
    """
    Ejecuta el pipeline completo de evaluación multimodelo.

    Genera:
    - resultados por modelo,
    - tabla comparativa de métricas,
    - gráficas observado vs estimado,
    - comparación RMSE,
    - comparación general de métricas,
    - curva de deviance para GBR.
    """
    resultados = evaluar_modelos(
        modelos=modelos,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test
    )

    df_metricas = construir_tabla_metricas(resultados)

    if ruta_metricas is not None:
        guardar_tabla_metricas(
            df_metricas=df_metricas,
            ruta_salida=ruta_metricas
        )

    if carpeta_figuras is not None:
        carpeta_figuras.mkdir(parents=True, exist_ok=True)

    graficar_observado_vs_estimado_modelos(
        resultados=resultados,
        y_test=y_test,
        target=target,
        carpeta_salida=carpeta_figuras
    )

    ruta_rmse = None
    ruta_metricas_fig = None
    ruta_deviance = None

    if carpeta_figuras is not None:
        ruta_rmse = carpeta_figuras / "comparacion_rmse_modelos.png"
        ruta_metricas_fig = carpeta_figuras / "comparacion_metricas_modelos.png"
        ruta_deviance = carpeta_figuras / "deviance_gbr.png"

    graficar_comparacion_rmse(
        df_metricas=df_metricas,
        target=target,
        ruta_salida=ruta_rmse,
        conjunto="test"
    )

    graficar_comparacion_metricas(
        df_metricas=df_metricas,
        target=target,
        ruta_salida=ruta_metricas_fig,
        conjunto="test"
    )

    if "GBR" in modelos:
        graficar_deviance_gbr(
            modelo_gbr=modelos["GBR"],
            X_test=X_test,
            y_test=y_test,
            target=target,
            ruta_salida=ruta_deviance
        )

    return resultados, df_metricas