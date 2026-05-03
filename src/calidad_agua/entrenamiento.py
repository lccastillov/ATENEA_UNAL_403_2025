from pathlib import Path

import joblib

from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GridSearchCV


# ============================================================
# DEFINICIÓN DE MODELOS
# ============================================================

def crear_modelo_svr(random_state: int = 42) -> Pipeline:
    """
    Crea un pipeline para Support Vector Regression.

    SVR es sensible a la escala de las variables, por lo que se incluye
    StandardScaler antes del modelo.
    """
    modelo = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", SVR())
        ]
    )

    return modelo


def crear_modelo_gbr(random_state: int = 42) -> GradientBoostingRegressor:
    """
    Crea un modelo Gradient Boosting Regressor.
    """
    return GradientBoostingRegressor(random_state=random_state)


def crear_modelo_rfr(random_state: int = 42) -> RandomForestRegressor:
    """
    Crea un modelo Random Forest Regressor.
    """
    return RandomForestRegressor(random_state=random_state)


# ============================================================
# ESPACIOS DE HIPERPARÁMETROS
# ============================================================

def obtener_param_grid_svr() -> dict:
    """
    Define el espacio de búsqueda de hiperparámetros para SVR.
    """
    return {
        "model__kernel": ["rbf", "linear"],
        "model__C": [0.1, 1, 10, 100],
        "model__epsilon": [0.01, 0.1, 0.2, 0.5],
        "model__gamma": ["scale", "auto"]
    }


def obtener_param_grid_gbr() -> dict:
    """
    Define el espacio de búsqueda de hiperparámetros para Gradient Boosting Regressor.
    """
    return {
        "n_estimators": [100, 200, 300, 500],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [2, 3, 4, 5],
        "min_samples_split": [2, 4, 6],
        "min_samples_leaf": [1, 2, 4],
        "subsample": [0.7, 0.8, 1.0]
    }


def obtener_param_grid_rfr() -> dict:
    """
    Define el espacio de búsqueda de hiperparámetros para Random Forest Regressor.
    """
    return {
        "n_estimators": [100, 200, 300, 500, 700],
        "max_depth": [None, 10, 20, 30, 50],
        "min_samples_split": [2, 4, 6, 8],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
        "bootstrap": [True, False]
    }


# ============================================================
# GRID SEARCH
# ============================================================

def ejecutar_grid_search(
    modelo,
    param_grid: dict,
    X_train,
    y_train,
    cv: int = 5,
    scoring: str = "neg_root_mean_squared_error",
    n_jobs: int = -1,
    verbose: int = 2
) -> GridSearchCV:
    """
    Ejecuta GridSearchCV para un modelo y espacio de hiperparámetros dado.
    """
    grid_search = GridSearchCV(
        estimator=modelo,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose
    )

    grid_search.fit(X_train, y_train)

    return grid_search


def entrenar_svr(
    X_train,
    y_train,
    cv: int = 5,
    scoring: str = "neg_root_mean_squared_error",
    n_jobs: int = -1,
    verbose: int = 2
):
    """
    Entrena un modelo SVR mediante GridSearchCV.
    """
    modelo = crear_modelo_svr()
    param_grid = obtener_param_grid_svr()

    search = ejecutar_grid_search(
        modelo=modelo,
        param_grid=param_grid,
        X_train=X_train,
        y_train=y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose
    )

    return search.best_estimator_, search


def entrenar_gbr(
    X_train,
    y_train,
    random_state: int = 42,
    cv: int = 5,
    scoring: str = "neg_root_mean_squared_error",
    n_jobs: int = -1,
    verbose: int = 2
):
    """
    Entrena un modelo Gradient Boosting Regressor mediante GridSearchCV.
    """
    modelo = crear_modelo_gbr(random_state=random_state)
    param_grid = obtener_param_grid_gbr()

    search = ejecutar_grid_search(
        modelo=modelo,
        param_grid=param_grid,
        X_train=X_train,
        y_train=y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose
    )

    return search.best_estimator_, search


def entrenar_rfr(
    X_train,
    y_train,
    random_state: int = 42,
    cv: int = 5,
    scoring: str = "neg_root_mean_squared_error",
    n_jobs: int = -1,
    verbose: int = 2
):
    """
    Entrena un modelo Random Forest Regressor mediante GridSearchCV.
    """
    modelo = crear_modelo_rfr(random_state=random_state)
    param_grid = obtener_param_grid_rfr()

    search = ejecutar_grid_search(
        modelo=modelo,
        param_grid=param_grid,
        X_train=X_train,
        y_train=y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose
    )

    return search.best_estimator_, search


# ============================================================
# ENTRENAMIENTO MULTIMODELO
# ============================================================

def entrenar_modelos(
    X_train,
    y_train,
    random_state: int = 42,
    cv: int = 5,
    scoring: str = "neg_root_mean_squared_error",
    n_jobs: int = -1,
    verbose: int = 2
) -> tuple[dict, dict]:
    """
    Entrena los tres modelos principales del flujo:

    - SVR
    - GBR
    - RFR

    Retorna:
    - modelos_entrenados
    - resultados_busqueda
    """
    modelos_entrenados = {}
    resultados_busqueda = {}

    print("\n==============================")
    print("Entrenando modelo SVR")
    print("==============================")
    modelo_svr, search_svr = entrenar_svr(
        X_train=X_train,
        y_train=y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose
    )
    modelos_entrenados["SVR"] = modelo_svr
    resultados_busqueda["SVR"] = search_svr

    print("\n==============================")
    print("Entrenando modelo GBR")
    print("==============================")
    modelo_gbr, search_gbr = entrenar_gbr(
        X_train=X_train,
        y_train=y_train,
        random_state=random_state,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose
    )
    modelos_entrenados["GBR"] = modelo_gbr
    resultados_busqueda["GBR"] = search_gbr

    print("\n==============================")
    print("Entrenando modelo RFR")
    print("==============================")
    modelo_rfr, search_rfr = entrenar_rfr(
        X_train=X_train,
        y_train=y_train,
        random_state=random_state,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose
    )
    modelos_entrenados["RFR"] = modelo_rfr
    resultados_busqueda["RFR"] = search_rfr

    print("\n✅ Entrenamiento de modelos finalizado.")

    return modelos_entrenados, resultados_busqueda


# ============================================================
# GUARDADO DE MODELOS Y BÚSQUEDAS
# ============================================================

def guardar_modelo(modelo, ruta_modelo: Path) -> None:
    """
    Guarda un modelo entrenado en formato .joblib.
    """
    ruta_modelo.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, ruta_modelo)

    print(f"Modelo guardado en: {ruta_modelo}")


def guardar_modelos(
    modelos_entrenados: dict,
    carpeta_modelos: Path
) -> dict:
    """
    Guarda todos los modelos entrenados en una carpeta.

    Los archivos se guardan con nombres:
    - SVR.joblib
    - GBR.joblib
    - RFR.joblib
    """
    carpeta_modelos.mkdir(parents=True, exist_ok=True)

    rutas_modelos = {}

    for nombre_modelo, modelo in modelos_entrenados.items():
        ruta_modelo = carpeta_modelos / f"{nombre_modelo}.joblib"
        guardar_modelo(modelo, ruta_modelo)
        rutas_modelos[nombre_modelo] = ruta_modelo

    return rutas_modelos


def guardar_resultados_busqueda(
    resultados_busqueda: dict,
    ruta_salida: Path
) -> None:
    """
    Guarda el diccionario de objetos GridSearchCV.

    Esto permite consultar posteriormente:
    - mejores hiperparámetros,
    - mejor puntuación,
    - resultados completos de validación cruzada.
    """
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(resultados_busqueda, ruta_salida)

    print(f"Resultados de búsqueda guardados en: {ruta_salida}")


def extraer_mejores_hiperparametros(resultados_busqueda: dict) -> dict:
    """
    Extrae los mejores hiperparámetros de cada modelo entrenado.
    """
    mejores_params = {}

    for nombre_modelo, search in resultados_busqueda.items():
        mejores_params[nombre_modelo] = search.best_params_

    return mejores_params


def guardar_mejores_hiperparametros(
    resultados_busqueda: dict,
    ruta_salida: Path
) -> None:
    """
    Guarda los mejores hiperparámetros de cada modelo en un archivo .joblib.
    """
    mejores_params = extraer_mejores_hiperparametros(resultados_busqueda)

    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(mejores_params, ruta_salida)

    print(f"Mejores hiperparámetros guardados en: {ruta_salida}")


# ============================================================
# PIPELINE COMPLETO DE ENTRENAMIENTO
# ============================================================

def pipeline_entrenamiento_modelos(
    X_train,
    y_train,
    carpeta_modelos: Path,
    ruta_busquedas: Path | None = None,
    ruta_hiperparametros: Path | None = None,
    random_state: int = 42,
    cv: int = 5,
    scoring: str = "neg_root_mean_squared_error",
    n_jobs: int = -1,
    verbose: int = 2
) -> tuple[dict, dict, dict]:
    """
    Ejecuta el pipeline completo de entrenamiento multimodelo.

    Entrena:
    - SVR
    - GBR
    - RFR

    Guarda:
    - modelos entrenados,
    - opcionalmente resultados de GridSearchCV,
    - opcionalmente mejores hiperparámetros.

    Retorna:
    - modelos_entrenados
    - resultados_busqueda
    - rutas_modelos
    """
    modelos_entrenados, resultados_busqueda = entrenar_modelos(
        X_train=X_train,
        y_train=y_train,
        random_state=random_state,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose
    )

    rutas_modelos = guardar_modelos(
        modelos_entrenados=modelos_entrenados,
        carpeta_modelos=carpeta_modelos
    )

    if ruta_busquedas is not None:
        guardar_resultados_busqueda(
            resultados_busqueda=resultados_busqueda,
            ruta_salida=ruta_busquedas
        )

    if ruta_hiperparametros is not None:
        guardar_mejores_hiperparametros(
            resultados_busqueda=resultados_busqueda,
            ruta_salida=ruta_hiperparametros
        )

    return modelos_entrenados, resultados_busqueda, rutas_modelos