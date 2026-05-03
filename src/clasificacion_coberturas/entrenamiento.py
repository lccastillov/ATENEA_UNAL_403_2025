from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    HistGradientBoostingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# DEFINICIÓN DE MODELOS
# ============================================================

def obtener_modelos_base(random_state: int = 42) -> dict:
    """
    Retorna el conjunto de modelos base de clasificación.

    Los nombres conservan una estructura corta e identificable para comparar
    resultados y asociar búsquedas de hiperparámetros.
    """
    modelos_base = {
        "CARTO_base": DecisionTreeClassifier(random_state=random_state),

        "RF_base": RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1
        ),

        "SVM_RBF_base": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            random_state=random_state
        ),

        "GB_base": GradientBoostingClassifier(
            random_state=random_state
        ),

        "ExtraTrees_base": ExtraTreesClassifier(
            n_estimators=400,
            random_state=random_state,
            n_jobs=-1
        ),

        "AdaBoost_base": AdaBoostClassifier(
            n_estimators=300,
            learning_rate=0.5,
            random_state=random_state
        ),

        "HistGB_base": HistGradientBoostingClassifier(
            random_state=random_state
        ),

        "KNN_base": KNeighborsClassifier(
            n_neighbors=15,
            weights="distance"
        )
    }

    return modelos_base


def requiere_escalamiento(nombre_modelo: str) -> bool:
    """
    Indica si el modelo es sensible a la escala de las variables.

    En este flujo se aplica escalamiento a modelos basados en distancias,
    márgenes o vecindades, como SVM y KNN.
    """
    modelos_con_scaler = [
        "SVM_RBF_base",
        "KNN_base"
    ]

    return nombre_modelo in modelos_con_scaler


def construir_estimador(
    nombre_modelo: str,
    modelo,
    aplicar_scaler: bool | None = None
):
    """
    Construye el estimador final.

    Si el modelo requiere escalamiento, se envuelve en un Pipeline con
    StandardScaler. Para modelos basados en árboles, se usa directamente el
    estimador.
    """
    if aplicar_scaler is None:
        aplicar_scaler = requiere_escalamiento(nombre_modelo)

    if aplicar_scaler:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clasificador", clone(modelo))
        ])

    return clone(modelo)


# ============================================================
# MÉTRICAS
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


# ============================================================
# ENTRENAMIENTO BASE
# ============================================================

def entrenar_modelos_base(
    X_train,
    y_train,
    X_test,
    y_test,
    random_state: int = 42
) -> tuple[dict, pd.DataFrame]:
    """
    Entrena y evalúa modelos base de clasificación.

    Returns
    -------
    resultados : dict
        Diccionario con modelos entrenados, predicciones y métricas.

    df_resultados : pd.DataFrame
        Tabla comparativa ordenada por F1 macro.
    """
    modelos_base = obtener_modelos_base(random_state=random_state)

    resultados = {}

    for nombre, modelo in modelos_base.items():
        print(f"Entrenando modelo: {nombre}...")

        estimador = construir_estimador(
            nombre_modelo=nombre,
            modelo=modelo
        )

        estimador.fit(X_train, y_train)
        y_pred = estimador.predict(X_test)

        metricas = calcular_metricas_clasificacion(
            y_true=y_test,
            y_pred=y_pred
        )

        resultados[nombre] = {
            "modelo": estimador,
            "y_pred": y_pred,
            "metricas": metricas
        }

    df_resultados = pd.DataFrame({
        nombre: valores["metricas"]
        for nombre, valores in resultados.items()
    }).T

    df_resultados = df_resultados.sort_values(
        "f1_macro",
        ascending=False
    )

    return resultados, df_resultados


def obtener_mejor_modelo_base(
    resultados: dict,
    df_resultados: pd.DataFrame,
    metrica: str = "f1_macro"
) -> tuple[str, object]:
    """
    Selecciona el mejor modelo base según la métrica indicada.
    """
    mejor_nombre = df_resultados.sort_values(
        metrica,
        ascending=False
    ).index[0]

    mejor_modelo = resultados[mejor_nombre]["modelo"]

    return mejor_nombre, mejor_modelo


# ============================================================
# BÚSQUEDA DE HIPERPARÁMETROS
# ============================================================

def definir_busqueda_modelo(
    nombre_modelo: str,
    cv,
    scoring: str = "f1_macro",
    n_jobs: int = -1,
    verbose: int = 1,
    random_state: int = 42
):
    """
    Define la búsqueda de hiperparámetros para un modelo base.
    """
    if nombre_modelo == "CARTO_base":
        return RandomizedSearchCV(
            DecisionTreeClassifier(random_state=random_state),
            param_distributions={
                "criterion": ["gini", "entropy"],
                "max_depth": [5, 10, 15, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 3, 5]
            },
            n_iter=25,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state
        )

    if nombre_modelo == "RF_base":
        return RandomizedSearchCV(
            RandomForestClassifier(random_state=random_state, n_jobs=-1),
            param_distributions={
                "n_estimators": [200, 300, 600],
                "max_depth": [None, 20, 40],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", 0.3, 0.5],
                "class_weight": [None, "balanced"],
                "bootstrap": [True],
                "max_samples": [None, 0.7]
            },
            n_iter=50,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state
        )

    if nombre_modelo == "SVM_RBF_base":
        return RandomizedSearchCV(
            Pipeline([
                ("scaler", StandardScaler()),
                ("clasificador", SVC(kernel="rbf", random_state=random_state))
            ]),
            param_distributions={
                "clasificador__C": [0.1, 0.3, 1, 3, 10, 30, 100],
                "clasificador__gamma": ["scale", 0.001, 0.01, 0.1, 1]
            },
            n_iter=30,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state
        )

    if nombre_modelo == "GB_base":
        return RandomizedSearchCV(
            GradientBoostingClassifier(random_state=random_state),
            param_distributions={
                "n_estimators": [100, 200, 400, 600],
                "learning_rate": [0.01, 0.03, 0.1, 0.2],
                "max_depth": [2, 3, 4],
                "subsample": [0.7, 0.85, 1.0],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 5]
            },
            n_iter=40,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state
        )

    if nombre_modelo == "ExtraTrees_base":
        return RandomizedSearchCV(
            ExtraTreesClassifier(random_state=random_state, n_jobs=-1),
            param_distributions={
                "n_estimators": [300, 600, 900],
                "max_depth": [None, 20, 40],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", 0.3, 0.5],
                "bootstrap": [False, True],
                "class_weight": [None, "balanced"]
            },
            n_iter=40,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state
        )

    if nombre_modelo == "AdaBoost_base":
        return RandomizedSearchCV(
            AdaBoostClassifier(random_state=random_state),
            param_distributions={
                "n_estimators": [100, 200, 300, 500, 800],
                "learning_rate": [0.01, 0.03, 0.1, 0.3, 0.5, 1.0]
            },
            n_iter=30,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state
        )

    if nombre_modelo == "HistGB_base":
        return RandomizedSearchCV(
            HistGradientBoostingClassifier(random_state=random_state),
            param_distributions={
                "learning_rate": [0.01, 0.03, 0.1, 0.2],
                "max_depth": [None, 10, 20],
                "max_leaf_nodes": [15, 31, 63, 127],
                "min_samples_leaf": [20, 50, 100, 200]
            },
            n_iter=35,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state
        )

    if nombre_modelo == "KNN_base":
        return RandomizedSearchCV(
            Pipeline([
                ("scaler", StandardScaler()),
                ("clasificador", KNeighborsClassifier())
            ]),
            param_distributions={
                "clasificador__n_neighbors": [3, 5, 7, 9, 11, 15, 21, 31],
                "clasificador__weights": ["uniform", "distance"],
                "clasificador__metric": ["minkowski", "manhattan"]
            },
            n_iter=25,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state
        )

    raise ValueError(f"No hay búsqueda definida para el modelo: {nombre_modelo}")


def ajustar_mejor_modelo(
    mejor_nombre: str,
    X_train,
    y_train,
    cv_splits: int = 5,
    scoring: str = "f1_macro",
    n_jobs: int = -1,
    verbose: int = 1,
    random_state: int = 42
) -> tuple[object, object]:
    """
    Ajusta hiperparámetros del mejor modelo base mediante RandomizedSearchCV.
    """
    cv = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=random_state
    )

    search = definir_busqueda_modelo(
        nombre_modelo=mejor_nombre,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose,
        random_state=random_state
    )

    search.fit(X_train, y_train)

    mejor_modelo_tuning = search.best_estimator_

    return search, mejor_modelo_tuning


# ============================================================
# COMPARACIÓN BASE VS TUNING
# ============================================================

def comparar_base_tuning(
    mejor_nombre: str,
    modelo_base,
    modelo_tuning,
    X_test,
    y_test
) -> pd.DataFrame:
    """
    Compara el mejor modelo base contra su versión optimizada.
    """
    nombre_limpio = mejor_nombre.replace("_base", "")

    y_pred_base = modelo_base.predict(X_test)
    y_pred_tuning = modelo_tuning.predict(X_test)

    metricas_base = calcular_metricas_clasificacion(
        y_true=y_test,
        y_pred=y_pred_base
    )

    metricas_tuning = calcular_metricas_clasificacion(
        y_true=y_test,
        y_pred=y_pred_tuning
    )

    tabla = pd.DataFrame([
        {
            "Modelo": f"{nombre_limpio}_base",
            "Accuracy": metricas_base["accuracy"],
            "Precision_macro": metricas_base["precision_macro"],
            "Recall_macro": metricas_base["recall_macro"],
            "F1_macro": metricas_base["f1_macro"]
        },
        {
            "Modelo": f"{nombre_limpio}_tuning",
            "Accuracy": metricas_tuning["accuracy"],
            "Precision_macro": metricas_tuning["precision_macro"],
            "Recall_macro": metricas_tuning["recall_macro"],
            "F1_macro": metricas_tuning["f1_macro"]
        }
    ])

    tabla = tabla.sort_values("F1_macro", ascending=False)

    return tabla


# ============================================================
# GUARDADO
# ============================================================

def guardar_dataframe(
    df: pd.DataFrame,
    ruta_salida: Path
) -> None:
    """
    Guarda un DataFrame en CSV.
    """
    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(ruta_salida, index=True, encoding="utf-8-sig")

    print("Tabla guardada en:")
    print(ruta_salida)


def guardar_modelo(
    modelo,
    ruta_salida: Path
) -> None:
    """
    Guarda un modelo o búsqueda en formato joblib.
    """
    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(modelo, ruta_salida)

    print("Archivo guardado en:")
    print(ruta_salida)


def guardar_modelos_base(
    resultados: dict,
    carpeta_modelos_base: Path
) -> dict:
    """
    Guarda los modelos base entrenados.
    """
    carpeta_modelos_base = Path(carpeta_modelos_base)
    carpeta_modelos_base.mkdir(parents=True, exist_ok=True)

    rutas = {}

    for nombre, info in resultados.items():
        ruta_modelo = carpeta_modelos_base / f"{nombre}.joblib"

        joblib.dump(info["modelo"], ruta_modelo)

        rutas[nombre] = ruta_modelo

    print("Modelos base guardados en:")
    print(carpeta_modelos_base)

    return rutas


# ============================================================
# VISUALIZACIÓN
# ============================================================

def graficar_radar_modelos(
    df_resultados: pd.DataFrame,
    ruta_figura: Path | None = None,
    top_n: int | None = None,
    titulo: str = "Radar de métricas – Modelos base",
    dpi: int = 300
) -> None:
    """
    Genera un radar plot de métricas para comparar modelos base.
    """
    import matplotlib.pyplot as plt

    if top_n is not None:
        radar_df = df_resultados.head(top_n).copy()
    else:
        radar_df = df_resultados.copy()

    radar_df = radar_df.reset_index().rename(columns={"index": "Modelo"})

    metricas = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro"
    ]

    labels_metricas = [
        "Accuracy",
        "Precision macro",
        "Recall macro",
        "F1 macro"
    ]

    N = len(metricas)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels_metricas)

    vals = radar_df[metricas].values.flatten()

    rmin = max(0.0, float(np.min(vals)) - 0.03)
    rmax = min(1.0, float(np.max(vals)) + 0.03)

    if rmin == rmax:
        rmin, rmax = 0.0, 1.0

    ax.set_ylim(rmin, rmax)

    for _, row in radar_df.iterrows():
        nombre = row["Modelo"]
        values = row[metricas].tolist()
        values += values[:1]

        ax.plot(angles, values, linewidth=1.8, label=nombre)
        ax.fill(angles, values, alpha=0.08)

    ax.set_title(titulo, y=1.08)

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.45),
        ncol=3,
        frameon=False
    )

    plt.subplots_adjust(bottom=0.35)
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

def pipeline_entrenamiento_clasificacion(
    X_train,
    y_train,
    X_test,
    y_test,
    carpeta_modelos: Path,
    carpeta_tablas: Path,
    carpeta_figuras: Path,
    random_state: int = 42,
    cv_splits: int = 5,
    scoring: str = "f1_macro",
    n_jobs: int = -1,
    verbose: int = 1
) -> dict:
    """
    Ejecuta entrenamiento de modelos base, ajuste del mejor modelo y guardado
    de resultados.
    """
    resultados_base, df_resultados = entrenar_modelos_base(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        random_state=random_state
    )

    mejor_nombre, mejor_modelo_base = obtener_mejor_modelo_base(
        resultados=resultados_base,
        df_resultados=df_resultados,
        metrica="f1_macro"
    )

    print("\nMejor modelo base:")
    print(mejor_nombre)

    carpeta_modelos = Path(carpeta_modelos)
    carpeta_tablas = Path(carpeta_tablas)
    carpeta_figuras = Path(carpeta_figuras)

    rutas_modelos_base = guardar_modelos_base(
        resultados=resultados_base,
        carpeta_modelos_base=carpeta_modelos / "base"
    )

    guardar_dataframe(
        df=df_resultados,
        ruta_salida=carpeta_tablas / "metricas_modelos_base.csv"
    )

    graficar_radar_modelos(
        df_resultados=df_resultados,
        ruta_figura=carpeta_figuras / "radar_modelos_base.png",
        top_n=None,
        titulo="Radar de métricas - Modelos base"
    )

    search, mejor_modelo_tuning = ajustar_mejor_modelo(
        mejor_nombre=mejor_nombre,
        X_train=X_train,
        y_train=y_train,
        cv_splits=cv_splits,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose,
        random_state=random_state
    )

    guardar_modelo(
        modelo=search,
        ruta_salida=carpeta_modelos / "busqueda_modelo_tuning.joblib"
    )

    guardar_modelo(
        modelo=mejor_modelo_tuning,
        ruta_salida=carpeta_modelos / "modelo_clasificacion_tuning.joblib"
    )

    tabla_comparacion = comparar_base_tuning(
        mejor_nombre=mejor_nombre,
        modelo_base=mejor_modelo_base,
        modelo_tuning=mejor_modelo_tuning,
        X_test=X_test,
        y_test=y_test
    )

    guardar_dataframe(
        df=tabla_comparacion,
        ruta_salida=carpeta_tablas / "comparacion_base_tuning.csv"
    )

    return {
        "resultados_base": resultados_base,
        "df_resultados": df_resultados,
        "mejor_nombre": mejor_nombre,
        "mejor_modelo_base": mejor_modelo_base,
        "search": search,
        "mejor_modelo_tuning": mejor_modelo_tuning,
        "tabla_comparacion": tabla_comparacion,
        "rutas_modelos_base": rutas_modelos_base
    }
