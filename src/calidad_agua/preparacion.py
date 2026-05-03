from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import joblib

from sklearn.model_selection import train_test_split


# ============================================================
# CARGA DE DATASET DE MODELO
# ============================================================

def cargar_dataset_modelo(ruta_datos_modelo: Path) -> pd.DataFrame:
    """
    Carga el archivo espacial integrado del modelo y lo convierte en un DataFrame tabular.

    El archivo de entrada corresponde al producto generado en la etapa de extracción
    de variables espectrales, el cual contiene reflectancias, índices espectrales,
    parámetros de laboratorio y geometría.

    Para el entrenamiento de los modelos, la geometría no se utiliza directamente,
    por lo que se elimina de la tabla de trabajo.
    """
    if not ruta_datos_modelo.exists():
        raise FileNotFoundError(f"No se encontró el archivo de datos del modelo: {ruta_datos_modelo}")

    gdf_modelo = gpd.read_file(ruta_datos_modelo)

    if gdf_modelo.empty:
        raise ValueError("El archivo espacial integrado está vacío.")

    df_modelo = pd.DataFrame(
        gdf_modelo.drop(columns="geometry", errors="ignore")
    ).copy()

    if df_modelo.empty:
        raise ValueError("El DataFrame generado a partir del archivo espacial está vacío.")

    return df_modelo


# ============================================================
# SELECCIÓN / RECORTE DE REGISTROS
# ============================================================

def recortar_registros(df: pd.DataFrame, n_registros: int | None = 65) -> pd.DataFrame:
    """
    Conserva los primeros n registros del DataFrame.

    Esta función replica la lógica del notebook original, donde se trabajó con
    los primeros 65 registros.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de entrada.

    n_registros : int | None
        Número de registros a conservar. Si es None, conserva todos los registros.

    Returns
    -------
    pd.DataFrame
        DataFrame recortado o completo.
    """
    df = df.copy()

    if n_registros is None:
        return df

    if not isinstance(n_registros, int):
        raise TypeError("n_registros debe ser un entero o None.")

    if n_registros <= 0:
        raise ValueError("n_registros debe ser mayor que cero o None.")

    return df.iloc[:n_registros].copy()


# ============================================================
# TRANSFORMACIÓN LOGARÍTMICA
# ============================================================

def crear_columnas_logaritmicas(
    df: pd.DataFrame,
    param_cols: list[str],
    valor_reemplazo: float = 0.3,
    prefijo: str = "ln_"
) -> pd.DataFrame:
    """
    Crea columnas transformadas con logaritmo natural para parámetros de calidad de agua.

    Esta función replica la lógica del notebook original:

    - Si el valor es válido y mayor que cero, aplica np.log(x).
    - Si el valor es nulo, cero o negativo, asigna valor_reemplazo.

    Las columnas originales no se reemplazan. Se crean nuevas columnas con prefijo,
    por ejemplo:

    - DQO -> ln_DQO
    - Fosfatos -> ln_Fosfatos

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de entrada.

    param_cols : list[str]
        Columnas de parámetros de calidad de agua a transformar.

    valor_reemplazo : float
        Valor asignado cuando el dato original es nulo, cero o negativo.

    prefijo : str
        Prefijo para las nuevas columnas transformadas.

    Returns
    -------
    pd.DataFrame
        DataFrame con las nuevas columnas logarítmicas.
    """
    df = df.copy()

    for col in param_cols:
        if col not in df.columns:
            print(f"⚠️ Columna '{col}' no encontrada. No se creó {prefijo}{col}.")
            continue

        new_col = f"{prefijo}{col}"

        serie = pd.to_numeric(df[col], errors="coerce")

        df[new_col] = serie.apply(
            lambda x: np.log(x) if pd.notnull(x) and x > 0 else valor_reemplazo
        )

    return df


def obtener_columnas_logaritmicas_disponibles(
    df: pd.DataFrame,
    param_cols: list[str],
    prefijo: str = "ln_"
) -> list[str]:
    """
    Retorna las columnas logarítmicas disponibles en el DataFrame.
    """
    return [f"{prefijo}{col}" for col in param_cols if f"{prefijo}{col}" in df.columns]


def obtener_target_salida(target_modelo: str, prefijo_log: str = "ln_") -> str:
    """
    Obtiene el nombre limpio del parámetro para carpetas y archivos de salida.

    Ejemplos
    --------
    ln_DQO -> DQO
    ln_Fosfatos -> Fosfatos
    pH -> pH
    """
    if target_modelo.startswith(prefijo_log):
        return target_modelo.replace(prefijo_log, "", 1)

    return target_modelo


def target_usa_log(target_modelo: str, prefijo_log: str = "ln_") -> bool:
    """
    Indica si la variable objetivo corresponde a una columna transformada con logaritmo.
    """
    return target_modelo.startswith(prefijo_log)


# ============================================================
# LIMPIEZA DE DATOS
# ============================================================

def limpiar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia el DataFrame eliminando valores infinitos y registros incompletos.
    """
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    return df


# ============================================================
# SELECCIÓN DE VARIABLES
# ============================================================

def validar_columnas_modelo(
    df: pd.DataFrame,
    target: str,
    vars_pred: list[str]
) -> None:
    """
    Valida que la variable objetivo y las variables predictoras existan en el DataFrame.
    """
    columnas_faltantes = []

    if target not in df.columns:
        columnas_faltantes.append(target)

    for var in vars_pred:
        if var not in df.columns:
            columnas_faltantes.append(var)

    if columnas_faltantes:
        raise ValueError(
            "Las siguientes columnas no existen en el DataFrame: "
            f"{columnas_faltantes}"
        )


def seleccionar_variables(
    df: pd.DataFrame,
    target: str,
    vars_pred: list[str]
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separa variables predictoras y variable objetivo.
    """
    validar_columnas_modelo(df, target, vars_pred)

    X = df[vars_pred].copy()
    y = df[target].copy()

    return X, y


# ============================================================
# PREPARACIÓN COMPLETA DEL DATASET
# ============================================================

def preparar_dataset(
    df: pd.DataFrame,
    target: str,
    vars_pred: list[str],
    n_registros: int | None = 65,
    param_cols_log: list[str] | None = None,
    crear_ln: bool = True,
    valor_reemplazo_log: float = 0.3
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Ejecuta la preparación del dataset para modelado.

    Incluye:
    - recorte opcional de registros;
    - creación opcional de columnas logarítmicas ln_;
    - validación de columnas;
    - limpieza de datos;
    - separación entre X e y.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de entrada.

    target : str
        Variable objetivo usada para entrenar el modelo. Puede ser una variable original
        o una variable transformada, por ejemplo ln_DQO.

    vars_pred : list[str]
        Variables predictoras.

    n_registros : int | None
        Número de registros a conservar. Si es None, conserva todos.

    param_cols_log : list[str] | None
        Columnas a partir de las cuales se crearán variables ln_.

    crear_ln : bool
        Si True, crea columnas logarítmicas a partir de param_cols_log.

    valor_reemplazo_log : float
        Valor usado cuando el dato original es nulo, cero o negativo.

    Returns
    -------
    X : pd.DataFrame
        Variables predictoras.

    y : pd.Series
        Variable objetivo.

    df_modelo : pd.DataFrame
        DataFrame preparado, incluyendo columnas originales y ln_.
    """
    df_modelo = recortar_registros(df, n_registros=n_registros)

    if crear_ln:
        if param_cols_log is None:
            raise ValueError("Debe especificar param_cols_log si crear_ln=True.")

        df_modelo = crear_columnas_logaritmicas(
            df=df_modelo,
            param_cols=param_cols_log,
            valor_reemplazo=valor_reemplazo_log
        )

    validar_columnas_modelo(df_modelo, target, vars_pred)

    columnas_modelo = vars_pred + [target]
    df_modelado = df_modelo[columnas_modelo].copy()
    df_modelado = limpiar_datos(df_modelado)

    X, y = seleccionar_variables(df_modelado, target, vars_pred)

    return X, y, df_modelo


# ============================================================
# SPLIT DE ENTRENAMIENTO Y PRUEBA
# ============================================================

def dividir_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Divide el dataset en entrenamiento y prueba.
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )


def guardar_split(
    ruta_split: Path,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    target_modelo: str,
    target_salida: str,
    vars_pred: list[str],
    crear_ln: bool,
    param_cols_log: list[str] | None,
    valor_reemplazo_log: float,
    n_registros: int | None,
    test_size: float,
    random_state: int
) -> None:
    """
    Guarda el split de entrenamiento y prueba junto con metadatos del modelado.
    """
    ruta_split.parent.mkdir(parents=True, exist_ok=True)

    split_data = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "target": target_modelo,
        "target_modelo": target_modelo,
        "target_salida": target_salida,
        "vars_pred": vars_pred,
        "crear_ln": crear_ln,
        "aplicar_log": target_usa_log(target_modelo),
        "columnas_log": param_cols_log or [],
        "param_cols_log": param_cols_log or [],
        "valor_reemplazo_log": valor_reemplazo_log,
        "n_registros": n_registros,
        "test_size": test_size,
        "random_state": random_state
    }

    joblib.dump(split_data, ruta_split)

    print("Split guardado correctamente en:")
    print(ruta_split)


def cargar_split(ruta_split: Path) -> dict:
    """
    Carga un split previamente guardado en formato .joblib.
    """
    if not ruta_split.exists():
        raise FileNotFoundError(f"No se encontró el archivo de split: {ruta_split}")

    return joblib.load(ruta_split)


def crear_y_guardar_split(
    X: pd.DataFrame,
    y: pd.Series,
    ruta_split: Path,
    target_modelo: str,
    target_salida: str,
    vars_pred: list[str],
    crear_ln: bool = True,
    param_cols_log: list[str] | None = None,
    valor_reemplazo_log: float = 0.3,
    n_registros: int | None = 65,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Crea y guarda un split de entrenamiento y prueba.

    Retorna:
    - X_train
    - X_test
    - y_train
    - y_test
    """
    X_train, X_test, y_train, y_test = dividir_dataset(
        X=X,
        y=y,
        test_size=test_size,
        random_state=random_state
    )

    guardar_split(
        ruta_split=ruta_split,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        target_modelo=target_modelo,
        target_salida=target_salida,
        vars_pred=vars_pred,
        crear_ln=crear_ln,
        param_cols_log=param_cols_log,
        valor_reemplazo_log=valor_reemplazo_log,
        n_registros=n_registros,
        test_size=test_size,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test

# ============================================================
# MATRICES DE CORRELACIÓN
# ============================================================

def seleccionar_columnas_existentes(
    df: pd.DataFrame,
    columnas: list[str]
) -> list[str]:
    """
    Retorna únicamente las columnas que existen en el DataFrame.
    """
    return [col for col in columnas if col in df.columns]


def calcular_matriz_correlacion(
    df: pd.DataFrame,
    columnas: list[str],
    metodo: str = "pearson"
) -> pd.DataFrame:
    """
    Calcula una matriz de correlación para un conjunto de columnas numéricas.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de entrada.

    columnas : list[str]
        Columnas a incluir en la matriz.

    metodo : str
        Método de correlación. Opciones comunes: 'pearson', 'spearman', 'kendall'.

    Returns
    -------
    pd.DataFrame
        Matriz de correlación.
    """
    columnas_disponibles = seleccionar_columnas_existentes(df, columnas)

    if not columnas_disponibles:
        raise ValueError("Ninguna de las columnas solicitadas existe en el DataFrame.")

    df_corr = df[columnas_disponibles].copy()

    for col in df_corr.columns:
        df_corr[col] = pd.to_numeric(df_corr[col], errors="coerce")

    matriz_corr = df_corr.corr(method=metodo)

    return matriz_corr


def guardar_matriz_correlacion(
    matriz_corr: pd.DataFrame,
    ruta_salida: Path
) -> None:
    """
    Guarda una matriz de correlación en formato CSV.
    """
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    matriz_corr.to_csv(ruta_salida, encoding="utf-8-sig")

    print("Matriz de correlación guardada en:")
    print(ruta_salida)


def graficar_matriz_correlacion(
    matriz_corr: pd.DataFrame,
    titulo: str,
    ruta_salida: Path | None = None,
    cmap: str = "coolwarm",
    annot: bool = False,
    figsize: tuple[int, int] = (14, 12),
    dpi: int = 300
) -> None:
    """
    Genera y guarda una figura tipo heatmap de una matriz de correlación.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=figsize)

    sns.heatmap(
        matriz_corr,
        cmap=cmap,
        center=0,
        annot=annot,
        fmt=".2f",
        square=True,
        linewidths=0.3,
        cbar_kws={"label": "Coeficiente de correlación"}
    )

    plt.title(titulo, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if ruta_salida is not None:
        ruta_salida.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(ruta_salida, dpi=dpi, bbox_inches="tight")

        print("Figura de matriz de correlación guardada en:")
        print(ruta_salida)

    plt.show()