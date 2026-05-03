# Flujo de modelado de calidad de agua

Este documento describe el flujo de trabajo implementado para el modelado de parámetros de calidad de agua en humedales de Bogotá a partir de información espectral derivada de imágenes multibanda.

El flujo está diseñado para permitir la extracción de variables espectrales, preparación del dataset, entrenamiento de modelos de regresión, evaluación comparativa, análisis de importancia de variables e inferencia espacial sobre ortomosaicos.

La implementación se organiza en notebooks documentados y funciones reutilizables ubicadas en `src/calidad_agua/`.

---

## Objetivo del flujo

El objetivo del flujo de calidad de agua es construir y documentar un procedimiento reproducible para estimar parámetros fisicoquímicos de calidad de agua a partir de variables espectrales obtenidas de imágenes multibanda.

El flujo permite:

- integrar puntos de muestreo con reflectancias e índices espectrales;
- preparar datasets de modelado;
- aplicar transformaciones logarítmicas siguiendo la lógica del flujo original;
- entrenar varios modelos de regresión;
- comparar su desempeño;
- analizar importancia de variables;
- aplicar los modelos sobre un raster multibanda;
- generar mapas continuos del parámetro estimado.

---

## Modelos incluidos

El flujo compara tres algoritmos de regresión supervisada:

- **Support Vector Regression (SVR)**;
- **Gradient Boosting Regressor (GBR)**;
- **Random Forest Regressor (RFR)**.

Los tres modelos se entrenan sobre el mismo conjunto de entrenamiento y se evalúan sobre el mismo conjunto de prueba. Esto permite una comparación consistente entre algoritmos.

---

## Estructura del flujo

El modelado de calidad de agua se organiza en seis notebooks principales:

    notebooks/calidad_agua/
    ├── 01_extraccion_variables.ipynb
    ├── 02_preparacion_dataset_modelos.ipynb
    ├── 03_entrenamiento_modelos.ipynb
    ├── 04_evaluacion_modelos.ipynb
    ├── 05_analisis_importancia_variables.ipynb
    └── 06_prediccion_espacial_modelos.ipynb

Cada notebook ejecuta una etapa específica del flujo. La lógica técnica principal se encuentra en los módulos de `src/calidad_agua/`.

---

## Módulos asociados

El código reutilizable del flujo se encuentra en:

    src/calidad_agua/
    ├── extraccion_reflectancia.py
    ├── preparacion.py
    ├── entrenamiento.py
    ├── evaluacion.py
    ├── interpretacion.py
    ├── inferencia.py
    └── estadisticas_descriptivas.py

### `extraccion_reflectancia.py`

Contiene funciones para extraer reflectancias promedio e índices espectrales a partir de un ortomosaico multibanda y puntos de muestreo.

### `preparacion.py`

Contiene funciones para cargar el dataset, crear columnas logarítmicas, preparar variables predictoras y objetivo, generar el split de entrenamiento/prueba y construir matrices de correlación.

### `entrenamiento.py`

Contiene funciones para entrenar SVR, GBR y RFR mediante `GridSearchCV`.

### `evaluacion.py`

Contiene funciones para evaluar modelos, calcular métricas, generar gráficas observado vs estimado, comparar métricas y generar la curva de deviance para Gradient Boosting Regressor.

### `interpretacion.py`

Contiene funciones para analizar la importancia de variables en modelos compatibles con `feature_importances_`.

### `inferencia.py`

Contiene funciones para aplicar los modelos entrenados sobre un raster multibanda, reconstruir superficies espaciales de predicción, recortar el resultado y exportar archivos GeoTIFF.

### `estadisticas_descriptivas.py`

Contiene funciones exploratorias para el análisis descriptivo de parámetros de calidad de agua.

---

# 1. Extracción de variables espectrales

Notebook asociado:

    notebooks/calidad_agua/01_extraccion_variables.ipynb

Módulo asociado:

    src/calidad_agua/extraccion_reflectancia.py

## Objetivo

Extraer reflectancias promedio e índices espectrales a partir del ortomosaico multibanda y asociarlos a los puntos de muestreo de calidad de agua.

## Entradas esperadas

Esta etapa requiere:

- ortomosaico multibanda;
- puntos de muestreo;
- datos de laboratorio integrados en el shapefile o, opcionalmente, en un archivo Excel;
- ruta de salida para el archivo procesado.

El flujo está preparado para trabajar con un ortomosaico multibanda del sensor MicaSense RedEdge-P, considerando el siguiente orden de bandas:

1. Blue;
2. Green;
3. Pan;
4. Red;
5. RedEdge;
6. NIR.

## Variables generadas

El script calcula reflectancias promedio por banda:

- `Blue`;
- `Green`;
- `Pan`;
- `Red`;
- `RedEdge`;
- `NIR`.

También calcula índices espectrales:

- `NDVI`;
- `NDWI`;
- `EVI`;
- `SAVI`;
- `GNDVI`;
- `VARI`;
- `NDRE`;
- `CIgreen`;
- `CIRE`;
- `ARI`;
- `RENDVI`.

## Lógica del procesamiento

Para cada punto de muestreo, el script toma una ventana de píxeles alrededor de la ubicación del punto. Sobre esa ventana se calcula el valor promedio de las bandas e índices espectrales.

El tamaño de ventana es configurable mediante el parámetro `window_size`.

El flujo conserva los atributos originales del shapefile de puntos. Si se suministra un archivo Excel adicional, puede realizarse una unión con los datos de laboratorio. Si no se suministra Excel, se asume que los atributos requeridos ya están incorporados en el shapefile de puntos.

## Salida principal

La salida esperada es un archivo espacial integrado, preferiblemente en formato GeoPackage:

    data/processed/calidad_agua/Puntos_Muestreo_Reflectancia_Indices_Excel_2.gpkg

Este archivo contiene:

- atributos originales de los puntos;
- parámetros de laboratorio;
- reflectancias promedio;
- índices espectrales calculados.

Este producto es la base para el modelado posterior.

---

# 2. Preparación del dataset de modelado

Notebook asociado:

    notebooks/calidad_agua/02_preparacion_dataset_modelos.ipynb

Módulo asociado:

    src/calidad_agua/preparacion.py

## Objetivo

Preparar el dataset que será utilizado para entrenar, evaluar e interpretar los modelos de calidad de agua.

## Principales operaciones

Esta etapa realiza:

- carga del archivo espacial integrado;
- revisión de columnas disponibles;
- definición de variables predictoras;
- creación de columnas logarítmicas;
- definición de la variable objetivo;
- recorte configurable del número de registros;
- generación del split de entrenamiento y prueba;
- generación de matrices de correlación.

## Variables predictoras

Las variables predictoras corresponden a reflectancias e índices espectrales:

- `Blue`;
- `Green`;
- `Pan`;
- `Red`;
- `RedEdge`;
- `NIR`;
- `NDVI`;
- `NDWI`;
- `EVI`;
- `SAVI`;
- `GNDVI`;
- `VARI`;
- `NDRE`;
- `CIgreen`;
- `CIRE`;
- `ARI`;
- `RENDVI`.

Estas variables son usadas como entrada para los modelos SVR, GBR y RFR.

## Variables objetivo

El flujo permite trabajar con variables originales o transformadas.

Variables originales disponibles:

- `DQO`;
- `Fosfatos`;
- `Turbidez`;
- `Nitratos`;
- `Sulfatos`;
- `ficocianin`;
- `Chl`;
- `CE`;
- `pH`.

Variables transformadas mediante logaritmo natural:

- `ln_DQO`;
- `ln_Fosfatos`;
- `ln_Turbidez`;
- `ln_Nitratos`;
- `ln_Sulfatos`;
- `ln_ficocianin`;
- `ln_Chl`;
- `ln_CE`.

El parámetro `pH` se conserva en escala original.

## Transformación logarítmica

Algunos modelos pueden tener mejor resultados al tranformar los datos aplicando logaritmo, se puede configurar para hacerlo o no según se decida. En ese caso, para cada parámetro definido en `param_cols_log`, se crea una nueva columna con prefijo `ln_`.

La transformación aplicada es el logaritmo natural:

    ln(x) = np.log(x)

Si el valor original es nulo, cero o negativo, se asigna el valor de reemplazo `0.3`, de acuerdo con la lógica original del código.

Las columnas originales no se reemplazan. Las columnas transformadas se agregan al dataset.

## Separación entre `target_modelo` y `target_salida`

El flujo diferencia entre:

- `target_modelo`: variable usada para entrenar el modelo;
- `target_salida`: nombre limpio usado para organizar carpetas y archivos de salida.

Ejemplo:

    target_modelo = "ln_DQO"
    target_salida = "DQO"

Esto permite entrenar con la variable transformada `ln_DQO`, pero guardar los resultados en carpetas y archivos asociados al parámetro original `DQO`.

## Recorte de registros

El flujo original trabajó con los primeros 65 registros. Para conservar esa lógica, el notebook permite configurar:

    n_registros = 65

Si se desea usar todos los registros disponibles, se puede definir:

    n_registros = None

## Split de entrenamiento y prueba

El split se genera una sola vez y se guarda en formato `.joblib`.

Ejemplo:

    data/interim/calidad_agua/splits/split_DQO.joblib

Este archivo contiene:

- `X_train`;
- `X_test`;
- `y_train`;
- `y_test`;
- `target_modelo`;
- `target_salida`;
- `vars_pred`;
- configuración de transformación logarítmica;
- número de registros usado;
- tamaño del conjunto de prueba;
- semilla aleatoria.

Guardar el split permite que los notebooks posteriores utilicen exactamente la misma partición de datos.

## Matrices de correlación

En esta etapa se generan dos matrices de correlación:

- matriz con parámetros en escala original;
- matriz con parámetros transformados mediante logaritmo natural.

Estas matrices permiten explorar relaciones lineales entre variables espectrales, índices derivados y parámetros de calidad de agua.

Salidas esperadas:

    outputs/tables/calidad_agua/DQO/matriz_correlacion_original.csv
    outputs/tables/calidad_agua/DQO/matriz_correlacion_transformada.csv
    outputs/figures/calidad_agua/DQO/matriz_correlacion_original.png
    outputs/figures/calidad_agua/DQO/matriz_correlacion_transformada.png

Estas matrices tienen una finalidad exploratoria. Una correlación alta no implica causalidad, y una baja correlación lineal no descarta relaciones no lineales.

---

# 3. Entrenamiento de modelos

Notebook asociado:

    notebooks/calidad_agua/03_entrenamiento_modelos.ipynb

Módulo asociado:

    src/calidad_agua/entrenamiento.py

## Objetivo

Entrenar tres modelos de regresión para estimar el parámetro de calidad de agua seleccionado.

## Modelos entrenados

Los modelos incluidos son:

- Support Vector Regression (`SVR`);
- Gradient Boosting Regressor (`GBR`);
- Random Forest Regressor (`RFR`).

El modelo SVR se implementa dentro de un `Pipeline` con `StandardScaler`, debido a su sensibilidad a la escala de las variables.

Los modelos GBR y RFR se entrenan directamente sobre las variables predictoras.

## Búsqueda de hiperparámetros

Cada modelo se entrena mediante `GridSearchCV`.

El objetivo de la búsqueda es identificar la combinación de hiperparámetros con mejor desempeño dentro del espacio definido para cada algoritmo.

La métrica de optimización usada es:

    neg_root_mean_squared_error

El número de folds de validación cruzada se define mediante el parámetro `cv`.

## Entradas

Esta etapa usa el split generado en el notebook 02:

    data/interim/calidad_agua/splits/split_DQO.joblib

## Salidas

Los modelos se guardan por parámetro:

    models/calidad_agua/entrenados/DQO/
    ├── SVR.joblib
    ├── GBR.joblib
    └── RFR.joblib

También se guardan resultados asociados a la búsqueda de hiperparámetros:

    models/calidad_agua/metricas/DQO/gridsearch_modelos.joblib

Y mejores hiperparámetros:

    outputs/tables/calidad_agua/DQO/mejores_hiperparametros.joblib

## Consideración sobre tiempo de cómputo

El entrenamiento puede tomar tiempo considerable, ya que se ejecuta `GridSearchCV` para tres modelos.

El tiempo de ejecución depende de:

- número de registros;
- número de variables predictoras;
- tamaño de las grillas de hiperparámetros;
- número de folds de validación cruzada;
- capacidad de cómputo disponible.

Si se requiere una prueba rápida, puede reducirse temporalmente el espacio de búsqueda en `src/calidad_agua/entrenamiento.py`.

---

# 4. Evaluación comparativa de modelos

Notebook asociado:

    notebooks/calidad_agua/04_evaluacion_modelos.ipynb

Módulo asociado:

    src/calidad_agua/evaluacion.py

## Objetivo

Evaluar y comparar el desempeño de los modelos entrenados usando el mismo split de entrenamiento y prueba.

## Métricas calculadas

Las métricas principales son:

- `R2`;
- `RMSE`;
- `MAE`.

Estas métricas se calculan para:

- conjunto de entrenamiento;
- conjunto de prueba.

## Interpretación de R²

El coeficiente de determinación `R2` mide qué tanto las predicciones del modelo explican la variabilidad observada frente a una predicción basada en la media.

Una interpretación general es:

- `R2 = 1`: predicción perfecta;
- `R2 = 0`: desempeño equivalente a predecir la media;
- `R2 < 0`: desempeño peor que predecir la media.

El hecho de que un modelo sea no lineal no implica que deba tener valores bajos de R². La métrica evalúa la concordancia entre valores observados y estimados, no la forma interna del algoritmo.

## Escala de evaluación

Si el modelo fue entrenado con una variable transformada, por ejemplo `ln_DQO`, las métricas se calculan sobre esa escala logarítmica.

Por tanto, la interpretación directa en unidades originales debe hacerse con precaución. Para una evaluación en unidades originales, sería necesario transformar las predicciones y observaciones mediante la función exponencial antes de calcular métricas adicionales.

## Figuras generadas

El notebook genera:

- observado vs estimado para SVR;
- observado vs estimado para GBR;
- observado vs estimado para RFR;
- comparación de RMSE entre modelos;
- comparación general de métricas entre modelos;
- curva de deviance para Gradient Boosting Regressor.

Salidas esperadas:

    outputs/figures/calidad_agua/DQO/
    ├── observado_estimado_SVR.png
    ├── observado_estimado_GBR.png
    ├── observado_estimado_RFR.png
    ├── comparacion_rmse_modelos.png
    ├── comparacion_metricas_modelos.png
    └── deviance_gbr.png

## Tabla de métricas

La tabla de métricas se guarda en:

    outputs/tables/calidad_agua/DQO/metricas_modelos.csv

## Uso de las métricas

Las métricas permiten comparar el desempeño de los modelos, pero deben interpretarse junto con:

- tamaño de muestra;
- variabilidad del parámetro;
- distribución de los datos;
- presencia de valores extremos;
- transformación aplicada;
- relación esperada entre variables espectrales y parámetro de calidad de agua.

---

# 5. Análisis de importancia de variables

Notebook asociado:

    notebooks/calidad_agua/05_analisis_importancia_variables.ipynb

Módulo asociado:

    src/calidad_agua/interpretacion.py

## Objetivo

Analizar la importancia relativa de las variables predictoras en los modelos compatibles con `feature_importances_`.

## Modelos incluidos

El análisis de importancia directa se aplica a:

- Gradient Boosting Regressor (`GBR`);
- Random Forest Regressor (`RFR`).

El modelo SVR no se incluye en este análisis porque no dispone directamente del atributo `feature_importances_`.

## Salidas tabulares

    outputs/tables/calidad_agua/DQO/
    ├── importancia_variables_GBR.csv
    ├── importancia_variables_RFR.csv
    └── importancia_variables_consolidado.csv

## Salidas gráficas

    outputs/figures/calidad_agua/DQO/
    ├── importancia_variables_GBR.png
    └── importancia_variables_RFR.png

## Consideraciones

La importancia de variables basada en `feature_importances_` debe interpretarse como una medida interna del modelo, no como evidencia causal.

Puede verse afectada por:

- correlación entre predictores;
- redundancia entre índices espectrales;
- estructura del modelo;
- distribución de las variables.

---

# 6. Predicción espacial multimodelo

Notebook asociado:

    notebooks/calidad_agua/06_prediccion_espacial_modelos.ipynb

Módulo asociado:

    src/calidad_agua/inferencia.py

## Objetivo

Aplicar los modelos entrenados sobre un ortomosaico multibanda para generar superficies espaciales continuas del parámetro de calidad de agua estimado.

## Lógica matricial del proceso

Un raster multibanda tiene una estructura tridimensional:

    bandas × filas × columnas

Para aplicar un modelo de machine learning, el raster se reorganiza como una matriz bidimensional:

    píxeles × variables predictoras

Cada píxel funciona como una observación y cada banda o índice espectral funciona como una variable predictora.

Después de predecir, el vector de resultados se reorganiza nuevamente con las dimensiones espaciales originales para construir un raster de salida.

## Diferencia frente al flujo original

En el notebook original, esta lógica se desarrollaba de forma lineal dentro del mismo notebook.

En la versión reorganizada del repositorio, la lógica se conserva, pero se implementa mediante funciones reutilizables en `src/calidad_agua/inferencia.py`.

Esto permite:

- validar que las variables usadas en predicción coincidan con las usadas en entrenamiento;
- manejar valores inválidos;
- aplicar la conversión de escala si el modelo fue entrenado con variables `ln_`;
- recortar espacialmente la salida;
- exportar un raster por modelo;
- visualizar los resultados.

## Conversión de escala

Si el modelo fue entrenado con un target transformado, por ejemplo:

    target_modelo = "ln_DQO"

la predicción directa del modelo está en escala logarítmica. Antes de exportar los raster finales, el flujo aplica la transformación inversa:

    DQO = exp(ln_DQO)

De esta manera, los archivos raster finales quedan en la escala original del parámetro.

## Salidas raster

Para un parámetro como `DQO`, se generan tres archivos GeoTIFF:

    outputs/rasters/calidad_agua/DQO/
    ├── DQO_SVR.tif
    ├── DQO_GBR.tif
    └── DQO_RFR.tif

## Visualización de mapas

El notebook incluye una visualización preliminar de los raster generados.

La figura se guarda en:

    outputs/figures/calidad_agua/DQO/mapas_prediccion_modelos.png

Esta visualización permite revisar de forma general:

- si los raster fueron exportados correctamente;
- si el recorte espacial fue aplicado;
- si los patrones espaciales presentan valores coherentes;
- cómo difieren las superficies estimadas entre modelos.

## Advertencia sobre interpretación espacial

Los mapas generados representan estimaciones derivadas de los modelos, no mediciones directas.

La calidad de estos productos depende directamente del desempeño del modelo evaluado en las etapas anteriores.

Si las métricas de evaluación muestran bajo desempeño, los raster deben interpretarse como productos exploratorios del flujo metodológico y no como mapas predictivos robustos para toma de decisiones sin validación adicional.

---

## Productos principales del flujo

Para un parámetro como `DQO`, el flujo puede generar:

### Datos intermedios

    data/interim/calidad_agua/splits/split_DQO.joblib

### Datos procesados

    data/processed/calidad_agua/Puntos_Muestreo_Reflectancia_Indices_Excel_2.gpkg

### Modelos entrenados

    models/calidad_agua/entrenados/DQO/
    ├── SVR.joblib
    ├── GBR.joblib
    └── RFR.joblib

### Tablas

    outputs/tables/calidad_agua/DQO/
    ├── metricas_modelos.csv
    ├── matriz_correlacion_original.csv
    ├── matriz_correlacion_transformada.csv
    ├── importancia_variables_GBR.csv
    ├── importancia_variables_RFR.csv
    └── importancia_variables_consolidado.csv

### Figuras

    outputs/figures/calidad_agua/DQO/
    ├── matriz_correlacion_original.png
    ├── matriz_correlacion_transformada.png
    ├── observado_estimado_SVR.png
    ├── observado_estimado_GBR.png
    ├── observado_estimado_RFR.png
    ├── comparacion_rmse_modelos.png
    ├── comparacion_metricas_modelos.png
    ├── deviance_gbr.png
    ├── importancia_variables_GBR.png
    ├── importancia_variables_RFR.png
    └── mapas_prediccion_modelos.png

### Rasters

    outputs/rasters/calidad_agua/DQO/
    ├── DQO_SVR.tif
    ├── DQO_GBR.tif
    └── DQO_RFR.tif

---

## Consideraciones metodológicas

El flujo permite reproducir la lógica general del notebook original, pero con una estructura modular y documentada.

Algunas consideraciones importantes:

- El tamaño del dataset puede limitar la capacidad de generalización de los modelos.
- La partición entrenamiento/prueba puede influir fuertemente en las métricas si el número de muestras es bajo.
- Las métricas calculadas sobre variables `ln_` deben interpretarse en escala logarítmica.
- La transformación logarítmica se conserva para respetar la lógica original del flujo.
- El valor de reemplazo `0.3` para datos nulos, cero o negativos debe entenderse como una regla heredada del código original.
- La importancia de variables no debe interpretarse como causalidad.
- La inferencia espacial sólo debe considerarse robusta si los modelos presentan desempeño adecuado en evaluación.

---

## Orden recomendado de ejecución

El flujo debe ejecutarse en el siguiente orden:

1. `01_extraccion_variables.ipynb`
2. `02_preparacion_dataset_modelos.ipynb`
3. `03_entrenamiento_modelos.ipynb`
4. `04_evaluacion_modelos.ipynb`
5. `05_analisis_importancia_variables.ipynb`
6. `06_prediccion_espacial_modelos.ipynb`

Cada notebook depende de productos generados en etapas previas.

---

## Estado del flujo

El flujo de calidad de agua se encuentra estructurado, modularizado y documentado para su ejecución dentro del repositorio `humedales_bogota_ml`.

El flujo de clasificación de coberturas se documentará en una etapa posterior.
