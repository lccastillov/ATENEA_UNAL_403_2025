# Flujo de clasificación de coberturas

Este documento describe el flujo de trabajo implementado para la clasificación de coberturas en humedales de Bogotá.

El flujo integra procesamiento en Google Earth Engine, preparación local de rasters, selección de variables, extracción de muestras, entrenamiento de modelos de clasificación, evaluación del modelo optimizado e inferencia espacial para generar mapas temáticos de coberturas.

La implementación se organiza en notebooks documentados y funciones reutilizables ubicadas en `src/clasificacion_coberturas/`.

---

## Objetivo del flujo

El objetivo del flujo de clasificación de coberturas es construir un procedimiento reproducible para generar mapas de coberturas en humedales a partir de imágenes Sentinel-2, índices espectrales, texturas GLCM y puntos de entrenamiento.

El flujo permite:

- generar stacks multibanda desde Google Earth Engine;
- recortar rasters por humedal;
- visualizar y revisar bandas espectrales, índices y texturas;
- reducir redundancia entre variables mediante correlación;
- preparar puntos de entrenamiento;
- extraer valores raster en puntos;
- crear un split estratificado;
- entrenar y comparar modelos de clasificación;
- optimizar el mejor modelo mediante búsqueda de hiperparámetros;
- evaluar el modelo optimizado;
- aplicar el modelo espacialmente sobre el raster reducido;
- exportar mapas clasificados y mapas filtrados.

---

## Estructura del flujo

El flujo de clasificación de coberturas se organiza en siete notebooks principales:

    notebooks/clasificacion_coberturas/
    ├── 01_descarga_gee.ipynb
    ├── 02_recorte_humedales.ipynb
    ├── 03_preparacion_raster.ipynb
    ├── 04_preparacion_muestras.ipynb
    ├── 05_entrenamiento_modelos.ipynb
    ├── 06_evaluacion_modelos.ipynb
    └── 07_clasificacion_espacial.ipynb

Cada notebook ejecuta una etapa específica del flujo. La lógica técnica principal se encuentra en los módulos de `src/clasificacion_coberturas/`.

---

## Módulos asociados

El código reutilizable del flujo se encuentra en:

    src/clasificacion_coberturas/
    ├── gee.py
    ├── procesamiento_raster.py
    ├── visualizacion.py
    ├── preparacion_raster.py
    ├── muestras.py
    ├── entrenamiento.py
    ├── evaluacion.py
    └── inferencia.py

### `gee.py`

Contiene funciones para inicializar Google Earth Engine, cargar el ROI, generar el stack Sentinel-2, calcular índices, calcular texturas GLCM, visualizar el stack y exportarlo a Google Drive, Earth Engine Asset o Google Cloud Storage.

### `procesamiento_raster.py`

Contiene funciones para recortar rasters por humedal, generar máscaras binarias y guardar un resumen de productos generados.

### `visualizacion.py`

Contiene funciones para visualizar composiciones RGB, mosaicos de rasters, mosaicos de bandas y matrices de correlación.

### `preparacion_raster.py`

Contiene funciones para definir nombres de variables, cargar raster multibanda, crear muestras de píxeles válidos, calcular correlaciones, seleccionar variables y generar rasters reducidos.

### `muestras.py`

Contiene funciones para cargar puntos de entrenamiento, revisar clases, extraer valores raster en puntos, crear muestras tabulares y generar splits estratificados.

### `entrenamiento.py`

Contiene funciones para definir modelos base, entrenarlos, comparar métricas, generar radar plot, seleccionar el mejor modelo base, ejecutar `RandomizedSearchCV` y guardar modelos.

### `evaluacion.py`

Contiene funciones para cargar el modelo optimizado, calcular métricas, generar matriz de confusión, crear reportes de clasificación e importancia por permutación.

### `inferencia.py`

Contiene funciones para aplicar el modelo optimizado sobre el raster reducido, reconstruir el mapa clasificado, aplicar filtro de mayoría, visualizar resultados y exportar GeoTIFF.

---

# 1. Descarga y exportación desde Google Earth Engine

Notebook asociado:

    notebooks/clasificacion_coberturas/01_descarga_gee.ipynb

Módulo asociado:

    src/clasificacion_coberturas/gee.py

---

## Objetivo

Generar un stack multibanda Sentinel-2 desde Google Earth Engine para el área de estudio.

---

## Entradas esperadas

Esta etapa requiere:

- cuenta de Google Earth Engine activa;
- proyecto de Google Cloud registrado para usar Earth Engine;
- ROI local del área de estudio;
- permisos para exportar a Google Drive, Earth Engine Asset o Google Cloud Storage.

El ROI local se espera en:

    data/raw/clasificacion_coberturas/vectores/ROI_BOGOTA.shp

Se recomienda conservar todos los archivos asociados al shapefile:

    ROI_BOGOTA.shp
    ROI_BOGOTA.shx
    ROI_BOGOTA.dbf
    ROI_BOGOTA.prj
    ROI_BOGOTA.cpg

---

## Parámetros principales

Los parámetros configurables incluyen:

- año de análisis;
- nubosidad máxima;
- escala de exportación;
- tamaño de ventana para texturas GLCM;
- bandas usadas para texturas;
- nombre de salida;
- modo de exportación.

Ejemplo:

    year = 2018
    cloud_pct = 15
    scale = 10
    glcm_size = 3
    tex_bands = ["B2", "B3", "B4", "B8"]
    nombre_salida = "S2_2018_STACK_3"

---

## Variables generadas

El stack incluye bandas Sentinel-2:

- `B2`;
- `B3`;
- `B4`;
- `B8`;
- `B5`;
- `B6`;
- `B7`;
- `B8A`;
- `B11`;
- `B12`.

También incluye índices espectrales:

- `NDVI`;
- `EVI`;
- `SAVI`;
- `MNDWI`;
- `NDMI`;
- `NDBI`.

Y texturas GLCM calculadas sobre bandas seleccionadas:

- `TEX_B2_*`;
- `TEX_B3_*`;
- `TEX_B4_*`;
- `TEX_B8_*`.

Las métricas de textura consideradas son:

- `contrast`;
- `diss`;
- `ent`;
- `asm`;
- `idm`;
- `corr`;
- `var`;
- `sent`.

---

## Exportación

El flujo permite tres modos de exportación:

- `drive`: exportación a Google Drive;
- `asset`: exportación a Earth Engine Asset;
- `cloud_storage`: exportación a Google Cloud Storage.

La opción más directa para usuarios técnicos es Google Drive:

    modo_exportacion = "drive"
    drive_folder = "GEE_Coberturas_Humedales"

Para flujos institucionales o empresariales, Google Cloud Storage puede ser una alternativa más robusta si existe un bucket configurado.

---

## Salidas esperadas

La salida principal es un GeoTIFF multibanda exportado desde Google Earth Engine. Una vez descargado, debe ubicarse en:

    data/raw/clasificacion_coberturas/rasters/

Ejemplos:

    data/raw/clasificacion_coberturas/rasters/S2_2018_STACK_3.tif
    data/raw/clasificacion_coberturas/rasters/S2_2024_STACK_3.tif

También se genera una tabla local con la lista de bandas del stack:

    outputs/tables/clasificacion_coberturas/bandas_stack_S2_2018_STACK_3.csv

---

# 2. Recorte de rasters por humedal

Notebook asociado:

    notebooks/clasificacion_coberturas/02_recorte_humedales.ipynb

Módulo asociado:

    src/clasificacion_coberturas/procesamiento_raster.py

---

## Objetivo

Separar espacialmente el stack multibanda por humedal, generando rasters recortados y máscaras binarias.

---

## Entradas esperadas

Esta etapa requiere:

- raster multibanda generado desde GEE;
- capa vectorial de humedales.

Ejemplo:

    data/raw/clasificacion_coberturas/rasters/S2_2018_STACK_3.tif
    data/raw/clasificacion_coberturas/vectores/ROI_HUMEDALES.shp

La capa de humedales debe contener el campo:

    nombre_ap

Este campo se usa para identificar cada humedal, disolver geometrías y nombrar los archivos de salida.

---

## Parámetros principales

El notebook permite configurar:

    year = 2018
    campo_nombre = "nombre_ap"
    nodata_val = -9999
    all_touched = False

El parámetro `all_touched` controla cómo se incluyen píxeles en los bordes:

- `False`: incluye principalmente píxeles cuyo centro cae dentro del polígono;
- `True`: incluye todos los píxeles tocados por el borde del polígono.

Usar `True` puede ser útil cuando se busca una inclusión más amplia en bordes o polígonos estrechos.

---

## Salidas esperadas

Se generan dos conjuntos de archivos intermedios.

Rasters recortados:

    data/interim/clasificacion_coberturas/RPR_2018/
    ├── humedal_1_clip.tif
    ├── humedal_2_clip.tif
    └── ...

Máscaras binarias:

    data/interim/clasificacion_coberturas/MBR_2018/
    ├── humedal_1_mask.tif
    ├── humedal_2_mask.tif
    └── ...

Las siglas se usan como nombres de trabajo:

- `RPR`: raster recortado por humedal;
- `MBR`: máscara binaria raster por humedal.

También se genera una tabla resumen con fecha de creación:

    outputs/tables/clasificacion_coberturas/resumen_recortes_humedales_2018_<timestamp>.csv

---

# 3. Preparación del raster

Notebook asociado:

    notebooks/clasificacion_coberturas/03_preparacion_raster.ipynb

Módulos asociados:

    src/clasificacion_coberturas/visualizacion.py
    src/clasificacion_coberturas/preparacion_raster.py

---

## Objetivo

Preparar el raster que será usado en el modelo de clasificación mediante visualización, análisis de correlación y reducción de variables.

---

## Entradas esperadas

Esta etapa usa un raster recortado generado en el notebook anterior:

    data/interim/clasificacion_coberturas/RPR_2018/<humedal>_clip.tif

El usuario debe definir el raster seleccionado en la celda de configuración:

    nombre_raster_seleccionado = "<humedal>_clip.tif"

---

## Visualizaciones generadas

El notebook genera y muestra:

- mosaico RGB de rasters recortados;
- mosaico de bandas del raster seleccionado;
- matriz de correlación Pearson;
- mosaico de bandas del raster reducido.

Las figuras se guardan en:

    outputs/figures/clasificacion_coberturas/

---

## Nombres de variables

Se define una lista de nombres esperados para las bandas del stack. Cuando el número de bandas coincide con el raster, el análisis usa nombres reales como:

    B2
    B3
    B4
    NDVI
    TEX_B2_contrast
    ...

Si no coincide, se generan nombres genéricos:

    b1
    b2
    b3
    ...

---

## Correlación y selección de variables

La reducción de variables se realiza con base en correlación. El flujo calcula:

- matriz de correlación Pearson;
- matriz de correlación Spearman;
- pares de variables altamente correlacionadas;
- variables seleccionadas;
- variables eliminadas.

Para evitar correlaciones artificiales, se excluyen valores NoData y fondo del raster. En los rasters recortados, el valor NoData usado es:

    nodata_val = -9999

Los parámetros principales son:

    max_pixeles = 100_000
    threshold_pares = 0.90
    threshold_seleccion = 0.925

El valor de `threshold_seleccion` puede ajustarse según el nivel de redundancia que se desee controlar. En las pruebas del flujo, un valor de `0.925` permitió conservar un número de variables cercano al comportamiento esperado del procesamiento original.

---

## Salidas esperadas

Raster reducido:

    data/interim/clasificacion_coberturas/reducidos/S2_REDUCIDO_PEARSON_2018.tif

Lista de variables seleccionadas:

    data/interim/clasificacion_coberturas/reducidos/bandas_pearson_2018.json

Tablas:

    outputs/tables/clasificacion_coberturas/pares_correlacion_pearson.csv
    outputs/tables/clasificacion_coberturas/pares_correlacion_spearman.csv
    outputs/tables/clasificacion_coberturas/variables_seleccionadas_pearson.csv

Figuras:

    outputs/figures/clasificacion_coberturas/mosaico_rasters_recortados_2018.png
    outputs/figures/clasificacion_coberturas/mosaico_bandas_<raster>.png
    outputs/figures/clasificacion_coberturas/matriz_correlacion_pearson_<raster>.png
    outputs/figures/clasificacion_coberturas/mosaico_bandas_S2_REDUCIDO_PEARSON_2018.png

---

# 4. Preparación de muestras

Notebook asociado:

    notebooks/clasificacion_coberturas/04_preparacion_muestras.ipynb

Módulo asociado:

    src/clasificacion_coberturas/muestras.py

---

## Objetivo

Preparar las muestras de entrenamiento mediante extracción de valores del raster reducido en puntos etiquetados.

---

## Entradas esperadas

Esta etapa requiere:

- raster reducido;
- archivo JSON con variables seleccionadas;
- puntos de entrenamiento.

Ejemplo:

    data/interim/clasificacion_coberturas/reducidos/S2_REDUCIDO_PEARSON_2018.tif
    data/interim/clasificacion_coberturas/reducidos/bandas_pearson_2018.json
    data/raw/clasificacion_coberturas/vectores/PUNTOS_N1.shp

La capa de puntos debe contener:

    Clase_N1
    d_nivel_1_

`Clase_N1` contiene el código numérico de clase usado como variable objetivo.

`d_nivel_1_` contiene la etiqueta descriptiva de clase, usada para visualización y revisión.

---

## Procesamiento

El notebook realiza:

- carga de puntos de entrenamiento;
- validación de columnas de clase;
- visualización interactiva de puntos por clase;
- conteo de puntos por clase;
- extracción de valores del raster en cada punto;
- construcción de `X_points` y `y_points`;
- creación de split estratificado;
- guardado de muestras y split.

El split se configura como:

    test_size = 0.3
    random_state = 42

Esto corresponde a:

- 70 % entrenamiento;
- 30 % prueba.

---

## Salidas esperadas

Tabla de muestras:

    data/interim/clasificacion_coberturas/muestras/muestras_entrenamiento_2018.csv

Split estratificado:

    data/interim/clasificacion_coberturas/splits/split_clasificacion_2018.joblib

Figura de distribución de clases:

    outputs/figures/clasificacion_coberturas/distribucion_clases_2018.png

---

# 5. Entrenamiento de modelos

Notebook asociado:

    notebooks/clasificacion_coberturas/05_entrenamiento_modelos.ipynb

Módulo asociado:

    src/clasificacion_coberturas/entrenamiento.py

---

## Objetivo

Entrenar y comparar modelos de clasificación, seleccionar el mejor modelo base y optimizarlo mediante búsqueda de hiperparámetros.

---

## Entrada esperada

Split generado en el notebook anterior:

    data/interim/clasificacion_coberturas/splits/split_clasificacion_2018.joblib

---

## Modelos base incluidos

El flujo compara los siguientes modelos:

- `CARTO_base`: Decision Tree Classifier;
- `RF_base`: Random Forest Classifier;
- `SVM_RBF_base`: Support Vector Classifier con kernel RBF;
- `GB_base`: Gradient Boosting Classifier;
- `ExtraTrees_base`: Extra Trees Classifier;
- `AdaBoost_base`: AdaBoost Classifier;
- `HistGB_base`: HistGradientBoosting Classifier;
- `KNN_base`: K-Nearest Neighbors Classifier.

El escalamiento con `StandardScaler` se aplica únicamente a modelos sensibles a la escala de las variables:

- SVM;
- KNN.

---

## Métricas de comparación

Se calculan:

- `accuracy`;
- `precision_macro`;
- `recall_macro`;
- `f1_macro`.

El mejor modelo base se selecciona según:

    f1_macro

Esta métrica es adecuada para clasificación multiclase cuando se desea evaluar el desempeño de forma equilibrada entre clases.

---

## Ajuste de hiperparámetros

El mejor modelo base se optimiza mediante:

    RandomizedSearchCV

La búsqueda usa validación cruzada estratificada:

    StratifiedKFold

La métrica de optimización es:

    f1_macro

---

## Salidas esperadas

Modelos base:

    models/clasificacion_coberturas/base/
    ├── CARTO_base.joblib
    ├── RF_base.joblib
    ├── SVM_RBF_base.joblib
    ├── GB_base.joblib
    ├── ExtraTrees_base.joblib
    ├── AdaBoost_base.joblib
    ├── HistGB_base.joblib
    └── KNN_base.joblib

Modelo optimizado:

    models/clasificacion_coberturas/modelo_clasificacion_tuning.joblib

Búsqueda de hiperparámetros:

    models/clasificacion_coberturas/busqueda_modelo_tuning.joblib

Tablas:

    outputs/tables/clasificacion_coberturas/metricas_modelos_base.csv
    outputs/tables/clasificacion_coberturas/comparacion_base_tuning.csv

Figura:

    outputs/figures/clasificacion_coberturas/radar_modelos_base.png

---

# 6. Evaluación del modelo optimizado

Notebook asociado:

    notebooks/clasificacion_coberturas/06_evaluacion_modelos.ipynb

Módulo asociado:

    src/clasificacion_coberturas/evaluacion.py

---

## Objetivo

Evaluar detalladamente el modelo optimizado generado en la etapa de entrenamiento.

---

## Entradas esperadas

Esta etapa requiere:

- split estratificado;
- modelo optimizado;
- puntos de entrenamiento para construir mapa de clases.

Ejemplo:

    data/interim/clasificacion_coberturas/splits/split_clasificacion_2018.joblib
    models/clasificacion_coberturas/modelo_clasificacion_tuning.joblib
    data/raw/clasificacion_coberturas/vectores/PUNTOS_N1.shp

---

## Evaluación

La evaluación se realiza sobre el conjunto de prueba.

Aunque la tabla de comparación pueda mostrar diferencias entre modelo base y modelo optimizado, esta etapa evalúa el modelo optimizado, ya que es el modelo definido para la fase de clasificación espacial.

Se generan:

- métricas globales;
- matriz de confusión;
- reporte de clasificación;
- importancia por permutación.

---

## Salidas esperadas

Tablas:

    outputs/tables/clasificacion_coberturas/metricas_modelo_final.csv
    outputs/tables/clasificacion_coberturas/classification_report.csv
    outputs/tables/clasificacion_coberturas/matriz_confusion.csv
    outputs/tables/clasificacion_coberturas/importancia_permutacion.csv

Figuras:

    outputs/figures/clasificacion_coberturas/matriz_confusion.png
    outputs/figures/clasificacion_coberturas/importancia_permutacion.png

---

# 7. Clasificación espacial

Notebook asociado:

    notebooks/clasificacion_coberturas/07_clasificacion_espacial.ipynb

Módulo asociado:

    src/clasificacion_coberturas/inferencia.py

---

## Objetivo

Aplicar el modelo optimizado sobre el raster reducido para generar mapas clasificados de coberturas.

---

## Entradas esperadas

Esta etapa requiere:

- split estratificado;
- modelo optimizado;
- raster reducido;
- capa de puntos para mapa de clases.

Ejemplo:

    data/interim/clasificacion_coberturas/splits/split_clasificacion_2018.joblib
    models/clasificacion_coberturas/modelo_clasificacion_tuning.joblib
    data/interim/clasificacion_coberturas/reducidos/S2_REDUCIDO_PEARSON_2018.tif
    data/raw/clasificacion_coberturas/vectores/PUNTOS_N1.shp

---

## Lógica espacial

El raster reducido se reorganiza desde:

    bandas × filas × columnas

hacia:

    píxeles × variables

Luego, el modelo genera una predicción por píxel. Finalmente, las predicciones se reconstruyen con la forma espacial original del raster.

---

## Filtro de mayoría

El flujo incluye un filtro de mayoría configurable:

    aplicar_filtro_mayoria = True
    majority_size = 3

Este filtro suaviza el mapa clasificado reemplazando cada píxel por la clase dominante en una ventana local.

---

## Salidas esperadas

Rasters:

    outputs/rasters/clasificacion_coberturas/clasificacion_coberturas_2018_clasificado.tif
    outputs/rasters/clasificacion_coberturas/clasificacion_coberturas_2018_clasificado_filtrado.tif

Figuras:

    outputs/figures/clasificacion_coberturas/clasificacion_coberturas_2018_clasificado.png
    outputs/figures/clasificacion_coberturas/clasificacion_coberturas_2018_clasificado_filtrado.png

---

## Entradas crudas principales del flujo

Los insumos crudos principales son:

    data/raw/clasificacion_coberturas/vectores/PUNTOS_N1.shp
    data/raw/clasificacion_coberturas/vectores/ROI_BOGOTA.shp
    data/raw/clasificacion_coberturas/vectores/ROI_HUMEDALES.shp
    data/raw/clasificacion_coberturas/rasters/S2_2018_STACK_3.tif
    data/raw/clasificacion_coberturas/rasters/S2_2024_STACK_3.tif

Para los shapefiles, deben conservarse todos los archivos asociados (`.shp`, `.shx`, `.dbf`, `.prj`, `.cpg`, entre otros).

---

## Orden recomendado de ejecución

El flujo debe ejecutarse en el siguiente orden:

1. `01_descarga_gee.ipynb`
2. `02_recorte_humedales.ipynb`
3. `03_preparacion_raster.ipynb`
4. `04_preparacion_muestras.ipynb`
5. `05_entrenamiento_modelos.ipynb`
6. `06_evaluacion_modelos.ipynb`
7. `07_clasificacion_espacial.ipynb`

Cada notebook depende de productos generados en etapas previas.

---

## Consideraciones metodológicas

- Los stacks Sentinel-2 exportados desde GEE son insumos crudos del flujo local.
- La visualización interactiva con `geemap` depende de `ipyleaflet`, `jupyter_leaflet`, `ipywidgets` y `jupyterlab_widgets`.
- El kernel del entorno debe registrarse manualmente para que JupyterLab use el entorno correcto.
- La reducción por correlación depende del raster seleccionado, el tratamiento de NoData y el umbral definido.
- El valor NoData `-9999` debe excluirse del cálculo de correlaciones para evitar relaciones artificialmente altas.
- Las métricas de clasificación deben interpretarse junto con la distribución de clases.
- El mapa clasificado final representa una estimación del modelo, no una observación directa.
- El filtro de mayoría mejora la apariencia espacial del mapa, pero también puede suavizar detalles finos de cobertura.

---

## Estado del flujo

El flujo de clasificación de coberturas se encuentra estructurado, modularizado, documentado y ejecutado dentro del repositorio `humedales_bogota_ml`.
