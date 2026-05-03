# Guía de uso

Este documento describe cómo ejecutar los flujos principales del repositorio `humedales_bogota_ml`.

El repositorio contiene dos líneas de trabajo:

1. **Modelado de calidad de agua**.
2. **Clasificación de coberturas**.

Cada flujo tiene notebooks, módulos, entradas y salidas específicas. Se recomienda ejecutar los notebooks en el orden indicado, ya que cada etapa depende de productos generados previamente.

---

## Recomendaciones generales

Antes de ejecutar cualquier notebook, se recomienda:

1. Instalar el entorno correspondiente.
2. Registrar el kernel del entorno en JupyterLab.
3. Abrir JupyterLab desde el entorno que se va a usar.
4. Seleccionar el kernel correcto en cada notebook.
5. Revisar las rutas de entrada.
6. Ejecutar los notebooks en orden.
7. Verificar las salidas generadas después de cada etapa.

---

## Activación de entornos

### Calidad de agua

Para ejecutar el flujo de calidad de agua:

    conda activate humedales_calidad_agua
    python -m jupyter lab

Kernel recomendado:

    Python (humedales_calidad_agua)

---

### Clasificación de coberturas

Para ejecutar el flujo de clasificación de coberturas:

    conda activate humedales_clasificacion
    python -m jupyter lab

Kernel recomendado:

    Python (humedales_clasificacion)

---

## Verificación del entorno activo

Dentro de un notebook se puede verificar el entorno activo con:

    import sys
    print(sys.executable)

La ruta debe apuntar al entorno correspondiente.

Para clasificación de coberturas debería apuntar a:

    .../envs/humedales_clasificacion/python.exe

Para calidad de agua debería apuntar a:

    .../envs/humedales_calidad_agua/python.exe

---

# Uso del flujo de calidad de agua

El flujo de calidad de agua permite estimar parámetros fisicoquímicos a partir de variables espectrales derivadas de imágenes multibanda.

Los notebooks se encuentran en:

    notebooks/calidad_agua/

---

## Orden de ejecución

El flujo debe ejecutarse en este orden:

1. `01_extraccion_variables.ipynb`
2. `02_preparacion_dataset_modelos.ipynb`
3. `03_entrenamiento_modelos.ipynb`
4. `04_evaluacion_modelos.ipynb`
5. `05_analisis_importancia_variables.ipynb`
6. `06_prediccion_espacial_modelos.ipynb`

---

## 0. Estadísticas descriptivas de laboratorio

Notebook:

    notebooks/calidad_agua/00_estadisticas_laboratorio.ipynb

### Propósito

Este notebook realiza un análisis exploratorio de los parámetros de laboratorio asociados al flujo de calidad de agua.

No hace parte obligatoria del pipeline de entrenamiento, pero permite revisar previamente:

- estadísticas descriptivas;
- valores faltantes;
- distribución de parámetros;
- diagramas de caja;
- matriz de correlación entre parámetros de laboratorio;
- posibles valores extremos.

### Entrada principal

Por defecto usa el archivo procesado del flujo de calidad de agua:

    data/processed/calidad_agua/Puntos_Muestreo_Reflectancia_Indices_Excel_2.gpkg

También puede adaptarse para leer archivos `.xlsx`, `.csv`, `.shp`, `.gpkg` o `.geojson`.

### Parámetros analizados

El notebook está configurado para revisar:

    DQO
    pH
    Fosfatos
    CE
    Turbidez
    Nitratos
    Sulfatos
    ficocianin
    Chl

### Salidas principales

Tablas:

    outputs/tables/calidad_agua/estadisticas_laboratorio/
    ├── resumen_estadistico_laboratorio.csv
    ├── valores_faltantes_laboratorio.csv
    ├── correlacion_parametros_laboratorio.csv
    ├── resumen_outliers_laboratorio.csv
    └── dataset_laboratorio_numerico.csv

Figuras:

    outputs/figures/calidad_agua/estadisticas_laboratorio/
    ├── histogramas_parametros_laboratorio.png
    ├── boxplots_parametros_laboratorio.png
    └── correlacion_parametros_laboratorio.png


## 1. Extracción de variables espectrales

Notebook:

    notebooks/calidad_agua/01_extraccion_variables.ipynb

Módulo principal:

    src/calidad_agua/extraccion_reflectancia.py

### Propósito

Extraer reflectancias promedio e índices espectrales desde el ortomosaico multibanda para los puntos de muestreo de calidad de agua.

### Entradas requeridas

- ortomosaico multibanda;
- puntos de muestreo;
- datos de laboratorio integrados en el shapefile o en archivo Excel;
- ruta de salida procesada.

El flujo puede trabajar con un shapefile de puntos que ya contenga los parámetros de laboratorio. En ese caso, el archivo Excel no es obligatorio.

### Salida principal

    data/processed/calidad_agua/Puntos_Muestreo_Reflectancia_Indices_Excel_2.gpkg

Este archivo contiene:

- atributos originales;
- parámetros de calidad de agua;
- reflectancias promedio;
- índices espectrales.

---

## 2. Preparación del dataset

Notebook:

    notebooks/calidad_agua/02_preparacion_dataset_modelos.ipynb

Módulo principal:

    src/calidad_agua/preparacion.py

### Propósito

Preparar el dataset usado para entrenar, evaluar e interpretar los modelos de calidad de agua.

### Configuraciones clave

Revisar especialmente:

- ruta del archivo procesado;
- lista de variables predictoras;
- variable objetivo;
- uso de variables transformadas con logaritmo natural;
- número de registros;
- proporción de entrenamiento y prueba.

### Variable objetivo

El flujo diferencia entre:

    target_modelo
    target_salida

Ejemplo:

    target_modelo = "ln_DQO"
    target_salida = "DQO"

Esto permite entrenar con `ln_DQO`, pero organizar salidas bajo el parámetro `DQO`.

### Salidas principales

Split de entrenamiento y prueba:

    data/interim/calidad_agua/splits/split_DQO.joblib

Matrices de correlación:

    outputs/tables/calidad_agua/DQO/matriz_correlacion_original.csv
    outputs/tables/calidad_agua/DQO/matriz_correlacion_transformada.csv

Figuras:

    outputs/figures/calidad_agua/DQO/matriz_correlacion_original.png
    outputs/figures/calidad_agua/DQO/matriz_correlacion_transformada.png

---

## 3. Entrenamiento de modelos

Notebook:

    notebooks/calidad_agua/03_entrenamiento_modelos.ipynb

Módulo principal:

    src/calidad_agua/entrenamiento.py

### Propósito

Entrenar modelos de regresión para el parámetro seleccionado.

### Modelos incluidos

- SVR;
- GBR;
- RFR.

### Entrada requerida

    data/interim/calidad_agua/splits/split_DQO.joblib

### Salidas principales

Modelos entrenados:

    models/calidad_agua/entrenados/DQO/
    ├── SVR.joblib
    ├── GBR.joblib
    └── RFR.joblib

Resultados de búsqueda:

    models/calidad_agua/metricas/DQO/gridsearch_modelos.joblib

---

## 4. Evaluación de modelos

Notebook:

    notebooks/calidad_agua/04_evaluacion_modelos.ipynb

Módulo principal:

    src/calidad_agua/evaluacion.py

### Propósito

Evaluar y comparar el desempeño de los modelos entrenados.

### Métricas calculadas

- R²;
- RMSE;
- MAE.

### Salidas principales

Tabla de métricas:

    outputs/tables/calidad_agua/DQO/metricas_modelos.csv

Figuras:

    outputs/figures/calidad_agua/DQO/
    ├── observado_estimado_SVR.png
    ├── observado_estimado_GBR.png
    ├── observado_estimado_RFR.png
    ├── comparacion_rmse_modelos.png
    ├── comparacion_metricas_modelos.png
    └── deviance_gbr.png

---

## 5. Análisis de importancia de variables

Notebook:

    notebooks/calidad_agua/05_analisis_importancia_variables.ipynb

Módulo principal:

    src/calidad_agua/interpretacion.py

### Propósito

Analizar la importancia relativa de las variables predictoras en modelos compatibles con `feature_importances_`.

### Modelos incluidos

- GBR;
- RFR.

SVR no se incluye en este análisis porque no cuenta directamente con `feature_importances_`.

### Salidas principales

Tablas:

    outputs/tables/calidad_agua/DQO/
    ├── importancia_variables_GBR.csv
    ├── importancia_variables_RFR.csv
    └── importancia_variables_consolidado.csv

Figuras:

    outputs/figures/calidad_agua/DQO/
    ├── importancia_variables_GBR.png
    └── importancia_variables_RFR.png

---

## 6. Predicción espacial

Notebook:

    notebooks/calidad_agua/06_prediccion_espacial_modelos.ipynb

Módulo principal:

    src/calidad_agua/inferencia.py

### Propósito

Aplicar los modelos entrenados sobre un ortomosaico multibanda para generar superficies espaciales continuas del parámetro estimado.

### Salidas principales

Rasters:

    outputs/rasters/calidad_agua/DQO/
    ├── DQO_SVR.tif
    ├── DQO_GBR.tif
    └── DQO_RFR.tif

Figura:

    outputs/figures/calidad_agua/DQO/mapas_prediccion_modelos.png

---

# Uso del flujo de clasificación de coberturas

El flujo de clasificación de coberturas permite generar mapas temáticos de coberturas a partir de imágenes Sentinel-2, índices espectrales, texturas GLCM y puntos de entrenamiento.

Los notebooks se encuentran en:

    notebooks/clasificacion_coberturas/

---

## Orden de ejecución

El flujo debe ejecutarse en este orden:

1. `01_descarga_gee.ipynb`
2. `02_recorte_humedales.ipynb`
3. `03_preparacion_raster.ipynb`
4. `04_preparacion_muestras.ipynb`
5. `05_entrenamiento_modelos.ipynb`
6. `06_evaluacion_modelos.ipynb`
7. `07_clasificacion_espacial.ipynb`

---

## Entradas crudas principales

Los insumos crudos esperados son:

    data/raw/clasificacion_coberturas/vectores/PUNTOS_N1.shp
    data/raw/clasificacion_coberturas/vectores/ROI_BOGOTA.shp
    data/raw/clasificacion_coberturas/vectores/ROI_HUMEDALES.shp
    data/raw/clasificacion_coberturas/rasters/S2_2018_STACK_3.tif
    data/raw/clasificacion_coberturas/rasters/S2_2024_STACK_3.tif

Para archivos shapefile, deben conservarse todos los componentes asociados:

    .shp
    .shx
    .dbf
    .prj
    .cpg

---

## 1. Descarga desde Google Earth Engine

Notebook:

    notebooks/clasificacion_coberturas/01_descarga_gee.ipynb

Módulo principal:

    src/clasificacion_coberturas/gee.py

### Propósito

Generar y exportar un stack Sentinel-2 desde Google Earth Engine.

### Requisitos específicos

- cuenta de Google Earth Engine activa;
- proyecto de Google Cloud registrado para Earth Engine;
- permisos para usar el proyecto;
- permisos de exportación hacia Drive, Asset o Cloud Storage.

### Configuración principal

Revisar en el notebook:

    GEE_PROJECT
    year
    cloud_pct
    scale
    glcm_size
    tex_bands
    nombre_salida
    modo_exportacion

### Modos de exportación

Opciones disponibles:

    drive
    asset
    cloud_storage

La opción por defecto es:

    modo_exportacion = "drive"
    drive_folder = "GEE_Coberturas_Humedales"

### Salidas principales

Stack exportado desde GEE:

    data/raw/clasificacion_coberturas/rasters/S2_2018_STACK_3.tif
    data/raw/clasificacion_coberturas/rasters/S2_2024_STACK_3.tif

Tabla local de bandas:

    outputs/tables/clasificacion_coberturas/bandas_stack_S2_2018_STACK_3.csv

---

## 2. Recorte de humedales

Notebook:

    notebooks/clasificacion_coberturas/02_recorte_humedales.ipynb

Módulo principal:

    src/clasificacion_coberturas/procesamiento_raster.py

### Propósito

Recortar el stack multibanda por humedal y generar máscaras binarias alineadas.

### Entradas

    data/raw/clasificacion_coberturas/rasters/S2_2018_STACK_3.tif
    data/raw/clasificacion_coberturas/vectores/ROI_HUMEDALES.shp

La capa de humedales debe contener:

    nombre_ap

### Configuraciones clave

    year = 2018
    campo_nombre = "nombre_ap"
    nodata_val = -9999
    all_touched = False

### Salidas principales

Rasters recortados:

    data/interim/clasificacion_coberturas/RPR_2018/

Máscaras binarias:

    data/interim/clasificacion_coberturas/MBR_2018/

Resumen:

    outputs/tables/clasificacion_coberturas/resumen_recortes_humedales_2018_<timestamp>.csv

---

## 3. Preparación del raster

Notebook:

    notebooks/clasificacion_coberturas/03_preparacion_raster.ipynb

Módulos principales:

    src/clasificacion_coberturas/visualizacion.py
    src/clasificacion_coberturas/preparacion_raster.py

### Propósito

Visualizar el raster, analizar correlaciones, seleccionar variables y generar un raster reducido.

### Entrada principal

    data/interim/clasificacion_coberturas/RPR_2018/<humedal>_clip.tif

### Configuraciones clave

    nombre_raster_seleccionado = "<humedal>_clip.tif"
    max_pixeles = 100_000
    threshold_pares = 0.90
    threshold_seleccion = 0.925
    nodata_val = -9999
    excluir_fondo_cero = True

### Notas de uso

- `stretch=True` permite visualizar mejor las bandas del raster.
- `nodata_val=-9999` debe excluirse para evitar correlaciones artificiales.
- `threshold_seleccion` controla el nivel de reducción de redundancia entre variables.

### Salidas principales

Raster reducido:

    data/interim/clasificacion_coberturas/reducidos/S2_REDUCIDO_PEARSON_2018.tif

Variables seleccionadas:

    data/interim/clasificacion_coberturas/reducidos/bandas_pearson_2018.json

Tablas:

    outputs/tables/clasificacion_coberturas/pares_correlacion_pearson.csv
    outputs/tables/clasificacion_coberturas/pares_correlacion_spearman.csv
    outputs/tables/clasificacion_coberturas/variables_seleccionadas_pearson.csv

Figuras:

    outputs/figures/clasificacion_coberturas/

---

## 4. Preparación de muestras

Notebook:

    notebooks/clasificacion_coberturas/04_preparacion_muestras.ipynb

Módulo principal:

    src/clasificacion_coberturas/muestras.py

### Propósito

Extraer valores del raster reducido en puntos de entrenamiento y generar el split estratificado.

### Entradas

    data/interim/clasificacion_coberturas/reducidos/S2_REDUCIDO_PEARSON_2018.tif
    data/interim/clasificacion_coberturas/reducidos/bandas_pearson_2018.json
    data/raw/clasificacion_coberturas/vectores/PUNTOS_N1.shp

La capa de puntos debe contener:

    Clase_N1
    d_nivel_1_

### Configuraciones clave

    columna_clase = "Clase_N1"
    columna_clase_nombre = "d_nivel_1_"
    test_size = 0.3
    random_state = 42
    normalizar = True

### Salidas principales

Muestras:

    data/interim/clasificacion_coberturas/muestras/muestras_entrenamiento_2018.csv

Split:

    data/interim/clasificacion_coberturas/splits/split_clasificacion_2018.joblib

Figura:

    outputs/figures/clasificacion_coberturas/distribucion_clases_2018.png

---

## 5. Entrenamiento de modelos

Notebook:

    notebooks/clasificacion_coberturas/05_entrenamiento_modelos.ipynb

Módulo principal:

    src/clasificacion_coberturas/entrenamiento.py

### Propósito

Entrenar modelos base, compararlos, seleccionar el mejor modelo base y optimizarlo.

### Entrada

    data/interim/clasificacion_coberturas/splits/split_clasificacion_2018.joblib

### Modelos incluidos

- CARTO;
- Random Forest;
- SVM con kernel RBF;
- Gradient Boosting;
- Extra Trees;
- AdaBoost;
- HistGradientBoosting;
- KNN.

### Métrica de selección

    f1_macro

### Salidas principales

Modelos base:

    models/clasificacion_coberturas/base/

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

## 6. Evaluación de modelos

Notebook:

    notebooks/clasificacion_coberturas/06_evaluacion_modelos.ipynb

Módulo principal:

    src/clasificacion_coberturas/evaluacion.py

### Propósito

Evaluar el modelo optimizado mediante métricas globales, matriz de confusión, reporte de clasificación e importancia por permutación.

### Entradas

    data/interim/clasificacion_coberturas/splits/split_clasificacion_2018.joblib
    models/clasificacion_coberturas/modelo_clasificacion_tuning.joblib
    data/raw/clasificacion_coberturas/vectores/PUNTOS_N1.shp

### Nota de uso

Esta etapa evalúa el modelo optimizado, ya que es el modelo definido para la fase de clasificación espacial.

### Salidas principales

Tablas:

    outputs/tables/clasificacion_coberturas/metricas_modelo_final.csv
    outputs/tables/clasificacion_coberturas/classification_report.csv
    outputs/tables/clasificacion_coberturas/matriz_confusion.csv
    outputs/tables/clasificacion_coberturas/importancia_permutacion.csv

Figuras:

    outputs/figures/clasificacion_coberturas/matriz_confusion.png
    outputs/figures/clasificacion_coberturas/importancia_permutacion.png

---

## 7. Clasificación espacial

Notebook:

    notebooks/clasificacion_coberturas/07_clasificacion_espacial.ipynb

Módulo principal:

    src/clasificacion_coberturas/inferencia.py

### Propósito

Aplicar el modelo optimizado sobre el raster reducido para generar mapas clasificados de coberturas.

### Entradas

    data/interim/clasificacion_coberturas/splits/split_clasificacion_2018.joblib
    models/clasificacion_coberturas/modelo_clasificacion_tuning.joblib
    data/interim/clasificacion_coberturas/reducidos/S2_REDUCIDO_PEARSON_2018.tif

### Configuraciones clave

    nombre_salida = "clasificacion_coberturas_2018"
    aplicar_filtro_mayoria = True
    majority_size = 3
    nodata_salida = 0

### Salidas principales

Rasters:

    outputs/rasters/clasificacion_coberturas/clasificacion_coberturas_2018_clasificado.tif
    outputs/rasters/clasificacion_coberturas/clasificacion_coberturas_2018_clasificado_filtrado.tif

Figuras:

    outputs/figures/clasificacion_coberturas/clasificacion_coberturas_2018_clasificado.png
    outputs/figures/clasificacion_coberturas/clasificacion_coberturas_2018_clasificado_filtrado.png

---

# Revisión de rutas

Los notebooks usan rutas relativas a la raíz del repositorio mediante `BASE_DIR`.

Antes de ejecutar, verificar que los archivos requeridos estén ubicados en las carpetas esperadas.

Si se cambia el nombre de un archivo de entrada, debe ajustarse la celda de configuración correspondiente.

---

# Recomendaciones para nuevos usuarios

1. No modificar directamente los archivos en `data/raw/`.
2. Mantener la estructura de carpetas del repositorio.
3. Ejecutar los notebooks en orden.
4. Seleccionar el kernel correspondiente antes de ejecutar.
5. Revisar salidas generadas después de cada notebook.
6. No interpretar mapas clasificados o mapas de predicción sin revisar las métricas del modelo.
7. Documentar cualquier cambio en rutas, parámetros, variables o modelos.

---

# Productos finales principales

## Calidad de agua

- modelos entrenados de regresión;
- métricas comparativas;
- matrices de correlación;
- importancia de variables;
- raster de predicción por modelo;
- visualización de mapas estimados.

## Clasificación de coberturas

- stack Sentinel-2 generado desde GEE;
- rasters recortados por humedal;
- raster reducido por correlación;
- muestras extraídas;
- split estratificado;
- modelos base y modelo optimizado;
- matriz de confusión;
- importancia por permutación;
- mapa clasificado;
- mapa clasificado filtrado.

---

## Estado actual

Esta guía cubre el uso de los dos flujos principales del repositorio:

    calidad_agua
    clasificacion_coberturas

# Uso del script de publicación en ArcGIS Online

Además de los flujos de modelado, el repositorio incluye un script complementario para migrar o clonar un StoryMap de ArcGIS Online o ArcGIS Enterprise Portal.

Este script no hace parte del flujo de modelado de calidad de agua ni del flujo de clasificación de coberturas. Su propósito es apoyar la fase de publicación o transferencia de productos en ArcGIS.

## Ubicación del script

El script se encuentra en:

    scripts/publicacion/migracion_storymap_agol.py

## Entorno recomendado

Debido a que la librería `arcgis` puede ser pesada y no es necesaria para los modelos, se usa un entorno independiente:

    config/publicacion_arcgis_config.yml

Crear el entorno:

    conda env create -f config/publicacion_arcgis_config.yml

Activar el entorno:

    conda activate humedales_publicacion_arcgis

## Propósito del script

El script permite:

- conectarse a una cuenta o portal origen de ArcGIS;
- conectarse a una cuenta o portal destino;
- obtener un StoryMap mediante su ID;
- identificar dependencias del StoryMap;
- clonar el StoryMap y sus elementos asociados;
- aplicar una reparación opcional sobre nodos tipo `tour-map`;
- generar una URL del StoryMap clonado para revisión y publicación manual.

## Uso básico

Ejecutar desde la raíz del repositorio:

    python scripts/publicacion/migracion_storymap_agol.py --story-id <ID_DEL_STORYMAP>

El script solicitará usuario y contraseña de la cuenta origen y destino mediante consola.

## Uso con variables de entorno

También puede ejecutarse usando variables de entorno para no ingresar credenciales manualmente en cada ejecución.

Variables requeridas:

    AGOL_SOURCE_USER
    AGOL_SOURCE_PASSWORD
    AGOL_TARGET_USER
    AGOL_TARGET_PASSWORD

Ejecución:

    python scripts/publicacion/migracion_storymap_agol.py --story-id <ID_DEL_STORYMAP> --use-env

## Opciones adicionales

No aplicar reparación de Map Tours:

    python scripts/publicacion/migracion_storymap_agol.py --story-id <ID_DEL_STORYMAP> --no-reparar-tours

No copiar datos asociados:

    python scripts/publicacion/migracion_storymap_agol.py --story-id <ID_DEL_STORYMAP> --no-copy-data

No reutilizar elementos existentes:

    python scripts/publicacion/migracion_storymap_agol.py --story-id <ID_DEL_STORYMAP> --no-search-existing

## Consideraciones

- No se deben escribir usuarios ni contraseñas directamente en el script.
- La cuenta origen debe tener acceso al StoryMap original.
- La cuenta destino debe tener permisos para crear contenido.
- Después de la migración, el StoryMap clonado debe abrirse, revisarse y publicarse manualmente desde el editor de ArcGIS StoryMaps.
- La reparación de nodos `tour-map` es una corrección específica para ciertos problemas de Map Tours después de la clonación.
- Este script debe considerarse una utilidad de publicación, no una etapa analítica del pipeline.