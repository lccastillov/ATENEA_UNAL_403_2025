# Arquitectura del repositorio

Este documento describe la estructura general del repositorio `humedales_bogota_ml`, la función de cada carpeta y la organización de los flujos de modelado incluidos.

El repositorio está diseñado para documentar, ejecutar y mantener dos flujos principales:

1. **Modelado de calidad de agua**.
2. **Clasificación de coberturas**.
3. **Publicación y migración de productos en ArcGIS Online / Portal**.

La arquitectura busca separar claramente:

- datos crudos;
- datos intermedios;
- datos procesados;
- notebooks;
- funciones reutilizables;
- modelos entrenados;
- salidas tabulares, gráficas y raster;
- documentación;
- archivos de configuración.

---

## Estructura general

La estructura general del repositorio es:

    humedales_bogota_ml/
    ├── README.md
    ├── LICENSE
    ├── .gitignore
    │
    ├── config/
    │   ├── calidad_agua_config.yml
    │   ├── clasificacion_config.yml
    │   └── publicacion_arcgis_config.yml
    │
    ├── data/
    │   ├── raw/
    │   ├── interim/
    │   └── processed/
    │
    ├── docs/
    │   ├── arquitectura_repositorio.md
    │   ├── flujo_calidad_agua.md
    │   ├── flujo_clasificacion_coberturas.md
    │   ├── guia_instalacion.md
    │   └── guia_uso.md
    │
    ├── models/
    │   ├── calidad_agua/
    │   └── clasificacion_coberturas/
    │
    ├── notebooks/
    │   ├── calidad_agua/
    │   └── clasificacion_coberturas/
    │
    ├── outputs/
    │   ├── figures/
    │   ├── tables/
    │   └── rasters/
    │
    ├── scripts/
    │   └── publicacion/
    │       └── migracion_storymap_agol.py
    │
    └── src/
        ├── calidad_agua/
        └── clasificacion_coberturas/

---

# 1. Archivos principales de la raíz

## `README.md`

Archivo principal de presentación del repositorio.

Debe incluir:

- objetivo general;
- descripción de los flujos disponibles;
- estructura resumida del repositorio;
- instrucciones básicas de instalación;
- orden general de ejecución;
- estado del proyecto.

---

## `LICENSE`

Archivo de licencia del repositorio.

Define las condiciones bajo las cuales se puede usar, modificar o distribuir el código y la documentación.

Antes de publicar el repositorio, debe verificarse que la licencia seleccionada sea compatible con las políticas de la entidad o del proyecto.

---

## `.gitignore`

Archivo que define qué elementos no deben subirse al repositorio.

Debe excluir especialmente:

- archivos temporales;
- cachés de Python;
- checkpoints de Jupyter;
- entornos locales;
- archivos raster pesados;
- salidas voluminosas;
- modelos pesados, si no se desea versionarlos.

La finalidad es evitar que el repositorio incluya archivos muy grandes o productos que puedan regenerarse ejecutando el flujo.

---

# 2. Carpeta `config/`

La carpeta `config/` contiene los archivos de configuración de entornos.

    config/
    ├── calidad_agua_config.yml
    └── clasificacion_config.yml

---

## `calidad_agua_config.yml`

Define el entorno de conda para ejecutar el flujo de modelado de calidad de agua.

Incluye dependencias para:

- notebooks;
- ciencia de datos;
- machine learning;
- visualización;
- manejo de datos geoespaciales.

Entorno esperado:

    humedales_calidad_agua

---

## `clasificacion_config.yml`

Define el entorno de conda para ejecutar el flujo de clasificación de coberturas.

Incluye dependencias para:

- notebooks;
- ciencia de datos;
- machine learning;
- visualización;
- manejo de datos geoespaciales;
- Google Earth Engine;
- mapas interactivos con `geemap` e `ipyleaflet`.

Entorno esperado:

    humedales_clasificacion

Dependencias importantes del flujo de clasificación:

    earthengine-api
    geemap
    ipyleaflet
    jupyter_leaflet
    jupyterlab_widgets

El kernel de cada entorno debe registrarse manualmente para aparecer en JupyterLab.

---

## `publicacion_arcgis_config.yml`

Define un entorno independiente para ejecutar utilidades de publicación en ArcGIS Online o ArcGIS Enterprise Portal.

Entorno esperado:

    humedales_publicacion_arcgis

Dependencia principal:

    arcgis

Este entorno se mantiene separado de los entornos de modelado porque la librería `arcgis` puede ser pesada y no es necesaria para entrenar modelos ni generar mapas clasificados.

---

# 3. Carpeta `data/`

La carpeta `data/` organiza los datos del proyecto en tres niveles:

    data/
    ├── raw/
    ├── interim/
    └── processed/

---

## `data/raw/`

Contiene datos crudos o insumos originales.

Estos archivos no deberían modificarse directamente. Si se requiere transformar un insumo, el resultado debe guardarse en `data/interim/` o `data/processed/`.

Estructura esperada:

    data/raw/
    ├── calidad_agua/
    └── clasificacion_coberturas/

---

### `data/raw/calidad_agua/`

Contiene los insumos crudos del flujo de calidad de agua.

Puede incluir:

- ortomosaicos multibanda;
- puntos de muestreo;
- archivos vectoriales;
- datos de laboratorio;
- insumos espaciales necesarios para extraer reflectancias e índices.

La estructura interna puede adaptarse según los insumos disponibles.

---

### `data/raw/clasificacion_coberturas/`

Contiene los insumos crudos del flujo de clasificación de coberturas.

Estructura esperada:

    data/raw/clasificacion_coberturas/
    ├── rasters/
    └── vectores/

Entradas principales:

    data/raw/clasificacion_coberturas/vectores/PUNTOS_N1.shp
    data/raw/clasificacion_coberturas/vectores/ROI_BOGOTA.shp
    data/raw/clasificacion_coberturas/vectores/ROI_HUMEDALES.shp
    data/raw/clasificacion_coberturas/rasters/S2_2018_STACK_3.tif
    data/raw/clasificacion_coberturas/rasters/S2_2024_STACK_3.tif

Los archivos `S2_2018_STACK_3.tif` y `S2_2024_STACK_3.tif` corresponden a stacks multibanda generados desde Google Earth Engine.

Para archivos shapefile, deben conservarse todos sus componentes asociados:

    .shp
    .shx
    .dbf
    .prj
    .cpg

---

## `data/interim/`

Contiene productos intermedios generados durante la ejecución de los flujos.

Estos productos pueden ser necesarios para notebooks posteriores, pero normalmente pueden regenerarse si se ejecuta el flujo completo.

Estructura esperada:

    data/interim/
    ├── calidad_agua/
    └── clasificacion_coberturas/

---

### `data/interim/calidad_agua/`

Contiene productos intermedios del flujo de calidad de agua.

Ejemplo:

    data/interim/calidad_agua/splits/
    └── split_DQO.joblib

El split permite reutilizar exactamente la misma partición de entrenamiento y prueba en entrenamiento, evaluación e inferencia.

---

### `data/interim/clasificacion_coberturas/`

Contiene productos intermedios del flujo de clasificación.

Estructura esperada:

    data/interim/clasificacion_coberturas/
    ├── RPR_2018/
    ├── MBR_2018/
    ├── reducidos/
    ├── muestras/
    └── splits/

---

#### `RPR_<año>/`

Contiene raster recortados por humedal.

Ejemplo:

    data/interim/clasificacion_coberturas/RPR_2018/
    └── <humedal>_clip.tif

`RPR` se usa como nombre de trabajo para raster recortado por humedal.

---

#### `MBR_<año>/`

Contiene máscaras binarias por humedal.

Ejemplo:

    data/interim/clasificacion_coberturas/MBR_2018/
    └── <humedal>_mask.tif

`MBR` se usa como nombre de trabajo para máscara binaria raster por humedal.

---

#### `reducidos/`

Contiene rasters reducidos por selección de variables y archivos auxiliares.

Ejemplo:

    data/interim/clasificacion_coberturas/reducidos/
    ├── S2_REDUCIDO_PEARSON_2018.tif
    └── bandas_pearson_2018.json

---

#### `muestras/`

Contiene tablas con valores extraídos desde el raster en puntos de entrenamiento.

Ejemplo:

    data/interim/clasificacion_coberturas/muestras/
    └── muestras_entrenamiento_2018.csv

---

#### `splits/`

Contiene splits estratificados de entrenamiento y prueba.

Ejemplo:

    data/interim/clasificacion_coberturas/splits/
    └── split_clasificacion_2018.joblib

---

## `data/processed/`

Contiene productos procesados que pueden considerarse datos finales o consolidados antes del modelado.

Estructura esperada:

    data/processed/
    ├── calidad_agua/
    └── clasificacion_coberturas/

---

### `data/processed/calidad_agua/`

Contiene productos consolidados del flujo de calidad de agua.

Ejemplo:

    data/processed/calidad_agua/
    └── Puntos_Muestreo_Reflectancia_Indices_Excel_2.gpkg

Este archivo integra:

- puntos de muestreo;
- parámetros de laboratorio;
- reflectancias promedio;
- índices espectrales.

---

### `data/processed/clasificacion_coberturas/`

Puede usarse para productos consolidados de clasificación si en el futuro se genera un dataset final no intermedio.

Actualmente, la mayoría de productos de clasificación se organizan en `data/interim/`, `models/` y `outputs/`.

---

# 4. Carpeta `docs/`

La carpeta `docs/` contiene la documentación técnica del repositorio.

    docs/
    ├── arquitectura_repositorio.md
    ├── flujo_calidad_agua.md
    ├── flujo_clasificacion_coberturas.md
    ├── guia_instalacion.md
    └── guia_uso.md

---

## `arquitectura_repositorio.md`

Describe la estructura general del repositorio y la función de cada carpeta.

---

## `flujo_calidad_agua.md`

Describe el flujo completo de modelado de calidad de agua.

Incluye:

- extracción de variables espectrales;
- preparación de dataset;
- entrenamiento de modelos;
- evaluación;
- importancia de variables;
- predicción espacial.

---

## `flujo_clasificacion_coberturas.md`

Describe el flujo completo de clasificación de coberturas.

Incluye:

- descarga desde GEE;
- recorte de humedales;
- preparación de raster;
- preparación de muestras;
- entrenamiento;
- evaluación;
- clasificación espacial.

---

## `guia_instalacion.md`

Describe cómo instalar y registrar los entornos de trabajo.

Incluye:

- entorno de calidad de agua;
- entorno de clasificación de coberturas;
- registro de kernels;
- configuración básica de Google Earth Engine;
- solución de problemas de visualización interactiva.

---

## `guia_uso.md`

Describe cómo ejecutar los notebooks y qué productos genera cada etapa.

---

# 5. Carpeta `notebooks/`

La carpeta `notebooks/` contiene los notebooks ejecutables del proyecto.

    notebooks/
    ├── calidad_agua/
    └── clasificacion_coberturas/

Los notebooks funcionan como interfaz documentada de ejecución. La lógica principal debe estar centralizada en `src/`.

---

## `notebooks/calidad_agua/`

Contiene los notebooks del flujo de modelado de calidad de agua.

    notebooks/calidad_agua/
    ├── 00_estadisticas_laboratorio.ipynb
    ├── 01_extraccion_variables.ipynb
    ├── 02_preparacion_dataset_modelos.ipynb
    ├── 03_entrenamiento_modelos.ipynb
    ├── 04_evaluacion_modelos.ipynb
    ├── 05_analisis_importancia_variables.ipynb
    └── 06_prediccion_espacial_modelos.ipynb

### `00_estadisticas_laboratorio.ipynb`

Notebook complementario para explorar los parámetros de laboratorio antes o después del flujo principal de modelado. Genera estadísticas descriptivas, valores faltantes, histogramas, diagramas de caja, matriz de correlación y una revisión exploratoria de valores extremos.

---

## `notebooks/clasificacion_coberturas/`

Contiene los notebooks del flujo de clasificación de coberturas.

    notebooks/clasificacion_coberturas/
    ├── 01_descarga_gee.ipynb
    ├── 02_recorte_humedales.ipynb
    ├── 03_preparacion_raster.ipynb
    ├── 04_preparacion_muestras.ipynb
    ├── 05_entrenamiento_modelos.ipynb
    ├── 06_evaluacion_modelos.ipynb
    └── 07_clasificacion_espacial.ipynb

---

# 6. Carpeta `src/`

La carpeta `src/` contiene código reutilizable del repositorio.

    src/
    ├── calidad_agua/
    └── clasificacion_coberturas/

La finalidad de esta carpeta es evitar que los notebooks concentren toda la lógica técnica. Los notebooks deben importar funciones desde `src/` y usarse principalmente para configurar, ejecutar y documentar cada etapa.

---

## `src/calidad_agua/`

Contiene funciones del flujo de calidad de agua.

    src/calidad_agua/
    ├── extraccion_reflectancia.py
    ├── preparacion.py
    ├── entrenamiento.py
    ├── evaluacion.py
    ├── interpretacion.py
    ├── inferencia.py
    └── estadisticas_descriptivas.py

---

### `extraccion_reflectancia.py`

Funciones para extraer reflectancias promedio e índices espectrales desde ortomosaicos multibanda.

---

### `preparacion.py`

Funciones para cargar datasets, crear variables logarítmicas, preparar variables predictoras y objetivo, generar splits y construir matrices de correlación.

---

### `entrenamiento.py`

Funciones para entrenar modelos de regresión con búsqueda de hiperparámetros.

---

### `evaluacion.py`

Funciones para calcular métricas, comparar modelos y generar figuras de evaluación.

---

### `interpretacion.py`

Funciones para analizar importancia de variables en modelos compatibles.

---

### `inferencia.py`

Funciones para aplicar modelos entrenados sobre rasters y exportar mapas de predicción.

---

### `estadisticas_descriptivas.py`

Funciones para análisis descriptivo de parámetros de calidad de agua.

---

## `src/clasificacion_coberturas/`

Contiene funciones del flujo de clasificación de coberturas.

    src/clasificacion_coberturas/
    ├── gee.py
    ├── procesamiento_raster.py
    ├── visualizacion.py
    ├── preparacion_raster.py
    ├── muestras.py
    ├── entrenamiento.py
    ├── evaluacion.py
    └── inferencia.py

---

### `gee.py`

Funciones para trabajar con Google Earth Engine:

- inicializar GEE;
- cargar ROI;
- crear stack Sentinel-2;
- calcular índices;
- calcular texturas;
- visualizar mapas interactivos;
- exportar a Drive, Asset o Cloud Storage.

---

### `procesamiento_raster.py`

Funciones para:

- recortar rasters por humedal;
- generar máscaras binarias;
- guardar resumen de productos generados.

---

### `visualizacion.py`

Funciones para:

- visualizar composiciones RGB;
- crear mosaicos de rasters;
- visualizar bandas;
- graficar matrices de correlación.

---

### `preparacion_raster.py`

Funciones para:

- definir nombres de variables;
- cargar raster multibanda;
- filtrar píxeles válidos;
- calcular correlaciones;
- seleccionar variables;
- generar raster reducido.

---

### `muestras.py`

Funciones para:

- cargar puntos de entrenamiento;
- revisar distribución de clases;
- extraer valores raster en puntos;
- crear muestras;
- generar split estratificado.

---

### `entrenamiento.py`

Funciones para:

- definir modelos base;
- entrenar modelos;
- comparar métricas;
- generar radar plot;
- ejecutar búsqueda de hiperparámetros;
- guardar modelos.

---

### `evaluacion.py`

Funciones para:

- evaluar el modelo optimizado;
- generar matriz de confusión;
- crear reporte de clasificación;
- calcular importancia por permutación.

---

### `inferencia.py`

Funciones para:

- aplicar el modelo al raster reducido;
- reconstruir mapa clasificado;
- aplicar filtro de mayoría;
- exportar GeoTIFF;
- visualizar resultados.

---

# 7. Carpeta `scripts/`

La carpeta `scripts/` contiene utilidades ejecutables complementarias que no forman parte directa de los pipelines analíticos, pero apoyan tareas de publicación, migración o entrega de productos.

Estructura actual:

    scripts/
    └── publicacion/
        └── migracion_storymap_agol.py

---

## `scripts/publicacion/`

Contiene scripts relacionados con publicación o transferencia de productos.

### `migracion_storymap_agol.py`

Script para migrar o clonar un StoryMap entre cuentas o portales de ArcGIS Online / ArcGIS Enterprise Portal.

Permite:

- conectarse a un portal origen;
- conectarse a un portal destino;
- obtener un StoryMap por ID;
- clonar el StoryMap y sus dependencias;
- aplicar una reparación opcional sobre nodos `tour-map`;
- generar la URL del StoryMap clonado para revisión y publicación manual.

Este script requiere el entorno:

    humedales_publicacion_arcgis

definido en:

    config/publicacion_arcgis_config.yml

No se deben escribir credenciales directamente en el código. El script solicita credenciales por consola o permite usar variables de entorno.

---

# 8. Carpeta `models/`

La carpeta `models/` almacena modelos entrenados y objetos de búsqueda.

    models/
    ├── calidad_agua/
    └── clasificacion_coberturas/

---

## `models/calidad_agua/`

Contiene modelos de regresión entrenados.

Ejemplo:

    models/calidad_agua/entrenados/DQO/
    ├── SVR.joblib
    ├── GBR.joblib
    └── RFR.joblib

También puede contener resultados de búsqueda de hiperparámetros:

    models/calidad_agua/metricas/DQO/gridsearch_modelos.joblib

---

## `models/clasificacion_coberturas/`

Contiene modelos de clasificación entrenados.

Estructura esperada:

    models/clasificacion_coberturas/
    ├── base/
    ├── modelo_clasificacion_tuning.joblib
    └── busqueda_modelo_tuning.joblib

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

---

# 9. Carpeta `outputs/`

La carpeta `outputs/` almacena productos finales y reportes generados por los flujos.

    outputs/
    ├── figures/
    ├── tables/
    └── rasters/

---

## `outputs/figures/`

Contiene figuras generadas por los notebooks.

Estructura esperada:

    outputs/figures/
    ├── calidad_agua/
    └── clasificacion_coberturas/

Ejemplos de clasificación:

    outputs/figures/clasificacion_coberturas/
    ├── mosaico_rasters_recortados_2018.png
    ├── matriz_correlacion_pearson_<raster>.png
    ├── distribucion_clases_2018.png
    ├── radar_modelos_base.png
    ├── matriz_confusion.png
    ├── importancia_permutacion.png
    ├── clasificacion_coberturas_2018_clasificado.png
    └── clasificacion_coberturas_2018_clasificado_filtrado.png

---

## `outputs/tables/`

Contiene tablas generadas durante el flujo.

Estructura esperada:

    outputs/tables/
    ├── calidad_agua/
    └── clasificacion_coberturas/

Ejemplos de clasificación:

    outputs/tables/clasificacion_coberturas/
    ├── bandas_stack_S2_2018_STACK_3.csv
    ├── resumen_recortes_humedales_2018_<timestamp>.csv
    ├── pares_correlacion_pearson.csv
    ├── pares_correlacion_spearman.csv
    ├── variables_seleccionadas_pearson.csv
    ├── metricas_modelos_base.csv
    ├── comparacion_base_tuning.csv
    ├── metricas_modelo_final.csv
    ├── classification_report.csv
    ├── matriz_confusion.csv
    └── importancia_permutacion.csv

---

## `outputs/rasters/`

Contiene productos raster finales.

Estructura esperada:

    outputs/rasters/
    ├── calidad_agua/
    └── clasificacion_coberturas/

Ejemplos de clasificación:

    outputs/rasters/clasificacion_coberturas/
    ├── clasificacion_coberturas_2018_clasificado.tif
    └── clasificacion_coberturas_2018_clasificado_filtrado.tif

---

# 10. Relación entre notebooks y módulos

La arquitectura del repositorio separa ejecución y lógica:

- los notebooks configuran rutas, parámetros y ejecutan etapas;
- los módulos `.py` contienen funciones reutilizables;
- los productos intermedios conectan una etapa con la siguiente.

Ejemplo del flujo de clasificación:

    01_descarga_gee.ipynb
        ↓
    S2_2018_STACK_3.tif
        ↓
    02_recorte_humedales.ipynb
        ↓
    RPR_2018/<humedal>_clip.tif
        ↓
    03_preparacion_raster.ipynb
        ↓
    reducidos/S2_REDUCIDO_PEARSON_2018.tif
        ↓
    04_preparacion_muestras.ipynb
        ↓
    splits/split_clasificacion_2018.joblib
        ↓
    05_entrenamiento_modelos.ipynb
        ↓
    modelo_clasificacion_tuning.joblib
        ↓
    06_evaluacion_modelos.ipynb
        ↓
    métricas y reportes
        ↓
    07_clasificacion_espacial.ipynb
        ↓
    mapas clasificados

---

# 11. Archivos que no deberían versionarse

Por tamaño o porque pueden regenerarse, se recomienda no subir al repositorio:

- rasters `.tif` o `.tiff`;
- archivos `.vrt`;
- archivos auxiliares `.ovr` o `.aux.xml`;
- carpetas de caché;
- checkpoints de Jupyter;
- entornos locales;
- salidas raster pesadas;
- productos intermedios voluminosos.

El archivo `.gitignore` debe controlar estos casos.

En cambio, sí puede ser útil versionar:

- notebooks;
- scripts `.py`;
- documentación;
- archivos `.yml`;
- tablas pequeñas de configuración o resultados;
- figuras livianas, si la entrega lo requiere.

---

# 12. Convenciones recomendadas

## Nombres de carpetas

Se recomienda usar:

- minúsculas;
- guiones bajos;
- sin tildes;
- sin espacios.

Ejemplo:

    clasificacion_coberturas
    calidad_agua
    reducidos

---

## Nombres de archivos

Se recomienda evitar:

- tildes;
- espacios;
- caracteres especiales;
- nombres demasiado genéricos.

Ejemplo recomendado:

    ROI_BOGOTA.shp
    ROI_HUMEDALES.shp
    PUNTOS_N1.shp
    S2_2018_STACK_3.tif

---

## Rutas

Los notebooks deben resolver rutas de forma relativa a la raíz del repositorio mediante `BASE_DIR`.

Esto permite que el proyecto sea más portable entre equipos.

---

# 13. Estado actual del repositorio

El repositorio cuenta con:

- flujo de calidad de agua organizado;
- flujo de clasificación de coberturas organizado;
- utilidad de publicación para migración de StoryMaps en ArcGIS;
- notebooks separados por etapa;
- funciones centralizadas en `src/`;
- entornos independientes en `config/`;
- documentación técnica en `docs/`;
- estructura de salidas para tablas, figuras y rasters;
- `.gitignore` para evitar subir archivos pesados o temporales.

Esta arquitectura permite reproducir, revisar y extender los flujos de modelado de forma ordenada.
