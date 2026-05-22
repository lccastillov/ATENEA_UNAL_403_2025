# humedales_bogota_ml

Repositorio para organizar, documentar y ejecutar flujos de modelado aplicados a humedales de Bogotá.

El proyecto integra procesamiento geoespacial, percepción remota, machine learning y análisis espacial mediante dos líneas principales:

1. **Modelado de calidad de agua**: estimación de parámetros fisicoquímicos a partir de variables espectrales derivadas de imágenes multibanda.
2. **Clasificación de coberturas**: generación de mapas temáticos de coberturas a partir de imágenes Sentinel-2, índices espectrales, texturas GLCM y puntos de entrenamiento.
3. **Publicación en ArcGIS Online / Portal**: utilidad complementaria para migrar o clonar StoryMaps entre cuentas o portales.

El repositorio está diseñado para que los procesos sean reproducibles, auditables y fáciles de extender a nuevos humedales, nuevas fechas, nuevos parámetros o nuevos insumos espaciales.

---

## Objetivos del repositorio

El repositorio tiene como objetivos principales:

- centralizar los flujos de modelado en una estructura reproducible;
- separar notebooks, funciones reutilizables, datos, modelos y salidas;
- documentar las entradas, procesos y productos de cada etapa;
- facilitar la instalación mediante entornos `conda`;
- permitir la ejecución ordenada de notebooks;
- conservar visualizaciones relevantes dentro de los notebooks;
- guardar tablas, figuras, modelos y rasters en carpetas de salida organizadas;
- permitir futuras extensiones del proyecto sin depender de notebooks monolíticos.

---

## Flujos disponibles

### 1. Calidad de agua

El flujo de calidad de agua permite estimar parámetros fisicoquímicos a partir de reflectancias e índices espectrales extraídos desde imágenes multibanda.

Incluye:

- extracción de reflectancias e índices espectrales;
- preparación de dataset;
- transformación logarítmica opcional de variables respuesta;
- entrenamiento de modelos de regresión;
- evaluación de métricas;
- análisis de importancia de variables;
- predicción espacial sobre rasters multibanda.

Modelos incluidos:

- SVR;
- Gradient Boosting Regressor;
- Random Forest Regressor.

---

### 2. Clasificación de coberturas

El flujo de clasificación de coberturas permite generar mapas temáticos de coberturas para humedales.

Incluye:

- generación de stacks Sentinel-2 desde Google Earth Engine;
- cálculo de índices espectrales;
- cálculo de texturas GLCM;
- exportación de stacks a Google Drive, Earth Engine Asset o Google Cloud Storage;
- recorte de rasters por humedal;
- generación de máscaras binarias;
- análisis de correlación Pearson y Spearman;
- reducción de variables;
- preparación de muestras de entrenamiento;
- entrenamiento y comparación de modelos de clasificación;
- ajuste de hiperparámetros;
- evaluación del modelo optimizado;
- clasificación espacial;
- filtro de mayoría;
- exportación de mapas clasificados.

Modelos base incluidos:

- Decision Tree Classifier;
- Random Forest Classifier;
- Support Vector Classifier con kernel RBF;
- Gradient Boosting Classifier;
- Extra Trees Classifier;
- AdaBoost Classifier;
- HistGradientBoosting Classifier;
- K-Nearest Neighbors Classifier.

---

## Estructura general del repositorio

```text
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
```

La descripción detallada de la arquitectura se encuentra en:

```text
docs/arquitectura_repositorio.md
```

---

## Instalación

El proyecto usa entornos independientes para cada flujo.

Los archivos de configuración se encuentran en:

```text
config/
├── calidad_agua_config.yml
└── clasificacion_config.yml
```

---

### Entorno de calidad de agua

Crear el entorno:

```bash
conda env create -f config/calidad_agua_config.yml
```

Activarlo:

```bash
conda activate humedales_calidad_agua
```

Registrar el kernel en JupyterLab:

```bash
python -m ipykernel install --user --name humedales_calidad_agua --display-name "Python (humedales_calidad_agua)"
```

---

### Entorno de clasificación de coberturas

Crear el entorno:

```bash
conda env create -f config/clasificacion_config.yml
```

Activarlo:

```bash
conda activate humedales_clasificacion
```

Registrar el kernel en JupyterLab:

```bash
python -m ipykernel install --user --name humedales_clasificacion --display-name "Python (humedales_clasificacion)"
```

---

### Entorno de publicación en ArcGIS

Este entorno es opcional y sólo se requiere para usar el script de migración de StoryMaps.

Crear el entorno:

```bash
conda env create -f config/publicacion_arcgis_config.yml
```

Activarlo:

```bash
conda activate humedales_publicacion_arcgis
```

Ejecutar el script desde la raíz del repositorio:

```bash
python scripts/publicacion/migracion_storymap_agol.py --story-id <ID_DEL_STORYMAP>
```

---

### Abrir JupyterLab

Se recomienda abrir JupyterLab desde el entorno que se va a usar.

Para calidad de agua:

```bash
conda activate humedales_calidad_agua
python -m jupyter lab
```

Para clasificación de coberturas:

```bash
conda activate humedales_clasificacion
python -m jupyter lab
```

Más detalles en:

```text
docs/guia_instalacion.md
```

---

## Uso general

Los notebooks deben ejecutarse en orden, ya que cada etapa depende de productos generados previamente.

La guía completa de uso se encuentra en:

```text
docs/guia_uso.md
```

---

# Flujo de calidad de agua

## Notebooks

```text
notebooks/calidad_agua/
├── 00_estadisticas_laboratorio.ipynb
├── 01_extraccion_variables.ipynb
├── 02_preparacion_dataset_modelos.ipynb
├── 03_entrenamiento_modelos.ipynb
├── 04_evaluacion_modelos.ipynb
├── 05_analisis_importancia_variables.ipynb
└── 06_prediccion_espacial_modelos.ipynb
```

---

## Módulos

```text
src/calidad_agua/
├── extraccion_reflectancia.py
├── preparacion.py
├── entrenamiento.py
├── evaluacion.py
├── interpretacion.py
├── inferencia.py
└── estadisticas_descriptivas.py
```

---

## Orden de ejecución

El notebook `00_estadisticas_laboratorio.ipynb` es complementario. Puede ejecutarse antes del flujo principal para revisar los parámetros de laboratorio, pero no es obligatorio para entrenar los modelos.

1. `01_extraccion_variables.ipynb`
2. `02_preparacion_dataset_modelos.ipynb`
3. `03_entrenamiento_modelos.ipynb`
4. `04_evaluacion_modelos.ipynb`
5. `05_analisis_importancia_variables.ipynb`
6. `06_prediccion_espacial_modelos.ipynb`

---

## Productos principales

### Datos procesados

```text
data/processed/calidad_agua/
└── Puntos_Muestreo_Reflectancia_Indices_Excel_2.gpkg
```

### Splits

```text
data/interim/calidad_agua/splits/
└── split_DQO.joblib
```

### Modelos

```text
models/calidad_agua/entrenados/DQO/
├── SVR.joblib
├── GBR.joblib
└── RFR.joblib
```

### Tablas y figuras

```text
outputs/tables/calidad_agua/DQO/
outputs/figures/calidad_agua/DQO/
```

### Rasters

```text
outputs/rasters/calidad_agua/DQO/
├── DQO_SVR.tif
├── DQO_GBR.tif
└── DQO_RFR.tif
```

---

# Flujo de clasificación de coberturas

## Notebooks

```text
notebooks/clasificacion_coberturas/
├── 01_descarga_gee.ipynb
├── 02_recorte_humedales.ipynb
├── 03_preparacion_raster.ipynb
├── 04_preparacion_muestras.ipynb
├── 05_entrenamiento_modelos.ipynb
├── 06_evaluacion_modelos.ipynb
└── 07_clasificacion_espacial.ipynb
```

---

## Módulos

```text
src/clasificacion_coberturas/
├── gee.py
├── procesamiento_raster.py
├── visualizacion.py
├── preparacion_raster.py
├── muestras.py
├── entrenamiento.py
├── evaluacion.py
└── inferencia.py
```

---

## Orden de ejecución

1. `01_descarga_gee.ipynb`
2. `02_recorte_humedales.ipynb`
3. `03_preparacion_raster.ipynb`
4. `04_preparacion_muestras.ipynb`
5. `05_entrenamiento_modelos.ipynb`
6. `06_evaluacion_modelos.ipynb`
7. `07_clasificacion_espacial.ipynb`

---

## Entradas crudas principales

```text
data/raw/clasificacion_coberturas/vectores/
├── PUNTOS_N1.shp
├── ROI_BOGOTA.shp
└── ROI_HUMEDALES.shp

data/raw/clasificacion_coberturas/rasters/
├── S2_2018_STACK_3.tif
└── S2_2024_STACK_3.tif
```

Para archivos shapefile, deben conservarse sus archivos asociados:

```text
.shp
.shx
.dbf
.prj
.cpg
```

Los archivos `S2_2018_STACK_3.tif` y `S2_2024_STACK_3.tif` corresponden a stacks generados desde Google Earth Engine.

---

## Etapas principales

### 1. Descarga desde GEE

Genera stacks Sentinel-2 con bandas espectrales, índices y texturas.

Salida esperada:

```text
data/raw/clasificacion_coberturas/rasters/S2_2018_STACK_3.tif
```

También se guarda la lista de bandas:

```text
outputs/tables/clasificacion_coberturas/bandas_stack_S2_2018_STACK_3.csv
```

---

### 2. Recorte por humedal

Genera rasters recortados y máscaras binarias.

```text
data/interim/clasificacion_coberturas/RPR_2018/
data/interim/clasificacion_coberturas/MBR_2018/
```

---

### 3. Preparación del raster

Calcula correlaciones, selecciona variables y genera el raster reducido.

```text
data/interim/clasificacion_coberturas/reducidos/
├── S2_REDUCIDO_PEARSON_2018.tif
└── bandas_pearson_2018.json
```

Parámetros relevantes:

```text
nodata_val = -9999
threshold_pares = 0.90
threshold_seleccion = 0.925
```

---

### 4. Preparación de muestras

Extrae valores del raster reducido en puntos de entrenamiento y genera un split estratificado.

```text
data/interim/clasificacion_coberturas/muestras/
└── muestras_entrenamiento_2018.csv

data/interim/clasificacion_coberturas/splits/
└── split_clasificacion_2018.joblib
```

---

### 5. Entrenamiento

Entrena modelos base, selecciona el mejor según `f1_macro` y optimiza hiperparámetros.

```text
models/clasificacion_coberturas/
├── base/
├── modelo_clasificacion_tuning.joblib
└── busqueda_modelo_tuning.joblib
```

---

### 6. Evaluación

Evalúa el modelo optimizado mediante métricas globales, matriz de confusión, reporte de clasificación e importancia por permutación.

```text
outputs/tables/clasificacion_coberturas/
├── metricas_modelo_final.csv
├── classification_report.csv
├── matriz_confusion.csv
└── importancia_permutacion.csv

outputs/figures/clasificacion_coberturas/
├── matriz_confusion.png
└── importancia_permutacion.png
```

---

### 7. Clasificación espacial

Aplica el modelo optimizado al raster reducido y exporta mapas clasificados.

```text
outputs/rasters/clasificacion_coberturas/
├── clasificacion_coberturas_2018_clasificado.tif
└── clasificacion_coberturas_2018_clasificado_filtrado.tif

outputs/figures/clasificacion_coberturas/
├── clasificacion_coberturas_2018_clasificado.png
└── clasificacion_coberturas_2018_clasificado_filtrado.png
```

---

## Configuración de Google Earth Engine

El flujo de clasificación de coberturas usa Google Earth Engine en:

```text
notebooks/clasificacion_coberturas/01_descarga_gee.ipynb
```

Para ejecutar esta etapa se requiere:

- cuenta de Google Earth Engine;
- proyecto de Google Cloud registrado para usar Earth Engine;
- permisos sobre el proyecto;
- permisos de exportación hacia Drive, Asset o Cloud Storage.

En el notebook se debe revisar:

```python
GEE_PROJECT = "nombre-del-proyecto"
```

También puede usarse:

```python
GEE_PROJECT = None
```

si se desea inicializar Earth Engine con la configuración por defecto del usuario autenticado.

---

## Visualización interactiva

El notebook `01_descarga_gee.ipynb` usa `geemap` e `ipyleaflet` para visualizar el stack sobre el ROI.

Si el mapa no se renderiza, verificar:

1. Que el kernel activo sea `Python (humedales_clasificacion)`.
2. Que el entorno incluya `ipyleaflet`, `jupyter_leaflet`, `ipywidgets` y `jupyterlab_widgets`.
3. Que JupyterLab haya sido abierto desde el entorno `humedales_clasificacion`.

Prueba mínima:

```python
from ipyleaflet import Map
Map(center=(4.65, -74.1), zoom=10)
```

---

# Utilidad de publicación en ArcGIS Online / Portal

Además de los dos flujos de modelado, el repositorio incluye un script complementario para migrar o clonar StoryMaps entre cuentas o portales de ArcGIS.

Este script no forma parte del pipeline analítico. Su propósito es apoyar la fase de publicación, transferencia o entrega de productos en ArcGIS Online o ArcGIS Enterprise Portal.

## Ubicación

```text
scripts/publicacion/migracion_storymap_agol.py
```

## Entorno recomendado

```text
config/publicacion_arcgis_config.yml
```

El entorno se mantiene separado porque la librería `arcgis` no es necesaria para los modelos de calidad de agua ni para clasificación de coberturas.

## Uso básico

```bash
conda activate humedales_publicacion_arcgis
python scripts/publicacion/migracion_storymap_agol.py --story-id <ID_DEL_STORYMAP>
```

El script solicita credenciales de origen y destino por consola usando entrada segura de contraseña.

## Uso con variables de entorno

También permite usar variables de entorno:

```text
AGOL_SOURCE_USER
AGOL_SOURCE_PASSWORD
AGOL_TARGET_USER
AGOL_TARGET_PASSWORD
```

Ejecución:

```bash
python scripts/publicacion/migracion_storymap_agol.py --story-id <ID_DEL_STORYMAP> --use-env
```

## Consideraciones

- No escribir usuarios ni contraseñas directamente en el código.
- La cuenta origen debe tener acceso al StoryMap.
- La cuenta destino debe tener permisos para crear contenido.
- El StoryMap clonado debe revisarse y publicarse manualmente desde el editor de ArcGIS StoryMaps.
- La reparación de nodos `tour-map` es opcional y está orientada a corregir ciertos problemas de Map Tours después de la clonación.

---

## Documentación adicional

La documentación técnica se encuentra en:

```text
docs/
├── arquitectura_repositorio.md
├── flujo_calidad_agua.md
├── flujo_clasificacion_coberturas.md
├── guia_instalacion.md
└── guia_uso.md
```

---

## Archivos pesados y control de versiones

El repositorio está pensado para versionar código, documentación, notebooks y configuraciones.

Se recomienda no versionar archivos pesados como:

- rasters `.tif` o `.tiff`;
- archivos `.vrt`;
- archivos auxiliares `.ovr` o `.aux.xml`;
- productos intermedios voluminosos;
- salidas raster grandes;
- cachés y archivos temporales.

Estos elementos deben controlarse mediante `.gitignore`.

Los notebooks `.ipynb` sí pueden versionarse con sus salidas ejecutadas si se desea conservar visualizaciones y resultados visibles dentro del repositorio.

---

## Consideraciones metodológicas

- Los mapas generados por los modelos son estimaciones, no mediciones directas.
- Las salidas raster deben interpretarse junto con las métricas de evaluación.
- En clasificación, la calidad del mapa depende de la calidad y distribución de los puntos de entrenamiento.
- En calidad de agua, la calidad de las predicciones depende de la relación entre los parámetros fisicoquímicos y las variables espectrales.
- El tratamiento de NoData es crítico para evitar correlaciones artificiales en rasters recortados.
- Los umbrales de selección de variables pueden ajustarse según el objetivo del análisis y el comportamiento de los datos.
- El filtro de mayoría suaviza el mapa clasificado, pero puede reducir detalles espaciales finos.

---

## Convenciones recomendadas

Se recomienda usar:

- nombres de carpetas en minúscula;
- guiones bajos en lugar de espacios;
- rutas relativas;
- archivos sin tildes ni caracteres especiales;
- notebooks numerados según el orden de ejecución;
- funciones reutilizables en `src/`.

Ejemplos:

```text
clasificacion_coberturas
calidad_agua
reducidos
ROI_BOGOTA.shp
S2_2018_STACK_3.tif
```

---

## Estado del proyecto

El repositorio cuenta con:

- flujo de calidad de agua organizado y documentado;
- flujo de clasificación de coberturas organizado y documentado;
- notebooks separados por etapa;
- funciones centralizadas en `src/`;
- entornos independientes en `config/`;
- documentación técnica en `docs/`;
- estructura de salidas para tablas, figuras, modelos y rasters;
- utilidad complementaria para publicación/migración de StoryMaps en ArcGIS.

---

## Licencia

Este repositorio se distribuye bajo la licencia **GNU General Public License v3.0 (GPL-3.0)**.

La licencia permite usar, estudiar, modificar y redistribuir el código fuente, siempre que las obras derivadas se mantengan bajo los mismos términos de la GPL-3.0. Esta condición garantiza que las modificaciones, adaptaciones o extensiones del repositorio conserven su carácter abierto y reproducible.

Los términos completos de la licencia se encuentran en el archivo `LICENSE`.

---

## Mantenimiento

Para mantener el repositorio en buen estado:

- actualizar los archivos `.yml` cuando se agreguen dependencias;
- documentar cambios metodológicos relevantes;
- evitar rutas absolutas personales;
- no subir archivos pesados innecesarios;
- revisar que los notebooks se ejecuten con el kernel correcto;
- conservar la separación entre código reutilizable y notebooks de ejecución.

---

## Información de contacto

Las consultas relacionadas con el alcance metodológico, la implementación técnica, la interpretación de resultados o el uso académico de este repositorio pueden dirigirse a:

**Liliana Carolina Villamor**  
PhD. Geography & Earth Sciences  
Docente, Universidad Nacional de Colombia  
Correo institucional: [lcastillov@unal.edu.co](mailto:lcastillov@unal.edu.co)

**Diego Joaquín Rugeles Martínez**  
Magíster en Geomática  
Ingeniero Ambiental  
Correo institucional: [drugeles@unal.edu.co](mailto:drugeles@unal.edu.co)

**Wilmer Alexander Martínez Martínez**  
Estudiante maestría en Ingeniería Ambiental  
Ingeniero Ambiental  
Correo institucional: [wiamartinezma@unal.edu.co](mailto:wiamartinezma@unal.edu.co)
