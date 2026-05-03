# Guía de instalación

Este documento describe el procedimiento recomendado para preparar los entornos de trabajo del repositorio `humedales_bogota_ml`.

El repositorio contiene dos flujos principales:

1. **Modelado de calidad de agua**.
2. **Clasificación de coberturas**.
3. **Publicación en ArcGIS Online / Portal**.

Cada flujo cuenta con su propio archivo de configuración `.yml`, debido a que las dependencias pueden diferir entre modelados.

---

## Requisitos previos

Antes de instalar los entornos, se recomienda contar con:

- Anaconda o Miniconda instalado;
- acceso a Anaconda Prompt, PowerShell o terminal equivalente;
- el repositorio descargado o clonado localmente;
- conexión a internet para descargar dependencias;
- cuenta de Google Earth Engine, en caso de ejecutar el flujo de clasificación de coberturas;
- proyecto de Google Cloud registrado para usar Earth Engine.

---

## Estructura de configuración

Los archivos de entorno se ubican en:

    config/

La estructura esperada es:

    config/
    ├── calidad_agua_config.yml
    ├── clasificacion_config.yml
    └── publicacion_arcgis_config.yml

El archivo `calidad_agua_config.yml` contiene las dependencias necesarias para ejecutar el flujo de calidad de agua.

El archivo `clasificacion_config.yml` contiene las dependencias necesarias para ejecutar el flujo de clasificación de coberturas, incluyendo librerías para Google Earth Engine y visualización interactiva con mapas.

El archivo `publicacion_arcgis_config.yml` contiene las dependencias necesarias para ejecutar el script complementario de migración de StoryMaps en ArcGIS Online o ArcGIS Enterprise Portal.

---

## Ubicación en la raíz del repositorio

Antes de ejecutar los comandos de instalación, se recomienda ubicarse en la raíz del repositorio.

Ejemplo genérico:

    cd ruta/al/repositorio/humedales_bogota_ml

En Windows, una ruta podría tener una forma similar a:

    cd C:\ruta\al\repositorio\humedales_bogota_ml

La ruta exacta depende de dónde se haya guardado el proyecto en el equipo local.

---

# Instalación del entorno de calidad de agua

## Crear el entorno

Desde la raíz del repositorio, ejecutar:

    conda env create -f config/calidad_agua_config.yml

Este comando crea el entorno definido en el archivo `.yml`.

---

## Activar el entorno

Después de crear el entorno, activarlo con:

    conda activate humedales_calidad_agua

El nombre exacto del entorno depende del campo `name:` definido dentro de `config/calidad_agua_config.yml`.

---

## Registrar el kernel de calidad de agua

Para que el entorno aparezca como opción dentro de JupyterLab, registrar el kernel:

    python -m ipykernel install --user --name humedales_calidad_agua --display-name "Python (humedales_calidad_agua)"

Luego, al abrir los notebooks de calidad de agua, seleccionar el kernel:

    Python (humedales_calidad_agua)

---

# Instalación del entorno de clasificación de coberturas

## Crear el entorno

Desde la raíz del repositorio, ejecutar:

    conda env create -f config/clasificacion_config.yml

Este comando crea el entorno:

    humedales_clasificacion

---

## Activar el entorno

Después de crear el entorno, activarlo con:

    conda activate humedales_clasificacion

---

## Registrar el kernel de clasificación

Para que el entorno aparezca en JupyterLab, registrar el kernel:

    python -m ipykernel install --user --name humedales_clasificacion --display-name "Python (humedales_clasificacion)"

Luego, al abrir los notebooks de clasificación de coberturas, seleccionar el kernel:

    Python (humedales_clasificacion)

Este paso es importante. Si el notebook se ejecuta con un kernel diferente, algunas librerías pueden no encontrarse o las visualizaciones interactivas pueden fallar.

---

# Instalación del entorno de publicación en ArcGIS

Este entorno es opcional y sólo se requiere para ejecutar el script de migración o clonación de StoryMaps.

## Crear el entorno

Desde la raíz del repositorio, ejecutar:

    conda env create -f config/publicacion_arcgis_config.yml

Este comando crea el entorno:

    humedales_publicacion_arcgis

---

## Activar el entorno

Después de crear el entorno, activarlo con:

    conda activate humedales_publicacion_arcgis

---

## Verificar instalación

Con el entorno activo, ejecutar:

    python -c "from arcgis.gis import GIS; print('Entorno de publicación ArcGIS funcionando correctamente')"

---

## Ejecutar el script de migración

Desde la raíz del repositorio:

    python scripts/publicacion/migracion_storymap_agol.py --story-id <ID_DEL_STORYMAP>

El script solicitará credenciales de origen y destino mediante consola.

También puede ejecutarse con variables de entorno:

    python scripts/publicacion/migracion_storymap_agol.py --story-id <ID_DEL_STORYMAP> --use-env

Variables requeridas para esta opción:

    AGOL_SOURCE_USER
    AGOL_SOURCE_PASSWORD
    AGOL_TARGET_USER
    AGOL_TARGET_PASSWORD

---

## Abrir JupyterLab desde el entorno correcto

Se recomienda abrir JupyterLab desde el entorno que se vaya a usar.

Para calidad de agua:

    conda activate humedales_calidad_agua
    python -m jupyter lab

Para clasificación de coberturas:

    conda activate humedales_clasificacion
    python -m jupyter lab

Usar `python -m jupyter lab` ayuda a asegurar que JupyterLab se ejecute con el Python del entorno activo.

---

## Verificación del kernel dentro del notebook

Dentro de cualquier notebook, se puede verificar el entorno activo con:

    import sys
    print(sys.executable)

Para el flujo de clasificación, la ruta debería apuntar a algo similar a:

    .../envs/humedales_clasificacion/python.exe

Para el flujo de calidad de agua, debería apuntar a algo similar a:

    .../envs/humedales_calidad_agua/python.exe

---

# Dependencias principales

## Calidad de agua

El entorno de calidad de agua incluye dependencias para:

### Ejecución en notebooks

- `jupyterlab`
- `notebook`
- `ipykernel`
- `ipywidgets`

### Ciencia de datos

- `numpy`
- `pandas`
- `scipy`
- `openpyxl`
- `joblib`

### Machine learning

- `scikit-learn`

### Visualización

- `matplotlib`
- `seaborn`
- `plotly`

### Datos geoespaciales

- `geopandas`
- `rasterio`
- `shapely`
- `fiona`
- `pyproj`

### Utilidades

- `tqdm`
- `requests`

---

## Clasificación de coberturas

El entorno de clasificación de coberturas incluye dependencias para:

### Ejecución en notebooks

- `jupyterlab`
- `notebook`
- `ipykernel`
- `ipywidgets`
- `jupyterlab_widgets`

### Ciencia de datos

- `numpy`
- `pandas`
- `scipy`
- `openpyxl`
- `joblib`

### Machine learning

- `scikit-learn`

### Visualización

- `matplotlib`
- `seaborn`
- `plotly`

### Datos geoespaciales

- `geopandas`
- `rasterio`
- `shapely`
- `fiona`
- `pyproj`

### Google Earth Engine y mapas interactivos

- `earthengine-api`
- `geemap`
- `folium`
- `ipyleaflet`
- `jupyter_leaflet`

### Utilidades

- `tqdm`
- `requests`
- `unidecode`

---

## Publicación en ArcGIS

El entorno de publicación incluye dependencias para migrar o clonar StoryMaps:

### ArcGIS Online / ArcGIS Enterprise Portal

- `arcgis`

### Utilidades

- `requests`

Este entorno no requiere registro como kernel de JupyterLab, porque está orientado a ejecutar scripts desde consola.

---

# Configuración de Google Earth Engine

El flujo de clasificación de coberturas usa Google Earth Engine en el notebook:

    notebooks/clasificacion_coberturas/01_descarga_gee.ipynb

Para ejecutar esta etapa, el usuario debe contar con:

- cuenta habilitada para Google Earth Engine;
- proyecto de Google Cloud registrado para usar Earth Engine;
- permisos para usar el proyecto;
- permisos de exportación hacia Google Drive, Earth Engine Asset o Google Cloud Storage, según el modo seleccionado.

---

## Registro de proyecto en Earth Engine

Si aparece un error indicando que el proyecto no está registrado para usar Earth Engine, debe registrarse desde la configuración de Earth Engine en Google Cloud.

Después de registrar el proyecto, el notebook puede inicializarse usando:

    GEE_PROJECT = "nombre-del-proyecto"

o, en algunos casos, usando:

    GEE_PROJECT = None

La opción `None` intenta inicializar Earth Engine con la configuración por defecto del usuario autenticado.

---

## Permisos del proyecto

Si aparece un error indicando que el usuario no tiene permisos para usar el proyecto, significa que la cuenta autenticada no tiene permisos suficientes sobre ese proyecto.

En ese caso, se debe:

- usar un proyecto propio con Earth Engine habilitado;
- o solicitar permisos al administrador del proyecto.

Un permiso común requerido es equivalente a `Service Usage Consumer`.

---

# Visualización interactiva con geemap

El flujo de clasificación usa `geemap` e `ipyleaflet` para mostrar mapas interactivos.

Si la visualización falla con un mensaje como:

    Failed to load model class 'LeafletMapModel' from module 'jupyter-leaflet'

se recomienda revisar:

1. Que el notebook use el kernel `Python (humedales_clasificacion)`.
2. Que `ipyleaflet` y `jupyter_leaflet` estén instalados en el entorno correcto.
3. Que JupyterLab se haya abierto desde el entorno `humedales_clasificacion`.

Prueba mínima dentro del notebook:

    from ipyleaflet import Map
    Map(center=(4.65, -74.1), zoom=10)

Si esta prueba funciona, `geemap.Map()` debería renderizar correctamente.

---

# Verificación rápida de instalación

## Calidad de agua

Con el entorno activo:

    conda activate humedales_calidad_agua

Ejecutar:

    python -c "import geopandas, rasterio, sklearn, pandas, numpy; print('Entorno de calidad de agua funcionando correctamente')"

---

## Clasificación de coberturas

Con el entorno activo:

    conda activate humedales_clasificacion

Ejecutar:

    python -c "import ee, geemap, geopandas, rasterio, sklearn, ipyleaflet; print('Entorno de clasificación funcionando correctamente')"

---

## Publicación en ArcGIS

Con el entorno activo:

    conda activate humedales_publicacion_arcgis

Ejecutar:

    python -c "from arcgis.gis import GIS; print('Entorno de publicación ArcGIS funcionando correctamente')"

---

# Actualización de entornos

Si un archivo `.yml` cambia después de crear el entorno, se puede actualizar con:

    conda env update -f config/calidad_agua_config.yml --prune

o para clasificación:

    conda env update -f config/clasificacion_config.yml --prune

o para publicación en ArcGIS:

    conda env update -f config/publicacion_arcgis_config.yml --prune

El argumento `--prune` elimina dependencias que ya no estén definidas en el archivo.

---

# Eliminación de entornos

Si es necesario eliminar un entorno, usar:

    conda env remove -n humedales_calidad_agua

o:

    conda env remove -n humedales_clasificacion

o:

    conda env remove -n humedales_publicacion_arcgis

Luego puede crearse nuevamente desde el archivo `.yml`.

---

# Recomendaciones generales

- Ejecutar los notebooks siempre con el kernel correspondiente.
- Abrir JupyterLab desde el entorno que se va a usar.
- Evitar instalar paquetes manualmente desde notebooks.
- Si se añade una nueva dependencia, actualizar el archivo `.yml`.
- Mantener separados los entornos de calidad de agua y clasificación de coberturas.
- Revisar que las rutas relativas se resuelvan correctamente desde la raíz del repositorio.
- Evitar tildes, espacios y caracteres especiales en nombres de archivos usados como insumos del flujo.

---

## Estado actual

El repositorio cuenta con dos entornos independientes:

    config/calidad_agua_config.yml
    config/clasificacion_config.yml
    config/publicacion_arcgis_config.yml

Los entornos de calidad de agua y clasificación deben registrarse como kernels si se desea ejecutar notebooks. El entorno de publicación en ArcGIS está orientado a ejecutar scripts desde consola.
