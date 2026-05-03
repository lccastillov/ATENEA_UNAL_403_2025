"""
Migración de StoryMap en ArcGIS Online / Portal.

Este script permite clonar un StoryMap desde una cuenta o portal origen hacia
una cuenta o portal destino, incluyendo sus dependencias. Además, aplica una
corrección opcional sobre nodos tipo Map Tour dentro del JSON interno del
StoryMap clonado.

Uso general:
    python scripts/publicacion/migracion_storymap_agol.py

Requisitos:
    - ArcGIS API for Python.
    - Credenciales con permisos sobre el contenido origen y destino.
    - ID del StoryMap a migrar.

Notas:
    - No escriba contraseñas directamente en el código.
    - Revise el StoryMap clonado antes de publicarlo.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from arcgis.gis import GIS


# ============================================================
# CONEXIÓN
# ============================================================

def conectar_gis(
    url: str,
    usuario: str | None = None,
    password: str | None = None,
    etiqueta: str = "GIS"
) -> GIS:
    """
    Crea una conexión a ArcGIS Online o ArcGIS Enterprise Portal.

    Parameters
    ----------
    url : str
        URL del portal. Para ArcGIS Online usar: https://www.arcgis.com

    usuario : str | None
        Usuario de ArcGIS. Si es None, se solicita por consola.

    password : str | None
        Contraseña. Si es None, se solicita de forma segura por consola.

    etiqueta : str
        Texto usado para identificar la conexión en mensajes de consola.

    Returns
    -------
    GIS
        Objeto de conexión autenticado.
    """
    print(f"
--- Conexión {etiqueta} ---")

    if usuario is None:
        usuario = input(f"Usuario {etiqueta}: ").strip()

    if password is None:
        password = getpass.getpass(f"Contraseña {etiqueta}: ")

    gis = GIS(url, usuario, password)

    print(f"Conexión {etiqueta} establecida como: {gis.properties.user.username}")

    return gis


def conectar_desde_variables_entorno(
    source_url: str,
    target_url: str
) -> tuple[GIS, GIS]:
    """
    Crea conexiones usando variables de entorno.

    Variables esperadas:
        AGOL_SOURCE_USER
        AGOL_SOURCE_PASSWORD
        AGOL_TARGET_USER
        AGOL_TARGET_PASSWORD

    Esta opción es útil para automatizar el script sin escribir credenciales
    dentro del archivo.
    """
    source_user = os.getenv("AGOL_SOURCE_USER")
    source_password = os.getenv("AGOL_SOURCE_PASSWORD")
    target_user = os.getenv("AGOL_TARGET_USER")
    target_password = os.getenv("AGOL_TARGET_PASSWORD")

    faltantes = [
        nombre
        for nombre, valor in {
            "AGOL_SOURCE_USER": source_user,
            "AGOL_SOURCE_PASSWORD": source_password,
            "AGOL_TARGET_USER": target_user,
            "AGOL_TARGET_PASSWORD": target_password,
        }.items()
        if not valor
    ]

    if faltantes:
        raise ValueError(
            "Faltan variables de entorno para conexión automática: "
            + ", ".join(faltantes)
        )

    source = conectar_gis(
        url=source_url,
        usuario=source_user,
        password=source_password,
        etiqueta="origen"
    )

    target = conectar_gis(
        url=target_url,
        usuario=target_user,
        password=target_password,
        etiqueta="destino"
    )

    return source, target


# ============================================================
# STORYMAP Y DEPENDENCIAS
# ============================================================

def obtener_storymap(
    gis: GIS,
    story_id: str
):
    """
    Obtiene un StoryMap desde ArcGIS Online o Portal usando su item ID.
    """
    story = gis.content.get(story_id)

    if story is None:
        raise ValueError(f"No se encontró ningún item con ID: {story_id}")

    if "StoryMap" not in story.type:
        print(
            "Advertencia: el item encontrado no tiene tipo 'StoryMap'. "
            f"Tipo detectado: {story.type}"
        )

    print("
StoryMap origen encontrado:")
    print(f"  Título: {story.title}")
    print(f"  Tipo: {story.type}")
    print(f"  ID: {story.id}")

    return story


def obtener_items_a_clonar(story, deep: bool = True) -> list:
    """
    Obtiene el StoryMap y sus dependencias para clonación.
    """
    print("
Buscando dependencias...")

    dependencias = story.get_dependencies(deep=deep)

    items_to_clone = [story] + dependencias

    print(f"Items a clonar: {len(items_to_clone)}")

    return items_to_clone


def clonar_items(
    target: GIS,
    items_to_clone: list,
    copy_data: bool = True,
    search_existing_items: bool = True
) -> list:
    """
    Clona el StoryMap y sus dependencias en el portal destino.
    """
    print("
Iniciando clonación de contenido...")

    cloned_items = target.content.clone_items(
        items_to_clone,
        copy_data=copy_data,
        search_existing_items=search_existing_items
    )

    print(f"Items clonados o reutilizados: {len(cloned_items)}")

    return cloned_items


def identificar_storymap_clonado(cloned_items: list):
    """
    Identifica el StoryMap clonado dentro de la lista de items clonados.
    """
    story_clone = next(
        (
            item
            for item in cloned_items
            if "StoryMap" in item.type
        ),
        None
    )

    if story_clone is None:
        raise RuntimeError("No se encontró un StoryMap dentro de los items clonados.")

    print("
StoryMap clonado identificado:")
    print(f"  Título: {story_clone.title}")
    print(f"  Tipo: {story_clone.type}")
    print(f"  ID: {story_clone.id}")

    return story_clone


# ============================================================
# RECURSOS JSON DEL STORYMAP
# ============================================================

def obtener_json_storymap(story_item) -> tuple[dict[str, Any], str]:
    """
    Obtiene el JSON interno del StoryMap clonado.

    Se intenta leer primero `published_data.json`. Si no existe, se intenta
    leer `draft.json`.

    Returns
    -------
    data : dict
        Contenido JSON del StoryMap.

    recurso_usado : str
        Nombre del recurso usado como fuente.
    """
    recursos_posibles = [
        "published_data.json",
        "draft.json"
    ]

    for recurso in recursos_posibles:
        data = story_item.resources.get(recurso)

        if data:
            if isinstance(data, str):
                data = json.loads(data)

            print(f"
Recurso JSON usado: {recurso}")
            return data, recurso

    raise RuntimeError(
        "No se encontró `published_data.json` ni `draft.json` en el StoryMap clonado."
    )


def reparar_tours_map(data: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """
    Aplica una corrección específica a nodos tipo Map Tour.

    La corrección elimina la propiedad `basemap` dentro de nodos `tour-map`
    cuando está presente, y asegura estructuras mínimas `data` y `config`.

    Esta operación puede ayudar cuando algunos Map Tours presentan problemas
    después de clonar un StoryMap entre cuentas o portales.
    """
    fixed_tours = 0

    nodes = data.get("nodes", {})

    for _, node in nodes.items():
        if node.get("type") == "tour-map":
            node.setdefault("data", {})
            node.setdefault("config", {})

            if "basemap" in node["data"]:
                del node["data"]["basemap"]
                fixed_tours += 1

    return data, fixed_tours


def actualizar_draft_storymap(
    story_item,
    data: dict[str, Any],
    nombre_recurso: str = "draft.json"
) -> None:
    """
    Actualiza el recurso `draft.json` del StoryMap clonado.

    El usuario debe abrir el StoryMap clonado, revisar los cambios y publicar
    manualmente desde el editor de ArcGIS StoryMaps.
    """
    print(f"
Actualizando recurso: {nombre_recurso}")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / nombre_recurso

        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

        try:
            story_item.resources.remove(nombre_recurso)
            print(f"Recurso previo eliminado: {nombre_recurso}")
        except Exception:
            print(f"No se encontró recurso previo para eliminar: {nombre_recurso}")

        story_item.resources.add(
            file=str(temp_path),
            file_name=nombre_recurso
        )

    print("Recurso actualizado correctamente.")


# ============================================================
# PROCESO PRINCIPAL
# ============================================================

def migrar_storymap(
    story_id: str,
    source: GIS,
    target: GIS,
    reparar_tours: bool = True,
    copy_data: bool = True,
    search_existing_items: bool = True
):
    """
    Ejecuta la migración completa de un StoryMap.

    Parameters
    ----------
    story_id : str
        ID del StoryMap origen.

    source : GIS
        Conexión GIS origen.

    target : GIS
        Conexión GIS destino.

    reparar_tours : bool
        Si True, aplica corrección sobre nodos `tour-map`.

    copy_data : bool
        Si True, copia los datos asociados.

    search_existing_items : bool
        Si True, reutiliza items existentes cuando ArcGIS los detecta.

    Returns
    -------
    story_clone
        Item del StoryMap clonado.
    """
    story = obtener_storymap(
        gis=source,
        story_id=story_id
    )

    items_to_clone = obtener_items_a_clonar(
        story=story,
        deep=True
    )

    cloned_items = clonar_items(
        target=target,
        items_to_clone=items_to_clone,
        copy_data=copy_data,
        search_existing_items=search_existing_items
    )

    story_clone = identificar_storymap_clonado(
        cloned_items=cloned_items
    )

    if reparar_tours:
        print("
Revisando JSON interno del StoryMap clonado...")

        data, recurso_usado = obtener_json_storymap(story_clone)

        data, fixed_tours = reparar_tours_map(data)

        print(f"Map Tours ajustados: {fixed_tours}")

        actualizar_draft_storymap(
            story_item=story_clone,
            data=data,
            nombre_recurso="draft.json"
        )

    print("
" + "-" * 60)
    print("PROCESO TERMINADO")
    print(f"StoryMap clonado: {story_clone.title}")
    print(f"Nuevo ID: {story_clone.id}")
    print(f"URL: {target.url}/apps/storymaps/stories/{story_clone.id}")
    print("Revise el StoryMap clonado y publíquelo manualmente desde el editor.")
    print("-" * 60)

    return story_clone


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    """
    Define argumentos de línea de comandos.
    """
    parser = argparse.ArgumentParser(
        description="Migrar un StoryMap entre cuentas o portales de ArcGIS."
    )

    parser.add_argument(
        "--story-id",
        required=True,
        help="ID del StoryMap origen."
    )

    parser.add_argument(
        "--source-url",
        default="https://www.arcgis.com",
        help="URL del portal origen."
    )

    parser.add_argument(
        "--target-url",
        default="https://www.arcgis.com",
        help="URL del portal destino."
    )

    parser.add_argument(
        "--use-env",
        action="store_true",
        help="Usar variables de entorno para credenciales."
    )

    parser.add_argument(
        "--no-reparar-tours",
        action="store_true",
        help="No aplicar reparación de nodos Map Tour."
    )

    parser.add_argument(
        "--no-copy-data",
        action="store_true",
        help="No copiar datos asociados durante la clonación."
    )

    parser.add_argument(
        "--no-search-existing",
        action="store_true",
        help="No buscar/reutilizar items existentes durante la clonación."
    )

    return parser.parse_args()


def main() -> None:
    """
    Punto de entrada del script.
    """
    args = parse_args()

    if args.use_env:
        source, target = conectar_desde_variables_entorno(
            source_url=args.source_url,
            target_url=args.target_url
        )
    else:
        source = conectar_gis(
            url=args.source_url,
            etiqueta="origen"
        )

        target = conectar_gis(
            url=args.target_url,
            etiqueta="destino"
        )

    migrar_storymap(
        story_id=args.story_id,
        source=source,
        target=target,
        reparar_tours=not args.no_reparar_tours,
        copy_data=not args.no_copy_data,
        search_existing_items=not args.no_search_existing
    )


if __name__ == "__main__":
    main()
