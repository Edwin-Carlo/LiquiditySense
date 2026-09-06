"""
tracking.py
-----------

Módulo de seguimiento y trazabilidad de experimentos para LiquiditySense.

Centraliza la configuración y registro de experimentos utilizando MLflow.

Funciones principales:
- Configuración del tracking URI.
- Creación/selección del experimento.
- Registro de parámetros.
- Registro de métricas.
- Registro de tags.
- Registro de artefactos.
- Registro del modelo.
- Consulta de información de runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import mlflow


# ============================================================
# CONFIGURACIÓN
# ============================================================

NOMBRE_EXPERIMENTO = "LiquiditySense"


# ============================================================
# CONFIGURAR MLFLOW
# ============================================================

def configurar_mlflow(
    ruta_tracking: str | Path,
    nombre_experimento: str = NOMBRE_EXPERIMENTO,
) -> str:
    """
    Configura el tracking URI de MLflow y selecciona el experimento.

    Parameters
    ----------
    ruta_tracking : str | Path
        Ruta de la base SQLite utilizada por MLflow.

    nombre_experimento : str, default="LiquiditySense"
        Nombre del experimento.

    Returns
    -------
    str
        Experiment ID de MLflow.
    """

    ruta_tracking = Path(ruta_tracking)

    if ruta_tracking.suffix.lower() != ".db":
        raise ValueError(
            "La ruta de tracking debe corresponder a una base SQLite "
            "con extensión '.db'."
        )

    ruta_tracking.parent.mkdir(parents=True, exist_ok=True)

    tracking_uri = "sqlite:///" + str(ruta_tracking.resolve()).replace("\\", "/")

    mlflow.set_tracking_uri(tracking_uri)

    experimento = mlflow.get_experiment_by_name(nombre_experimento)

    if experimento is None:
        experiment_id = mlflow.create_experiment(nombre_experimento)
    else:
        experiment_id = experimento.experiment_id

    mlflow.set_experiment(nombre_experimento)

    return experiment_id


# ============================================================
# INICIAR RUN
# ============================================================

def iniciar_run(
    nombre_run: str | None = None,
):
    """
    Inicia un nuevo run de MLflow.

    Parameters
    ----------
    nombre_run : str | None
        Nombre opcional del run.

    Returns
    -------
    ActiveRun
        Run activo de MLflow.
    """

    return mlflow.start_run(run_name=nombre_run)


# ============================================================
# REGISTRAR PARÁMETROS
# ============================================================

def registrar_parametros(
    parametros: Mapping[str, Any],
) -> None:
    """
    Registra parámetros de un modelo o experimento.

    Parameters
    ----------
    parametros : Mapping[str, Any]
        Diccionario con parámetros a registrar.
    """

    if not parametros:
        return

    parametros_limpios = {
        str(clave): str(valor)
        for clave, valor in parametros.items()
    }

    mlflow.log_params(parametros_limpios)


# ============================================================
# REGISTRAR MÉTRICAS
# ============================================================

def registrar_metricas(
    metricas: Mapping[str, float],
) -> None:
    """
    Registra métricas numéricas en MLflow.

    Parameters
    ----------
    metricas : Mapping[str, float]
        Diccionario con las métricas del modelo.
    """

    if not metricas:
        return

    metricas_limpias = {}

    for clave, valor in metricas.items():
        valor_float = float(valor)

        if not (-float("inf") < valor_float < float("inf")):
            raise ValueError(
                f"La métrica '{clave}' contiene un valor no finito."
            )

        metricas_limpias[str(clave)] = valor_float

    mlflow.log_metrics(metricas_limpias)


# ============================================================
# REGISTRAR TAGS
# ============================================================

def registrar_tags(
    tags: Mapping[str, Any],
) -> None:
    """
    Registra etiquetas descriptivas del experimento.

    Parameters
    ----------
    tags : Mapping[str, Any]
        Diccionario con tags.
    """

    if not tags:
        return

    tags_limpios = {
        str(clave): str(valor)
        for clave, valor in tags.items()
    }

    mlflow.set_tags(tags_limpios)


# ============================================================
# REGISTRAR ARTEFACTOS
# ============================================================

def registrar_artefacto(
    ruta: str | Path,
    artifact_path: str | None = None,
) -> None:
    """
    Registra un archivo como artefacto de MLflow.

    Parameters
    ----------
    ruta : str | Path
        Ruta del archivo.

    artifact_path : str | None
        Carpeta dentro de los artefactos del run.
    """

    ruta = Path(ruta)

    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró el artefacto: {ruta}"
        )

    if not ruta.is_file():
        raise ValueError(
            f"La ruta indicada no corresponde a un archivo: {ruta}"
        )

    mlflow.log_artifact(
        str(ruta),
        artifact_path=artifact_path,
    )


# ============================================================
# REGISTRAR MÚLTIPLES ARTEFACTOS
# ============================================================

def registrar_artefactos(
    artefactos: Mapping[str, str | Path],
) -> None:
    """
    Registra múltiples artefactos.

    Parameters
    ----------
    artefactos : Mapping[str, str | Path]
        Diccionario donde:

        clave   = carpeta del artefacto
        valor   = ruta del archivo
    """

    for carpeta, ruta in artefactos.items():
        registrar_artefacto(
            ruta=ruta,
            artifact_path=carpeta,
        )


# ============================================================
# INFORMACIÓN DEL RUN
# ============================================================

def obtener_run(run_id: str):
    """
    Obtiene información de un run específico.

    Parameters
    ----------
    run_id : str
        Identificador del run.

    Returns
    -------
    mlflow.entities.Run
        Información del run.
    """

    return mlflow.get_run(run_id)


# ============================================================
# INFORMACIÓN DEL EXPERIMENTO
# ============================================================

def obtener_experimento(
    nombre_experimento: str = NOMBRE_EXPERIMENTO,
):
    """
    Obtiene información de un experimento.

    Parameters
    ----------
    nombre_experimento : str
        Nombre del experimento.

    Returns
    -------
    Experiment | None
        Información del experimento.
    """

    return mlflow.get_experiment_by_name(
        nombre_experimento
    )


# ============================================================
# REGISTRO COMPLETO DE EXPERIMENTO
# ============================================================

def registrar_experimento(
    nombre_run: str,
    parametros: Mapping[str, Any],
    metricas: Mapping[str, float],
    tags: Mapping[str, Any] | None = None,
    artefactos: Mapping[str, str | Path] | None = None,
):
    """
    Registra un experimento completo en MLflow.

    El run se crea dinámicamente y se devuelve su información.

    Parameters
    ----------
    nombre_run : str
        Nombre del run.

    parametros : Mapping[str, Any]
        Parámetros del modelo.

    metricas : Mapping[str, float]
        Métricas del modelo.

    tags : Mapping[str, Any] | None
        Tags descriptivos.

    artefactos : Mapping[str, str | Path] | None
        Artefactos a registrar.

    Returns
    -------
    dict
        Información básica del run creado.
    """

    with mlflow.start_run(run_name=nombre_run) as run:

        registrar_parametros(parametros)

        registrar_metricas(metricas)

        if tags is not None:
            registrar_tags(tags)

        if artefactos is not None:
            registrar_artefactos(artefactos)

        return {
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "run_name": nombre_run,
        }