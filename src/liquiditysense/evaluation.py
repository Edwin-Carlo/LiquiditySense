
"""
evaluation.py
-------------

Módulo de evaluación de modelos para LiquiditySense.

Centraliza las métricas y herramientas utilizadas en
04_modelado.ipynb:

- AUC-ROC
- Accuracy
- Precision
- Recall
- F1-score
- Matriz de confusión
- Classification report
- KS
- Curva ROC
- Curva Precision-Recall
- Average Precision
- Evaluación de umbrales
- Selección del mejor umbral
- Comparación de modelos
- Resumen completo de evaluación
"""


from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)


# ============================================================
# VALIDACIÓN
# ============================================================

def validar_probabilidades(
    y_prob: np.ndarray | pd.Series,
) -> np.ndarray:
    """
    Valida y convierte las probabilidades a numpy.

    Parameters
    ----------
    y_prob : array-like
        Probabilidades estimadas para la clase positiva.

    Returns
    -------
    np.ndarray
        Vector de probabilidades validado.
    """

    probabilidades = np.asarray(
        y_prob
    )

    if probabilidades.ndim != 1:
        probabilidades = probabilidades.ravel()

    if np.isnan(probabilidades).any():
        raise ValueError(
            "Las probabilidades contienen valores NaN."
        )

    if np.isinf(probabilidades).any():
        raise ValueError(
            "Las probabilidades contienen valores infinitos."
        )

    if (
        (probabilidades < 0)
        | (probabilidades > 1)
    ).any():

        raise ValueError(
            "Las probabilidades deben estar entre 0 y 1."
        )

    return probabilidades


# ============================================================
# PREDICCIÓN SEGÚN UMBRAL
# ============================================================

def clasificar_por_umbral(
    y_prob: np.ndarray | pd.Series,
    umbral: float = 0.50,
) -> np.ndarray:
    """
    Convierte probabilidades en clases binarias.

    Parameters
    ----------
    y_prob : array-like
        Probabilidades de la clase positiva.

    umbral : float, default=0.50
        Punto de corte utilizado para clasificar.

    Returns
    -------
    np.ndarray
        Clases binarias 0/1.
    """

    if not 0 < umbral < 1:
        raise ValueError(
            "El umbral debe estar entre 0 y 1."
        )

    probabilidades = validar_probabilidades(
        y_prob
    )

    return (
        probabilidades >= umbral
    ).astype(int)


# ============================================================
# MATRIZ DE CONFUSIÓN
# ============================================================

def obtener_matriz_confusion(
    y_true,
    y_pred,
) -> np.ndarray:
    """
    Calcula la matriz de confusión.

    Returns
    -------
    np.ndarray
        Matriz:

            [[TN, FP],
             [FN, TP]]
    """

    return confusion_matrix(
        y_true,
        y_pred,
    )


# ============================================================
# MÉTRICAS PRINCIPALES
# ============================================================

def calcular_metricas(
    y_true,
    y_prob,
    umbral: float = 0.50,
) -> dict:
    """
    Calcula las principales métricas de clasificación.

    Métricas:

    - AUC
    - Accuracy
    - Precision
    - Recall
    - F1

    Parameters
    ----------
    y_true : array-like
        Variable objetivo real.

    y_prob : array-like
        Probabilidades estimadas.

    umbral : float, default=0.50
        Umbral de clasificación.

    Returns
    -------
    dict
        Diccionario con las métricas.
    """

    probabilidades = validar_probabilidades(
        y_prob
    )

    y_pred = clasificar_por_umbral(
        probabilidades,
        umbral,
    )

    metricas = {
        "auc": float(
            roc_auc_score(
                y_true,
                probabilidades,
            )
        ),
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
    }

    return metricas


# ============================================================
# EVALUACIÓN COMPLETA
# ============================================================

def evaluar_modelo(
    y_true,
    y_prob,
    umbral: float = 0.50,
) -> dict:
    """
    Ejecuta una evaluación completa del modelo.

    Incluye:

    - AUC
    - Accuracy
    - Precision
    - Recall
    - F1
    - Matriz de confusión
    - Classification report
    - Predicciones
    - Umbral utilizado

    Las métricas principales se devuelven directamente
    en el diccionario para facilitar su utilización
    desde train.py y otros scripts.

    Returns
    -------
    dict
        Diccionario de resultados.
    """

    probabilidades = validar_probabilidades(
        y_prob
    )

    y_pred = clasificar_por_umbral(
        probabilidades,
        umbral,
    )

    metricas = calcular_metricas(
        y_true,
        probabilidades,
        umbral,
    )

    matriz = obtener_matriz_confusion(
        y_true,
        y_pred,
    )

    reporte = classification_report(
        y_true,
        y_pred,
        zero_division=0,
    )

    return {
        # ----------------------------------------------------
        # Métricas principales
        # ----------------------------------------------------

        "auc": metricas["auc"],
        "accuracy": metricas["accuracy"],
        "precision": metricas["precision"],
        "recall": metricas["recall"],
        "f1": metricas["f1"],

        # ----------------------------------------------------
        # Resultados adicionales
        # ----------------------------------------------------

        "metricas": metricas,
        "y_pred": y_pred,
        "y_prob": probabilidades,
        "matriz_confusion": matriz,
        "classification_report": reporte,
        "umbral": float(umbral),
    }


# ============================================================
# KS
# ============================================================

def calcular_ks(
    y_true,
    y_prob,
) -> float:
    """
    Calcula el estadístico Kolmogorov-Smirnov (KS).

    KS = máximo |TPR - FPR|

    Returns
    -------
    float
        Estadístico KS.
    """

    probabilidades = validar_probabilidades(
        y_prob
    )

    fpr, tpr, _ = roc_curve(
        y_true,
        probabilidades,
    )

    ks = np.max(
        np.abs(
            tpr - fpr
        )
    )

    return float(ks)


# ============================================================
# DATOS CURVA ROC
# ============================================================

def obtener_curva_roc(
    y_true,
    y_prob,
) -> pd.DataFrame:
    """
    Obtiene los puntos necesarios para construir
    la curva ROC.

    Returns
    -------
    pd.DataFrame
        Columnas:

        - fpr
        - tpr
        - threshold
    """

    probabilidades = validar_probabilidades(
        y_prob
    )

    fpr, tpr, thresholds = roc_curve(
        y_true,
        probabilidades,
    )

    return pd.DataFrame({
        "fpr": fpr,
        "tpr": tpr,
        "threshold": thresholds,
    })


# ============================================================
# DATOS PRECISION-RECALL
# ============================================================

def obtener_curva_precision_recall(
    y_true,
    y_prob,
) -> pd.DataFrame:
    """
    Obtiene los puntos necesarios para construir
    la curva Precision-Recall.

    Returns
    -------
    pd.DataFrame
        Columnas:

        - precision
        - recall
    """

    probabilidades = validar_probabilidades(
        y_prob
    )

    precision, recall, thresholds = (
        precision_recall_curve(
            y_true,
            probabilidades,
        )
    )

    return pd.DataFrame({
        "precision": precision,
        "recall": recall,
    })


# ============================================================
# AVERAGE PRECISION
# ============================================================

def calcular_average_precision(
    y_true,
    y_prob,
) -> float:
    """
    Calcula Average Precision (AP).

    Returns
    -------
    float
        Average Precision.
    """

    probabilidades = validar_probabilidades(
        y_prob
    )

    return float(
        average_precision_score(
            y_true,
            probabilidades,
        )
    )


# ============================================================
# EVALUACIÓN DE UN UMBRAL
# ============================================================

def evaluar_umbral(
    y_true,
    y_prob,
    umbral: float,
) -> dict:
    """
    Calcula métricas para un umbral específico.

    Returns
    -------
    dict
        Umbral, Accuracy, Precision, Recall y F1.
    """

    probabilidades = validar_probabilidades(
        y_prob
    )

    y_pred = clasificar_por_umbral(
        probabilidades,
        umbral,
    )

    return {
        "umbral": float(umbral),

        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),

        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),

        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),

        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
    }


# ============================================================
# EVALUACIÓN DE MÚLTIPLES UMBRALES
# ============================================================

def evaluar_umbral_multiple(
    y_true,
    y_prob,
    umbrales: np.ndarray | list[float] | None = None,
) -> pd.DataFrame:
    """
    Evalúa el modelo para múltiples umbrales.

    Si no se especifican umbrales, utiliza valores desde
    0.20 hasta 0.80 con incrementos de 0.01.

    Returns
    -------
    pd.DataFrame
        Tabla de métricas por umbral.
    """

    if umbrales is None:

        umbrales = np.arange(
            0.20,
            0.81,
            0.01,
        )

    resultados = []

    for umbral in umbrales:

        resultado = evaluar_umbral(
            y_true,
            y_prob,
            float(umbral),
        )

        resultados.append(
            resultado
        )

    return pd.DataFrame(
        resultados
    )


# ============================================================
# MEJOR UMBRAL SEGÚN F1
# ============================================================

def seleccionar_umbral_por_f1(
    tabla_umbral: pd.DataFrame,
) -> dict:
    """
    Selecciona el umbral que maximiza F1.

    Returns
    -------
    dict
        Registro correspondiente al mejor umbral.
    """

    if tabla_umbral.empty:
        raise ValueError(
            "La tabla de umbrales está vacía."
        )

    if "f1" not in tabla_umbral.columns:
        raise ValueError(
            "La tabla de umbrales debe contener "
            "la columna 'f1'."
        )

    fila = tabla_umbral.loc[
        tabla_umbral["f1"].idxmax()
    ]

    return fila.to_dict()


# ============================================================
# COMPARACIÓN DE MODELOS
# ============================================================

def comparar_modelos(
    resultados: dict[str, dict],
) -> pd.DataFrame:
    """
    Construye una tabla comparativa de modelos.

    Parameters
    ----------
    resultados : dict
        Diccionario con estructura:

        {
            "Random Forest": {
                "y_true": ...,
                "y_prob": ...,
                "umbral": 0.50
            },
            ...
        }

    Returns
    -------
    pd.DataFrame
        Tabla comparativa de métricas.
    """

    filas = []

    for nombre, resultado in resultados.items():

        metricas = calcular_metricas(
            resultado["y_true"],
            resultado["y_prob"],
            resultado.get(
                "umbral",
                0.50,
            ),
        )

        filas.append({
            "Modelo": nombre,
            "AUC": metricas["auc"],
            "Accuracy": metricas["accuracy"],
            "Precision": metricas["precision"],
            "Recall": metricas["recall"],
            "F1": metricas["f1"],
        })

    return (
        pd.DataFrame(filas)
        .sort_values(
            "AUC",
            ascending=False,
        )
        .reset_index(drop=True)
    )


# ============================================================
# RESUMEN COMPLETO
# ============================================================

def generar_resumen_evaluacion(
    y_true,
    y_prob,
    umbral: float = 0.50,
) -> dict:
    """
    Genera un resumen completo de evaluación.

    Incluye:

    - métricas;
    - matriz de confusión;
    - KS;
    - Average Precision;
    - classification report;
    - predicciones;
    - probabilidades;
    - umbral utilizado.
    """

    evaluacion = evaluar_modelo(
        y_true,
        y_prob,
        umbral,
    )

    ks = calcular_ks(
        y_true,
        y_prob,
    )

    average_precision = (
        calcular_average_precision(
            y_true,
            y_prob,
        )
    )

    return {
        **evaluacion,
        "ks": ks,
        "average_precision": average_precision,
    }
