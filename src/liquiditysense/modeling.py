"""
modeling.py
-----------

Módulo de modelado para el proyecto LiquiditySense.

Este módulo centraliza la construcción, entrenamiento y generación
de predicciones de los modelos utilizados en el Notebook 04.

Modelos considerados
--------------------
1. Random Forest
2. XGBoost
3. HistGradientBoosting
4. Extra Trees

Modelo final
------------
Random Forest optimizado con:

    n_estimators=400
    max_depth=12
    min_samples_leaf=30
    max_features="sqrt"
    class_weight="balanced"
    random_state=42
    n_jobs=-1
"""


from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)

from xgboost import XGBClassifier


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

RANDOM_STATE = 42


# ============================================================
# RANDOM FOREST
# ============================================================

def crear_random_forest(
    n_estimators: int = 300,
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
    max_features: str = "sqrt",
    class_weight: str | dict | None = "balanced",
    random_state: int = RANDOM_STATE,
    n_jobs: int = -1,
) -> RandomForestClassifier:
    """
    Construye un modelo Random Forest.

    Parameters
    ----------
    n_estimators : int
        Número de árboles.

    max_depth : int or None
        Profundidad máxima de los árboles.

    min_samples_leaf : int
        Número mínimo de observaciones por hoja.

    max_features : str
        Número de variables consideradas por división.

    class_weight : str, dict or None
        Estrategia de ponderación de clases.

    random_state : int
        Semilla aleatoria.

    n_jobs : int
        Número de núcleos utilizados.

    Returns
    -------
    RandomForestClassifier
        Modelo Random Forest.
    """

    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=n_jobs,
    )


# ============================================================
# XGBOOST
# ============================================================

def crear_xgboost(
    n_estimators: int = 400,
    max_depth: int = 5,
    learning_rate: float = 0.03,
    subsample: float = 0.85,
    colsample_bytree: float = 0.85,
    scale_pos_weight: float | None = None,
    random_state: int = RANDOM_STATE,
    n_jobs: int = -1,
) -> XGBClassifier:
    """
    Construye el modelo XGBoost utilizado en el proyecto.

    Parameters
    ----------
    scale_pos_weight : float, optional
        Peso aplicado a la clase positiva.

    Returns
    -------
    XGBClassifier
        Modelo XGBoost.
    """

    parametros = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "random_state": random_state,
        "n_jobs": n_jobs,
    }

    if scale_pos_weight is not None:
        parametros["scale_pos_weight"] = scale_pos_weight

    return XGBClassifier(
        **parametros
    )


# ============================================================
# HISTOGRAM GRADIENT BOOSTING
# ============================================================

def crear_hist_gradient_boosting(
    random_state: int = RANDOM_STATE,
) -> HistGradientBoostingClassifier:
    """
    Construye HistGradientBoostingClassifier.
    """

    return HistGradientBoostingClassifier(
        random_state=random_state
    )


# ============================================================
# EXTRA TREES
# ============================================================

def crear_extra_trees(
    n_estimators: int = 300,
    random_state: int = RANDOM_STATE,
    n_jobs: int = -1,
) -> ExtraTreesClassifier:
    """
    Construye ExtraTreesClassifier.
    """

    return ExtraTreesClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
        class_weight="balanced",
    )


# ============================================================
# CÁLCULO DEL PESO DE LA CLASE POSITIVA
# ============================================================

def calcular_scale_pos_weight(
    y: pd.Series | np.ndarray,
) -> float:
    """
    Calcula el parámetro scale_pos_weight de XGBoost.

    Fórmula:

        negativos / positivos
    """

    valores = pd.Series(y)

    positivos = int(
        (valores == 1).sum()
    )

    negativos = int(
        (valores == 0).sum()
    )

    if positivos == 0:
        raise ValueError(
            "No existen observaciones de la clase positiva."
        )

    return negativos / positivos


# ============================================================
# ENTRENAMIENTO
# ============================================================

def entrenar_modelo(
    modelo: Any,
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
) -> Any:
    """
    Entrena un modelo utilizando exclusivamente Train.
    """

    if X_train is None or y_train is None:
        raise ValueError(
            "X_train e y_train no pueden ser None."
        )

    modelo.fit(
        X_train,
        y_train,
    )

    return modelo


# ============================================================
# PREDICCIÓN DE PROBABILIDADES
# ============================================================

def predecir_probabilidad(
    modelo: Any,
    X: pd.DataFrame | np.ndarray,
) -> np.ndarray:
    """
    Genera probabilidades de pertenencia a la clase positiva.
    """

    if not hasattr(
        modelo,
        "predict_proba",
    ):
        raise AttributeError(
            "El modelo no dispone del método predict_proba()."
        )

    probabilidades = modelo.predict_proba(X)

    return probabilidades[:, 1]


# ============================================================
# PREDICCIÓN DE CLASE
# ============================================================

def predecir_clase(
    modelo: Any,
    X: pd.DataFrame | np.ndarray,
    umbral: float = 0.50,
) -> np.ndarray:
    """
    Convierte probabilidades en clasificación binaria.

    Parameters
    ----------
    umbral : float
        Umbral de clasificación.

    Returns
    -------
    np.ndarray
        Predicciones 0/1.
    """

    if not 0 < umbral < 1:
        raise ValueError(
            "El umbral debe estar entre 0 y 1."
        )

    probabilidades = predecir_probabilidad(
        modelo,
        X,
    )

    return (
        probabilidades >= umbral
    ).astype(int)


# ============================================================
# ENTRENAMIENTO DE MODELOS COMPARATIVOS
# ============================================================

def entrenar_modelos_comparativos(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
) -> dict[str, Any]:
    """
    Entrena los modelos utilizados para la comparación inicial.

    Modelos:

        Random Forest
        XGBoost
        HistGradientBoosting
        Extra Trees

    Returns
    -------
    dict
        Diccionario con los modelos entrenados.
    """

    scale_pos_weight = (
        calcular_scale_pos_weight(
            y_train
        )
    )

    modelos = {
        "Random Forest": crear_random_forest(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=1,
            max_features="sqrt",
            class_weight="balanced",
        ),

        "XGBoost": crear_xgboost(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=scale_pos_weight,
        ),

        "HistGradientBoosting":
            crear_hist_gradient_boosting(),

        "Extra Trees": crear_extra_trees(
            n_estimators=300,
        ),
    }

    modelos_entrenados = {}

    for nombre, modelo in modelos.items():

        print(
            f"Entrenando {nombre}..."
        )

        modelo = entrenar_modelo(
            modelo,
            X_train,
            y_train,
        )

        modelos_entrenados[nombre] = modelo

        print(
            f"✅ {nombre} entrenado"
        )

    return modelos_entrenados


# ============================================================
# CONFIGURACIONES RANDOM FOREST
# ============================================================

CONFIGURACIONES_RF = [
    {
        "configuracion": 1,
        "n_estimators": 300,
        "max_depth": 10,
        "min_samples_leaf": 20,
        "max_features": "sqrt",
    },
    {
        "configuracion": 2,
        "n_estimators": 300,
        "max_depth": 12,
        "min_samples_leaf": 20,
        "max_features": "sqrt",
    },
    {
        "configuracion": 3,
        "n_estimators": 400,
        "max_depth": 10,
        "min_samples_leaf": 30,
        "max_features": "sqrt",
    },
    {
        "configuracion": 4,
        "n_estimators": 400,
        "max_depth": 12,
        "min_samples_leaf": 30,
        "max_features": "sqrt",
    },
]


# ============================================================
# RANDOM FOREST OPTIMIZADO
# ============================================================

def crear_random_forest_optimizado(
    configuracion: int = 4,
) -> RandomForestClassifier:
    """
    Construye una de las configuraciones evaluadas
    para la optimización del Random Forest.

    Configuración final seleccionada:

        configuración 4

        n_estimators=400
        max_depth=12
        min_samples_leaf=30
        max_features="sqrt"
        class_weight="balanced"
    """

    configuraciones = {
        config["configuracion"]: config
        for config in CONFIGURACIONES_RF
    }

    if configuracion not in configuraciones:
        raise ValueError(
            "Configuración no válida. "
            "Utilice 1, 2, 3 o 4."
        )

    parametros = configuraciones[
        configuracion
    ].copy()

    parametros.pop(
        "configuracion"
    )

    return RandomForestClassifier(
        **parametros,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


# ============================================================
# ENTRENAMIENTO DEL RANDOM FOREST OPTIMIZADO
# ============================================================

def entrenar_random_forest_optimizado(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    configuracion: int = 4,
) -> RandomForestClassifier:
    """
    Entrena el Random Forest optimizado.
    """

    modelo = crear_random_forest_optimizado(
        configuracion=configuracion
    )

    modelo = entrenar_modelo(
        modelo,
        X_train,
        y_train,
    )

    return modelo


# ============================================================
# IMPORTANCIA DE VARIABLES
# ============================================================

def obtener_importancia_variables(
    modelo: Any,
    nombres_variables: list[str],
) -> pd.DataFrame:
    """
    Obtiene la importancia de variables de modelos basados
    en árboles.

    Returns
    -------
    pd.DataFrame
        Tabla ordenada de mayor a menor importancia.
    """

    if not hasattr(
        modelo,
        "feature_importances_",
    ):
        raise AttributeError(
            "El modelo no contiene "
            "feature_importances_."
        )

    importancia = np.asarray(
        modelo.feature_importances_
    )

    if len(importancia) != len(
        nombres_variables
    ):
        raise ValueError(
            "El número de importancias no coincide "
            "con el número de variables."
        )

    resultado = pd.DataFrame({
        "variable": nombres_variables,
        "importancia": importancia,
    })

    resultado = resultado.sort_values(
        "importancia",
        ascending=False,
    ).reset_index(drop=True)

    resultado["importancia_acumulada"] = (
        resultado["importancia"]
        .cumsum()
    )

    return resultado


# ============================================================
# CONFIGURACIÓN DEL MODELO FINAL
# ============================================================

def obtener_parametros_modelo_final() -> dict:
    """
    Devuelve los parámetros del Random Forest final
    utilizado en LiquiditySense.
    """

    return {
        "modelo": "RandomForestClassifier",
        "n_estimators": 400,
        "max_depth": 12,
        "min_samples_leaf": 30,
        "max_features": "sqrt",
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "umbral": 0.50,
    }


# ============================================================
# CONSTRUCCIÓN DEL MODELO FINAL
# ============================================================

def crear_modelo_final() -> RandomForestClassifier:
    """
    Construye el Random Forest final de LiquiditySense.

    Parámetros:

        n_estimators=400
        max_depth=12
        min_samples_leaf=30
        max_features="sqrt"
        class_weight="balanced"
        random_state=42
        n_jobs=-1
    """

    return RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=30,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


# ============================================================
# ENTRENAMIENTO DEL MODELO FINAL
# ============================================================

def entrenar_modelo_final(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
) -> RandomForestClassifier:
    """
    Entrena el modelo final de LiquiditySense.
    """

    modelo = crear_modelo_final()

    modelo.fit(
        X_train,
        y_train,
    )

    return modelo
    