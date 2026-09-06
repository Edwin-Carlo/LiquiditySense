
"""
preprocessing.py
----------------

Módulo de preprocesamiento para el proyecto LiquiditySense.

Este módulo reproduce la lógica utilizada en:
    04_modelado.ipynb

Responsabilidades
-----------------
1. Identificar variables predictoras.
2. Excluir variables con riesgo de leakage.
3. Excluir variables auxiliares no utilizadas como predictores.
4. Mantener los indicadores estructurales de faltantes como
   variables predictoras.
5. Convertir variables booleanas a formato numérico.
6. Seleccionar variables numéricas.
7. Construir el pipeline de imputación y escalamiento.
8. Ajustar el preprocesador únicamente con Train.
9. Transformar Train, Validation y Test.
10. Recuperar los nombres de las variables transformadas.

Metodología
-----------
Imputación:
    SimpleImputer(strategy="median")

Escalamiento:
    RobustScaler()

Transformación:
    ColumnTransformer
"""


from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


# ============================================================
# VARIABLES DEL TARGET / LEAKAGE
# ============================================================

VARIABLES_LEAKAGE = [
    "liquidez_anterior",
    "cambio_liquidez_qoq",
    "deterioro_liquidez",
    "target",
]


# ============================================================
# VALIDACIÓN
# ============================================================

def validar_dataframe(
    df: pd.DataFrame,
    nombre: str = "DataFrame",
) -> None:
    """
    Valida que el objeto recibido sea un DataFrame no vacío.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a validar.

    nombre : str, default="DataFrame"
        Nombre utilizado en el mensaje de error.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"{nombre} debe ser un pandas.DataFrame."
        )

    if df.empty:
        raise ValueError(
            f"{nombre} está vacío."
        )


# ============================================================
# IDENTIFICACIÓN DE VARIABLES
# ============================================================

def identificar_variables_predictoras(
    df: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """
    Identifica las variables predictoras y las variables excluidas.

    Se excluyen:

    - identificador del banco;
    - fecha;
    - variables del target;
    - variables intermedias utilizadas para construir el target;
    - denominadores auxiliares.

    Se mantienen como variables predictoras:

    - variables financieras;
    - indicadores de liquidez;
    - indicadores de crecimiento YoY;
    - indicadores de crecimiento QoQ;
    - indicadores estructurales de faltantes;
    - variable temporal periodo_banco.

    Los indicadores estructurales de faltantes NO se consideran
    leakage, ya que representan información disponible en el
    momento de la observación y fueron utilizados como
    predictores en 04_modelado.ipynb.

    Parameters
    ----------
    df : pd.DataFrame
        Base de modelado.

    Returns
    -------
    tuple[list[str], list[str]]
        variables_predictoras,
        variables_excluidas
    """

    validar_dataframe(df)

    variables_excluir = []

    # --------------------------------------------------------
    # Identificador y fecha
    # --------------------------------------------------------

    for columna in ["id_rssd", "date"]:

        if columna in df.columns:
            variables_excluir.append(columna)

    # --------------------------------------------------------
    # Variables del target y leakage
    # --------------------------------------------------------

    for columna in VARIABLES_LEAKAGE:

        if columna in df.columns:
            variables_excluir.append(columna)

    # --------------------------------------------------------
    # Denominadores auxiliares
    # --------------------------------------------------------

    # Los denominadores fueron utilizados para construir
    # indicadores, pero no forman parte de las variables
    # predictoras finales del modelo.
    for columna in df.columns:

        if columna.endswith("_denominador"):

            variables_excluir.append(columna)

    # --------------------------------------------------------
    # Eliminar duplicados conservando el orden original
    # --------------------------------------------------------

    variables_excluir = list(
        dict.fromkeys(variables_excluir)
    )

    # --------------------------------------------------------
    # Variables predictoras
    # --------------------------------------------------------

    # IMPORTANTE:
    # No se excluyen las variables que contienen
    # "faltante_estructural", porque forman parte de las
    # 71 variables utilizadas en el modelo del notebook.
    variables_predictoras = [
        columna
        for columna in df.columns
        if columna not in variables_excluir
    ]

    return (
        variables_predictoras,
        variables_excluir,
    )


# ============================================================
# CONVERSIÓN DE BOOLEANOS
# ============================================================

def convertir_booleanos(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convierte columnas booleanas a variables binarias 0/1.

    True  -> 1
    False -> 0

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a transformar.

    Returns
    -------
    pd.DataFrame
        Copia del DataFrame con booleanos convertidos.
    """

    validar_dataframe(df)

    base = df.copy()

    columnas_booleanas = base.select_dtypes(
        include=["bool", "boolean"]
    ).columns.tolist()

    for columna in columnas_booleanas:

        base[columna] = (
            base[columna]
            .astype("int8")
        )

    return base


# ============================================================
# IDENTIFICACIÓN DE VARIABLES NUMÉRICAS
# ============================================================

def identificar_variables_numericas(
    df: pd.DataFrame,
) -> list[str]:
    """
    Identifica las variables numéricas disponibles.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de entrada.

    Returns
    -------
    list[str]
        Lista de variables numéricas.
    """

    validar_dataframe(df)

    columnas_numericas = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    return columnas_numericas


# ============================================================
# CONSTRUCCIÓN DEL PREPROCESADOR
# ============================================================

def construir_preprocesador(
    variables_numericas: list[str],
) -> ColumnTransformer:
    """
    Construye el preprocesador utilizado por el modelo.

    Pipeline:

        SimpleImputer(strategy="median")
                    ↓
              RobustScaler

    Parameters
    ----------
    variables_numericas : list[str]
        Variables numéricas a transformar.

    Returns
    -------
    ColumnTransformer
        Preprocesador sklearn.
    """

    if not variables_numericas:
        raise ValueError(
            "No existen variables numéricas "
            "para construir el preprocesador."
        )

    pipeline_numerico = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                RobustScaler(),
            ),
        ]
    )

    preprocesador = ColumnTransformer(
        transformers=[
            (
                "numericas",
                pipeline_numerico,
                variables_numericas,
            )
        ],
        remainder="drop",
    )

    return preprocesador


# ============================================================
# AJUSTE DEL PREPROCESADOR
# ============================================================

def ajustar_preprocesador(
    X_train: pd.DataFrame,
) -> tuple[
    ColumnTransformer,
    list[str],
]:
    """
    Ajusta el preprocesador utilizando exclusivamente Train.

    Esto evita utilizar información de Validation o Test
    durante el entrenamiento del modelo.

    Parameters
    ----------
    X_train : pd.DataFrame
        Variables predictoras de entrenamiento.

    Returns
    -------
    tuple
        preprocesador ajustado,
        variables numéricas.
    """

    validar_dataframe(
        X_train,
        "X_train",
    )

    X_train = convertir_booleanos(
        X_train
    )

    variables_numericas = identificar_variables_numericas(
        X_train
    )

    if not variables_numericas:
        raise ValueError(
            "X_train no contiene variables numéricas."
        )

    preprocesador = construir_preprocesador(
        variables_numericas
    )

    # --------------------------------------------------------
    # FIT EXCLUSIVAMENTE CON TRAIN
    # --------------------------------------------------------

    preprocesador.fit(
        X_train
    )

    return (
        preprocesador,
        variables_numericas,
    )


# ============================================================
# TRANSFORMACIÓN DE DATOS
# ============================================================

def transformar_datos(
    preprocesador: ColumnTransformer,
    X: pd.DataFrame,
) -> pd.DataFrame:
    """
    Transforma un conjunto de datos utilizando un
    preprocesador previamente ajustado.

    Importante
    ----------
    El preprocesador NO se vuelve a ajustar.

    Parameters
    ----------
    preprocesador : ColumnTransformer
        Preprocesador previamente ajustado con Train.

    X : pd.DataFrame
        Datos a transformar.

    Returns
    -------
    pd.DataFrame
        Datos transformados.
    """

    validar_dataframe(
        X,
        "X",
    )

    X = convertir_booleanos(
        X
    )

    # --------------------------------------------------------
    # TRANSFORMACIÓN SIN NUEVO FIT
    # --------------------------------------------------------

    X_transformado = preprocesador.transform(
        X
    )

    nombres_variables = (
        preprocesador
        .get_feature_names_out()
    )

    X_transformado = pd.DataFrame(
        X_transformado,
        columns=nombres_variables,
        index=X.index,
    )

    return X_transformado


# ============================================================
# PREPROCESAMIENTO COMPLETO
# ============================================================

def preparar_variables_modelado(
    X_train: pd.DataFrame,
    X_validacion: pd.DataFrame,
    X_test: pd.DataFrame,
) -> dict:
    """
    Ejecuta el proceso completo de preprocesamiento.

    El preprocesador se ajusta exclusivamente con Train.

    Flujo:

        X_train
           ↓
        FIT
           ↓
    Preprocesador
       ↙   ↓   ↘
    Train Val  Test
       ↓    ↓    ↓
    Transformación

    Parameters
    ----------
    X_train : pd.DataFrame
        Predictoras de entrenamiento.

    X_validacion : pd.DataFrame
        Predictoras de validación.

    X_test : pd.DataFrame
        Predictoras de prueba.

    Returns
    -------
    dict
        Diccionario con:

        - preprocesador
        - variables_numericas
        - X_train_transformado
        - X_validacion_transformado
        - X_test_transformado
        - nombres_variables
    """

    validar_dataframe(
        X_train,
        "X_train",
    )

    validar_dataframe(
        X_validacion,
        "X_validacion",
    )

    validar_dataframe(
        X_test,
        "X_test",
    )

    # --------------------------------------------------------
    # Conversión de booleanos
    # --------------------------------------------------------

    X_train = convertir_booleanos(
        X_train
    )

    X_validacion = convertir_booleanos(
        X_validacion
    )

    X_test = convertir_booleanos(
        X_test
    )

    # --------------------------------------------------------
    # Variables numéricas
    # --------------------------------------------------------

    variables_numericas = identificar_variables_numericas(
        X_train
    )

    if not variables_numericas:
        raise ValueError(
            "No existen variables numéricas "
            "en X_train."
        )

    # --------------------------------------------------------
    # Construcción del preprocesador
    # --------------------------------------------------------

    preprocesador = construir_preprocesador(
        variables_numericas
    )

    # --------------------------------------------------------
    # FIT SOLO CON TRAIN
    # --------------------------------------------------------

    preprocesador.fit(
        X_train
    )

    # --------------------------------------------------------
    # TRANSFORMACIÓN
    # --------------------------------------------------------

    X_train_transformado = transformar_datos(
        preprocesador,
        X_train,
    )

    X_validacion_transformado = transformar_datos(
        preprocesador,
        X_validacion,
    )

    X_test_transformado = transformar_datos(
        preprocesador,
        X_test,
    )

    # --------------------------------------------------------
    # Nombres de variables
    # --------------------------------------------------------

    nombres_variables = (
        preprocesador
        .get_feature_names_out()
        .tolist()
    )

    return {
        "preprocesador": preprocesador,
        "variables_numericas": variables_numericas,
        "X_train_transformado": X_train_transformado,
        "X_validacion_transformado": X_validacion_transformado,
        "X_test_transformado": X_test_transformado,
        "nombres_variables": nombres_variables,
    }


# ============================================================
# VALIDACIÓN DEL RESULTADO
# ============================================================

def validar_datos_transformados(
    X: pd.DataFrame,
    nombre: str = "X",
) -> None:
    """
    Comprueba que el conjunto transformado no contenga
    valores NaN ni infinitos.

    Parameters
    ----------
    X : pd.DataFrame
        Datos transformados.

    nombre : str, default="X"
        Nombre del conjunto para mensajes de error.
    """

    validar_dataframe(
        X,
        nombre,
    )

    cantidad_nan = int(
        X.isna().sum().sum()
    )

    cantidad_inf = int(
        np.isinf(X.to_numpy()).sum()
    )

    if cantidad_nan > 0:

        raise ValueError(
            f"{nombre} contiene "
            f"{cantidad_nan:,} valores NaN."
        )

    if cantidad_inf > 0:

        raise ValueError(
            f"{nombre} contiene "
            f"{cantidad_inf:,} valores infinitos."
        )


# ============================================================
# RESUMEN DEL PREPROCESAMIENTO
# ============================================================

def resumen_preprocesamiento(
    resultado: dict,
) -> pd.DataFrame:
    """
    Genera un resumen del proceso de preprocesamiento.

    Parameters
    ----------
    resultado : dict
        Resultado generado por preparar_variables_modelado().

    Returns
    -------
    pd.DataFrame
        Resumen de Train, Validación y Test.
    """

    X_train = resultado[
        "X_train_transformado"
    ]

    X_validacion = resultado[
        "X_validacion_transformado"
    ]

    X_test = resultado[
        "X_test_transformado"
    ]

    resumen = pd.DataFrame({
        "conjunto": [
            "Train",
            "Validación",
            "Test",
        ],
        "filas": [
            len(X_train),
            len(X_validacion),
            len(X_test),
        ],
        "variables": [
            X_train.shape[1],
            X_validacion.shape[1],
            X_test.shape[1],
        ],
        "missing": [
            X_train.isna().sum().sum(),
            X_validacion.isna().sum().sum(),
            X_test.isna().sum().sum(),
        ],
        "infinitos": [
            np.isinf(X_train.to_numpy()).sum(),
            np.isinf(X_validacion.to_numpy()).sum(),
            np.isinf(X_test.to_numpy()).sum(),
        ],
    })

    return resumen

