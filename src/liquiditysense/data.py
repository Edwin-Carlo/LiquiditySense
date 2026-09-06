
"""
LiquiditySense - Gestión y preparación de datos.

Este módulo contiene las funciones reutilizables relacionadas con:

- Carga de bases de datos.
- Validación básica de datos.
- Construcción de la variable objetivo.
- Preparación del dataset para modelado.
- División temporal Train / Validation / Test.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd


# ============================================================
# TIPOS
# ============================================================

PathLike = Union[str, Path]


# ============================================================
# VARIABLES DE LA BASE ORIGINAL
# ============================================================

# Variables seleccionadas de la base original Call Reports /
# Balance Sheets.
#
# La fuente original contiene muchas más variables, pero para
# LiquiditySense se seleccionaron únicamente las variables
# necesarias para la preparación de la base y la ingeniería
# posterior de indicadores.

VARIABLES_BASE = [
    "id_rssd",
    "date",
    "assets",
    "cash",
    "securities",
    "deposits",
    "demand_deposits",
    "time_deposits",
    "brokered_dep",
    "insured_deposits",
    "ln_tot",
    "ln_tot_gross",
    "ln_re",
    "ln_ci",
    "ln_cons",
    "ln_agr",
    "ln_rre",
    "npl_tot",
    "liab_tot",
    "equity",
    "subdebt",
]


# ============================================================
# 1. CARGA DE DATOS
# ============================================================

def cargar_base(
    ruta: PathLike,
    convertir_fecha: bool = True,
) -> pd.DataFrame:
    """
    Carga una base de datos desde DTA, CSV o Parquet.

    Para archivos DTA se cargan únicamente las 21 variables
    definidas en VARIABLES_BASE, evitando cargar las 194
    variables originales que no son necesarias para LiquiditySense.

    Parameters
    ----------
    ruta : str | Path
        Ruta del archivo.

    convertir_fecha : bool, default=True
        Convierte la columna 'date' a datetime cuando existe.

    Returns
    -------
    pd.DataFrame
        Base de datos cargada.

    Raises
    ------
    FileNotFoundError
        Si el archivo no existe.

    ValueError
        Si el formato no es compatible o la base está vacía.
    """

    ruta = Path(ruta)

    # --------------------------------------------------------
    # Validación de existencia
    # --------------------------------------------------------

    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo:\n{ruta.resolve()}"
        )

    if not ruta.is_file():
        raise ValueError(
            f"La ruta indicada no corresponde a un archivo:\n"
            f"{ruta.resolve()}"
        )

    if ruta.stat().st_size == 0:
        raise ValueError(
            f"El archivo está vacío:\n{ruta.resolve()}"
        )

    # --------------------------------------------------------
    # Identificar extensión
    # --------------------------------------------------------

    extension = ruta.suffix.lower()

    # --------------------------------------------------------
    # Cargar archivo DTA
    # --------------------------------------------------------

    if extension == ".dta":

        try:

            base = pd.read_stata(
                ruta,
                columns=VARIABLES_BASE,
            )

        except ValueError as error:

            raise ValueError(
                "No fue posible cargar el archivo DTA con las "
                "variables requeridas. Verifique que la fuente "
                "contenga las variables definidas en "
                "VARIABLES_BASE."
            ) from error

    # --------------------------------------------------------
    # Cargar Parquet
    # --------------------------------------------------------

    elif extension == ".parquet":

        try:

            base = pd.read_parquet(
                ruta,
                engine="pyarrow",
            )

        except ImportError as error:

            raise ImportError(
                "Para leer archivos Parquet se requiere "
                "la librería 'pyarrow'."
            ) from error

    # --------------------------------------------------------
    # Cargar CSV
    # --------------------------------------------------------

    elif extension == ".csv":

        # Primero intentamos UTF-8 con BOM,
        # que corresponde al formato utilizado
        # en 02_Ingenieria_variables.ipynb.

        try:

            base = pd.read_csv(
                ruta,
                encoding="utf-8-sig",
            )

        except UnicodeDecodeError:

            # Segundo intento con UTF-8 estándar

            try:

                base = pd.read_csv(
                    ruta,
                    encoding="utf-8",
                )

            except UnicodeDecodeError:

                # Último intento para archivos generados
                # en entornos Windows.

                base = pd.read_csv(
                    ruta,
                    encoding="latin-1",
                )

    # --------------------------------------------------------
    # Formato no soportado
    # --------------------------------------------------------

    else:

        raise ValueError(
            f"Formato de archivo no soportado: '{extension}'. "
            "Utilice un archivo .dta, .parquet o .csv."
        )

    # --------------------------------------------------------
    # Validar base
    # --------------------------------------------------------

    if base.empty:

        raise ValueError(
            f"La base de datos está vacía:\n{ruta.resolve()}"
        )

    # --------------------------------------------------------
    # Conversión de fecha
    # --------------------------------------------------------

    if convertir_fecha and "date" in base.columns:

        base["date"] = pd.to_datetime(
            base["date"],
            errors="coerce",
        )

        if base["date"].isna().any():

            cantidad_nan = base["date"].isna().sum()

            raise ValueError(
                f"La columna 'date' contiene "
                f"{cantidad_nan:,} valores que no pudieron "
                "convertirse correctamente a fecha."
            )

    return base


# ============================================================
# 2. CONSTRUCCIÓN DE TARGET
# ============================================================

def crear_target_liquidez(
    base: pd.DataFrame,
    columna_liquidez: str = "liquid_assets_assets",
    percentil: float = 0.10,
) -> tuple[pd.DataFrame, float]:
    """
    Construye la variable objetivo de deterioro de liquidez.

    El deterioro se define utilizando el percentil indicado
    de la variación trimestral de la liquidez.

    El target representa el deterioro observado en el
    siguiente período.

    Parameters
    ----------
    base : pd.DataFrame
        Base con indicadores de liquidez.

    columna_liquidez : str
        Variable utilizada para medir la liquidez.

    percentil : float, default=0.10
        Percentil utilizado para definir el umbral de deterioro.

    Returns
    -------
    base : pd.DataFrame
        Base con las variables auxiliares y target.

    umbral : float
        Umbral de deterioro utilizado.
    """

    if not isinstance(base, pd.DataFrame):

        raise TypeError(
            "El argumento 'base' debe ser un pandas.DataFrame."
        )

    if columna_liquidez not in base.columns:

        raise ValueError(
            f"No existe la columna '{columna_liquidez}'."
        )

    if "id_rssd" not in base.columns:

        raise ValueError(
            "La base debe contener la columna 'id_rssd'."
        )

    if "date" not in base.columns:

        raise ValueError(
            "La base debe contener la columna 'date'."
        )

    if not 0 < percentil < 1:

        raise ValueError(
            "El percentil debe estar entre 0 y 1."
        )

    df = base.copy()

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    df = df.sort_values(
        ["id_rssd", "date"]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Liquidez del período anterior
    # --------------------------------------------------------

    df["liquidez_anterior"] = (
        df.groupby("id_rssd")[columna_liquidez]
        .shift(1)
    )

    # --------------------------------------------------------
    # Cambio QoQ de liquidez
    # --------------------------------------------------------

    df["cambio_liquidez_qoq"] = np.where(
        df["liquidez_anterior"] != 0,
        (
            df[columna_liquidez]
            - df["liquidez_anterior"]
        )
        / df["liquidez_anterior"],
        np.nan,
    )

    # --------------------------------------------------------
    # Umbral de deterioro
    # --------------------------------------------------------

    umbral = df["cambio_liquidez_qoq"].quantile(
        percentil
    )

    if pd.isna(umbral):

        raise ValueError(
            "No fue posible calcular el umbral de deterioro. "
            "Revise la variable de liquidez."
        )

    # --------------------------------------------------------
    # Evento de deterioro
    # --------------------------------------------------------

    df["deterioro_liquidez"] = (
        df["cambio_liquidez_qoq"] <= umbral
    ).astype("Int64")

    # --------------------------------------------------------
    # Target futuro
    # --------------------------------------------------------

    df["target"] = (
        df.groupby("id_rssd")["deterioro_liquidez"]
        .shift(-1)
    )

    return df, float(umbral)


# ============================================================
# 3. PREPARACIÓN DEL DATASET PARA MODELADO
# ============================================================

def preparar_dataset_modelado(
    base: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    list[str],
]:
    """
    Prepara la base para el proceso de Machine Learning.

    Acciones:

    1. Elimina observaciones sin target.
    2. Convierte target a entero.
    3. Ordena temporalmente.
    4. Excluye identificadores.
    5. Excluye variables utilizadas para construir el target.
    6. Excluye indicadores de faltante estructural.
    7. Excluye denominadores auxiliares.

    Returns
    -------
    base_modelo
    X
    y
    variables_excluir
    """

    if not isinstance(base, pd.DataFrame):

        raise TypeError(
            "El argumento 'base' debe ser un pandas.DataFrame."
        )

    columnas_requeridas = [
        "id_rssd",
        "date",
        "target",
    ]

    faltantes = [
        col
        for col in columnas_requeridas
        if col not in base.columns
    ]

    if faltantes:

        raise ValueError(
            "Faltan columnas requeridas: "
            + ", ".join(faltantes)
        )

    df = base.copy()

    # --------------------------------------------------------
    # Eliminar observaciones sin target
    # --------------------------------------------------------

    df = df.dropna(
        subset=["target"]
    ).copy()

    # --------------------------------------------------------
    # Convertir target
    # --------------------------------------------------------

    df["target"] = (
        df["target"]
        .astype(int)
    )

    # --------------------------------------------------------
    # Orden temporal
    # --------------------------------------------------------

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    df = df.sort_values(
        ["date", "id_rssd"]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Variables a excluir
    # --------------------------------------------------------

    variables_excluir = {
        "id_rssd",
        "date",
        "target",
        "liquidez_anterior",
        "cambio_liquidez_qoq",
        "deterioro_liquidez",
    }

    # Indicadores estructurales
    variables_excluir.update(
        [
            col
            for col in df.columns
            if "faltante_estructural" in col
        ]
    )

    # Denominadores auxiliares
    variables_excluir.update(
        [
            col
            for col in df.columns
            if col.endswith("_denominador")
        ]
    )

    variables_excluir = sorted(
        variables_excluir
    )

    # --------------------------------------------------------
    # X e y
    # --------------------------------------------------------

    X = df.drop(
        columns=[
            col
            for col in variables_excluir
            if col in df.columns
        ]
    ).copy()

    y = df["target"].copy()

    return (
        df,
        X,
        y,
        variables_excluir,
    )


# ============================================================
# 4. DIVISIÓN TEMPORAL
# ============================================================

def dividir_temporalmente(
    X: pd.DataFrame,
    y: pd.Series,
    base_modelo: pd.DataFrame,
    fecha_train_fin: str = "2020-12-31",
    fecha_validacion_fin: str = "2022-12-31",
) -> dict:
    """
    Realiza una división temporal Train / Validation / Test.

    Train:
        fecha <= 2020-12-31

    Validation:
        2021-01-01 hasta 2022-12-31

    Test:
        fecha > 2022-12-31

    Returns
    -------
    dict
        Diccionario con los datasets y máscaras temporales.
    """

    if not isinstance(X, pd.DataFrame):

        raise TypeError(
            "X debe ser un pandas.DataFrame."
        )

    if not isinstance(y, pd.Series):

        raise TypeError(
            "y debe ser un pandas.Series."
        )

    if not isinstance(base_modelo, pd.DataFrame):

        raise TypeError(
            "base_modelo debe ser un pandas.DataFrame."
        )

    if len(X) != len(y):

        raise ValueError(
            "X e y deben tener el mismo número de observaciones."
        )

    if len(X) != len(base_modelo):

        raise ValueError(
            "X y base_modelo deben tener el mismo número "
            "de observaciones."
        )

    if "date" not in base_modelo.columns:

        raise ValueError(
            "base_modelo debe contener la columna 'date'."
        )

    fechas = pd.to_datetime(
        base_modelo["date"],
        errors="coerce",
    )

    if fechas.isna().any():

        raise ValueError(
            "Existen fechas inválidas en base_modelo."
        )

    fecha_train_fin = pd.Timestamp(
        fecha_train_fin
    )

    fecha_validacion_fin = pd.Timestamp(
        fecha_validacion_fin
    )

    if fecha_train_fin >= fecha_validacion_fin:

        raise ValueError(
            "fecha_train_fin debe ser anterior "
            "a fecha_validacion_fin."
        )

    # --------------------------------------------------------
    # Máscaras
    # --------------------------------------------------------

    mask_train = (
        fechas <= fecha_train_fin
    )

    mask_validacion = (
        (fechas > fecha_train_fin)
        & (fechas <= fecha_validacion_fin)
    )

    mask_test = (
        fechas > fecha_validacion_fin
    )

    # --------------------------------------------------------
    # Validación
    # --------------------------------------------------------

    if mask_train.sum() == 0:

        raise ValueError(
            "El conjunto Train quedó vacío."
        )

    if mask_validacion.sum() == 0:

        raise ValueError(
            "El conjunto Validation quedó vacío."
        )

    if mask_test.sum() == 0:

        raise ValueError(
            "El conjunto Test quedó vacío."
        )

    # --------------------------------------------------------
    # Retornar datasets
    # --------------------------------------------------------

    resultado = {
        "X_train": X.loc[mask_train].copy(),
        "y_train": y.loc[mask_train].copy(),

        "X_validacion": X.loc[mask_validacion].copy(),
        "y_validacion": y.loc[mask_validacion].copy(),

        "X_test": X.loc[mask_test].copy(),
        "y_test": y.loc[mask_test].copy(),

        "mask_train": mask_train,
        "mask_validacion": mask_validacion,
        "mask_test": mask_test,
    }

    return resultado

