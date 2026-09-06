
"""
features.py
-----------

Módulo para la construcción de variables derivadas e indicadores
de liquidez para el proyecto LiquiditySense.

Este módulo reproduce la lógica utilizada en:
    02_Ingenieria_variables.ipynb

Flujo:
    base_liquidez.parquet
            ↓
    indicadores de liquidez
            ↓
    crecimientos YoY
            ↓
    crecimientos QoQ
            ↓
    reemplazo de infinitos
            ↓
    base_liquidez_indicadores
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================
# VARIABLES REQUERIDAS
# ============================================================

VARIABLES_BASE_REQUERIDAS = [
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
    "npl_tot",
    "liab_tot",
    "equity",
]


# ============================================================
# INDICADORES DE LIQUIDEZ
# ============================================================

INDICADORES_LIQUIDEZ = [
    "liquidez_inmediata",
    "cash_deposits",
    "securities_deposits",
    "loan_deposit",
    "loan_assets",
    "deposits_assets",
    "brokered_deposits_ratio",
    "insured_deposits_ratio",
    "time_deposits_ratio",
    "demand_deposits_ratio",
    "leverage_ratio",
    "liquid_assets_liabilities",
    "capital_assets_ratio",
    "npl_ratio",
    "liquid_assets_assets",
    "time_deposits_assets",
    "demand_deposits_assets",
    "loans_liquid_assets",
]


# ============================================================
# CRECIMIENTOS YoY
# ============================================================

INDICADORES_YOY = [
    "loan_growth_yoy",
    "deposit_growth_yoy",
    "cash_growth_yoy",
    "securities_growth_yoy",
    "liabilities_growth_yoy",
    "assets_growth_yoy",
    "liquid_assets_growth_yoy",
]


# ============================================================
# CRECIMIENTOS QoQ
# ============================================================

INDICADORES_QOQ = [
    "deposit_growth_qoq",
    "cash_growth_qoq",
    "liquid_assets_growth_qoq",
    "loan_growth_qoq",
    "liabilities_growth_qoq",
    "assets_growth_qoq",
    "securities_growth_qoq",
    "equity_growth_qoq",
]


# ============================================================
# VALIDACIÓN DE COLUMNAS
# ============================================================

def validar_columnas_requeridas(
    df: pd.DataFrame,
    columnas: list[str] | None = None,
) -> None:
    """
    Valida que el DataFrame contenga las columnas necesarias.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de entrada.

    columnas : list[str], optional
        Lista de columnas requeridas. Si no se especifica,
        se utilizan VARIABLES_BASE_REQUERIDAS.

    Raises
    ------
    TypeError
        Si df no es un DataFrame.

    ValueError
        Si existen columnas requeridas ausentes.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "El objeto de entrada debe ser un pandas.DataFrame."
        )

    if columnas is None:
        columnas = VARIABLES_BASE_REQUERIDAS

    faltantes = [
        columna
        for columna in columnas
        if columna not in df.columns
    ]

    if faltantes:
        raise ValueError(
            "Faltan columnas requeridas: "
            + ", ".join(faltantes)
        )


# ============================================================
# PREPARACIÓN DE LA BASE
# ============================================================

def preparar_base_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepara la base antes de construir los indicadores.

    Operaciones:
    - valida columnas;
    - realiza una copia;
    - convierte date a datetime;
    - ordena por banco y fecha.

    Returns
    -------
    pd.DataFrame
        Base preparada.
    """

    validar_columnas_requeridas(df)

    base = df.copy()

    if "date" in base.columns:
        base["date"] = pd.to_datetime(
            base["date"],
            errors="coerce",
        )

        if base["date"].isna().any():
            raise ValueError(
                "La columna 'date' contiene valores "
                "que no pudieron convertirse a fecha."
            )

    base = base.sort_values(
        ["id_rssd", "date"]
    ).reset_index(drop=True)

    return base


# ============================================================
# INDICADORES DE LIQUIDEZ
# ============================================================

def crear_indicadores_liquidez(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construye los indicadores estructurales de liquidez.

    Los cálculos reproducen la lógica del Notebook 02.

    Returns
    -------
    pd.DataFrame
        DataFrame con los indicadores agregados.
    """

    validar_columnas_requeridas(df)

    base = df.copy()

    # --------------------------------------------------------
    # Activos líquidos
    # --------------------------------------------------------

    base["liquid_assets"] = (
        base["cash"] +
        base["securities"]
    )

    # --------------------------------------------------------
    # Liquidez inmediata
    # --------------------------------------------------------

    base["liquidez_inmediata"] = np.where(
        base["deposits"] != 0,
        base["liquid_assets"] / base["deposits"],
        np.nan,
    )

    # --------------------------------------------------------
    # Cash / depósitos
    # --------------------------------------------------------

    base["cash_deposits"] = np.where(
        base["deposits"] != 0,
        base["cash"] / base["deposits"],
        np.nan,
    )

    # --------------------------------------------------------
    # Securities / depósitos
    # --------------------------------------------------------

    base["securities_deposits"] = np.where(
        base["deposits"] != 0,
        base["securities"] / base["deposits"],
        np.nan,
    )

    # --------------------------------------------------------
    # Loans / depósitos
    # --------------------------------------------------------

    base["loan_deposit"] = np.where(
        base["deposits"] != 0,
        base["ln_tot"] / base["deposits"],
        np.nan,
    )

    # --------------------------------------------------------
    # Loans / activos
    # --------------------------------------------------------

    base["loan_assets"] = np.where(
        base["assets"] != 0,
        base["ln_tot"] / base["assets"],
        np.nan,
    )

    # --------------------------------------------------------
    # Depósitos / activos
    # --------------------------------------------------------

    base["deposits_assets"] = np.where(
        base["assets"] != 0,
        base["deposits"] / base["assets"],
        np.nan,
    )

    # --------------------------------------------------------
    # Brokered deposits / depósitos
    # --------------------------------------------------------

    base["brokered_deposits_ratio"] = np.where(
        base["deposits"] != 0,
        base["brokered_dep"] / base["deposits"],
        np.nan,
    )

    # --------------------------------------------------------
    # Insured deposits / depósitos
    # --------------------------------------------------------

    base["insured_deposits_ratio"] = np.where(
        base["deposits"] != 0,
        base["insured_deposits"] / base["deposits"],
        np.nan,
    )

    # --------------------------------------------------------
    # Time deposits / depósitos
    # --------------------------------------------------------

    base["time_deposits_ratio"] = np.where(
        base["deposits"] != 0,
        base["time_deposits"] / base["deposits"],
        np.nan,
    )

    # --------------------------------------------------------
    # Demand deposits / depósitos
    # --------------------------------------------------------

    base["demand_deposits_ratio"] = np.where(
        base["deposits"] != 0,
        base["demand_deposits"] / base["deposits"],
        np.nan,
    )

    # --------------------------------------------------------
    # Pasivos / activos
    # --------------------------------------------------------

    base["leverage_ratio"] = np.where(
        base["assets"] != 0,
        base["liab_tot"] / base["assets"],
        np.nan,
    )

    # --------------------------------------------------------
    # Activos líquidos / pasivos
    # --------------------------------------------------------

    base["liquid_assets_liabilities"] = np.where(
        base["liab_tot"] != 0,
        base["liquid_assets"] / base["liab_tot"],
        np.nan,
    )

    # --------------------------------------------------------
    # Capital / activos
    # --------------------------------------------------------

    base["capital_assets_ratio"] = np.where(
        base["assets"] != 0,
        base["equity"] / base["assets"],
        np.nan,
    )

    # --------------------------------------------------------
    # NPL ratio
    # --------------------------------------------------------

    base["npl_ratio"] = np.where(
        base["ln_tot"] != 0,
        base["npl_tot"] / base["ln_tot"],
        np.nan,
    )

    # --------------------------------------------------------
    # Activos líquidos / activos
    # --------------------------------------------------------

    base["liquid_assets_assets"] = np.where(
        base["assets"] != 0,
        base["liquid_assets"] / base["assets"],
        np.nan,
    )

    # --------------------------------------------------------
    # Time deposits / activos
    # --------------------------------------------------------

    base["time_deposits_assets"] = np.where(
        base["assets"] != 0,
        base["time_deposits"] / base["assets"],
        np.nan,
    )

    # --------------------------------------------------------
    # Demand deposits / activos
    # --------------------------------------------------------

    base["demand_deposits_assets"] = np.where(
        base["assets"] != 0,
        base["demand_deposits"] / base["assets"],
        np.nan,
    )

    # --------------------------------------------------------
    # Loans / activos líquidos
    # --------------------------------------------------------

    base["loans_liquid_assets"] = np.where(
        base["liquid_assets"] != 0,
        base["ln_tot"] / base["liquid_assets"],
        np.nan,
    )

    return base


# ============================================================
# CRECIMIENTOS YoY
# ============================================================

def crear_crecimientos_yoy(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construye los crecimientos interanuales (YoY).

    Se utiliza exactamente pct_change(periods=4), replicando
    la metodología utilizada en el Notebook 02.

    El período 4 representa cuatro observaciones trimestrales.

    fill_method=None evita el relleno implícito de valores
    faltantes realizado por versiones anteriores de pandas.

    Returns
    -------
    pd.DataFrame
        DataFrame con variables YoY agregadas.
    """

    validar_columnas_requeridas(df)

    base = df.copy()

    # Orden temporal por banco
    base = base.sort_values(
        ["id_rssd", "date"]
    ).reset_index(drop=True)

    grupo = base.groupby("id_rssd")

    # --------------------------------------------------------
    # Crecimiento YoY de cartera
    # --------------------------------------------------------

    base["loan_growth_yoy"] = (
        grupo["ln_tot"].pct_change(
            periods=4,
            fill_method=None,
        )
    )

    # --------------------------------------------------------
    # Crecimiento YoY de depósitos
    # --------------------------------------------------------

    base["deposit_growth_yoy"] = (
        grupo["deposits"].pct_change(
            periods=4,
            fill_method=None,
        )
    )

    # --------------------------------------------------------
    # Crecimiento YoY de efectivo
    # --------------------------------------------------------

    base["cash_growth_yoy"] = (
        grupo["cash"].pct_change(
            periods=4,
            fill_method=None,
        )
    )

    # --------------------------------------------------------
    # Crecimiento YoY de valores
    # --------------------------------------------------------

    base["securities_growth_yoy"] = (
        grupo["securities"].pct_change(
            periods=4,
            fill_method=None,
        )
    )

    # --------------------------------------------------------
    # Crecimiento YoY de pasivos
    # --------------------------------------------------------

    base["liabilities_growth_yoy"] = (
        grupo["liab_tot"].pct_change(
            periods=4,
            fill_method=None,
        )
    )

    # --------------------------------------------------------
    # Crecimiento YoY de activos
    # --------------------------------------------------------

    base["assets_growth_yoy"] = (
        grupo["assets"].pct_change(
            periods=4,
            fill_method=None,
        )
    )

    # --------------------------------------------------------
    # Crecimiento YoY de activos líquidos
    # --------------------------------------------------------

    base["liquid_assets_growth_yoy"] = (
        grupo["liquid_assets"].pct_change(
            periods=4,
            fill_method=None,
        )
    )

    return base


# ============================================================
# CRECIMIENTOS QoQ
# ============================================================

def crear_crecimientos_qoq(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construye los crecimientos trimestrales (QoQ).

    Se utiliza exactamente pct_change(), replicando la lógica
    utilizada en el Notebook 02.

    fill_method=None evita el relleno implícito de valores
    faltantes realizado por versiones anteriores de pandas.

    Returns
    -------
    pd.DataFrame
        DataFrame con variables QoQ agregadas.
    """

    validar_columnas_requeridas(df)

    base = df.copy()

    # Orden temporal por banco
    base = base.sort_values(
        ["id_rssd", "date"]
    ).reset_index(drop=True)

    grupo = base.groupby("id_rssd")

    # --------------------------------------------------------
    # Crecimiento QoQ de depósitos
    # --------------------------------------------------------

    base["deposit_growth_qoq"] = (
        grupo["deposits"].pct_change(
            fill_method=None,
        )
    )

    # --------------------------------------------------------
    # Crecimiento QoQ de efectivo
    # --------------------------------------------------------

    base["cash_growth_qoq"] = (
        grupo["cash"].pct_change(
            fill_method=None,
        )
    )

    # --------------------------------------------------------
    # Crecimiento QoQ de activos líquidos
    # --------------------------------------------------------

    base["liquid_assets_growth_qoq"] = (
        grupo["liquid_assets"].pct_change(
            fill_method=None,
        )
    )

    # --------------------------------------------------------
    # Crecimiento QoQ de cartera
    # --------------------------------------------------------

    base["loan_growth_qoq"] = (
        grupo["ln_tot"].pct_change(
            fill_method=None,
        )
    )

    # --------------------------------------------------------
    # Crecimiento QoQ de pasivos
    # --------------------------------------------------------

    base["liabilities_growth_qoq"] = (
        grupo["liab_tot"].pct_change(
            fill_method=None,
        )
    )

    # --------------------------------------------------------
    # Crecimiento QoQ de activos
    # --------------------------------------------------------

    base["assets_growth_qoq"] = (
        grupo["assets"].pct_change(
            fill_method=None,
        )
    )

    # --------------------------------------------------------
    # Crecimiento QoQ de valores
    # --------------------------------------------------------

    base["securities_growth_qoq"] = (
        grupo["securities"].pct_change(
            fill_method=None,
        )
    )

    # --------------------------------------------------------
    # Crecimiento QoQ de patrimonio
    # --------------------------------------------------------

    base["equity_growth_qoq"] = (
        grupo["equity"].pct_change(
            fill_method=None,
        )
    )

    return base


# ============================================================
# REEMPLAZO DE INFINITOS
# ============================================================

def reemplazar_infinito_por_nan(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reemplaza valores infinitos positivos y negativos por NaN.

    Returns
    -------
    pd.DataFrame
        DataFrame sin valores infinitos.
    """

    base = df.copy()

    base = base.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return base


# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================

def construir_indicadores_liquidez(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ejecuta todo el proceso de ingeniería de variables.

    Flujo:

        1. Preparación de la base
        2. Indicadores de liquidez
        3. Crecimientos YoY
        4. Crecimientos QoQ
        5. Reemplazo de infinitos
        6. Orden final

    Parameters
    ----------
    df : pd.DataFrame
        Base de entrada.

    Returns
    -------
    pd.DataFrame
        Base con indicadores de liquidez y crecimientos.
    """

    # --------------------------------------------------------
    # 1. Preparar base
    # --------------------------------------------------------

    base = preparar_base_features(df)

    # --------------------------------------------------------
    # 2. Indicadores de liquidez
    # --------------------------------------------------------

    base = crear_indicadores_liquidez(base)

    # --------------------------------------------------------
    # 3. Crecimientos YoY
    # --------------------------------------------------------

    base = crear_crecimientos_yoy(base)

    # --------------------------------------------------------
    # 4. Crecimientos QoQ
    # --------------------------------------------------------

    base = crear_crecimientos_qoq(base)

    # --------------------------------------------------------
    # 5. Reemplazar infinitos
    # --------------------------------------------------------

    base = reemplazar_infinito_por_nan(base)

    # --------------------------------------------------------
    # 6. Orden final
    # --------------------------------------------------------

    base = base.sort_values(
        ["id_rssd", "date"]
    ).reset_index(drop=True)

    return base

