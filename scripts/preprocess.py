
"""
preprocess.py
-------------

Script de ejecución para la preparación y transformación de los datos
del proyecto LiquiditySense.

Flujo:
    1. Carga la base original.
    2. Prepara la estructura temporal y de identificación.
    3. Construye indicadores de liquidez.
    4. Calcula crecimientos YoY y QoQ.
    5. Reemplaza valores infinitos por NaN.
    6. Guarda la base procesada.

Uso:
    python scripts/preprocess.py
"""

from pathlib import Path
import sys

import pandas as pd


# ============================================================
# CONFIGURACIÓN DE RUTAS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"

ARCHIVO_ENTRADA = DATA_RAW_DIR / "call-reports-balance-sheets-Jun2026.dta"
ARCHIVO_SALIDA = DATA_PROCESSED_DIR / "base_liquidez_indicadores.csv"


# ============================================================
# IMPORTACIÓN DEL CÓDIGO REUTILIZABLE
# ============================================================

SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from liquiditysense.data import cargar_base
from liquiditysense.features import construir_indicadores_liquidez


# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================

def main() -> None:
    """Ejecuta el pipeline de preparación e ingeniería de variables."""

    print("=" * 70)
    print("LIQUIDITYSENSE - PREPROCESAMIENTO")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Validación de rutas
    # --------------------------------------------------------

    if not ARCHIVO_ENTRADA.exists():
        raise FileNotFoundError(
            f"No se encontró la base de entrada:\n{ARCHIVO_ENTRADA}\n\n"
            "Coloque el archivo original en data/raw/."
        )

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 2. Carga de datos
    # --------------------------------------------------------

    print("\n[1/4] Cargando base original...")

    base = cargar_base(
        ARCHIVO_ENTRADA,
        convertir_fecha=True,
    )

    print(f"Registros cargados: {len(base):,}")
    print(f"Variables cargadas: {base.shape[1]:,}")

    # --------------------------------------------------------
    # 3. Ingeniería de variables
    # --------------------------------------------------------

    print("\n[2/4] Construyendo indicadores de liquidez...")

    base_indicadores = construir_indicadores_liquidez(base)

    print(
        f"Registros después de transformación: "
        f"{len(base_indicadores):,}"
    )

    print(
        f"Variables después de transformación: "
        f"{base_indicadores.shape[1]:,}"
    )

    # --------------------------------------------------------
    # 4. Control y almacenamiento
    # --------------------------------------------------------

    print("\n[3/4] Aplicando control de valores infinitos...")

    infinitos = (
        base_indicadores.select_dtypes(include="number")
        .isin([float("inf"), float("-inf")])
        .sum()
        .sum()
    )

    print(f"Valores infinitos detectados: {int(infinitos):,}")

    print("\n[4/4] Guardando base procesada...")

    base_indicadores.to_csv(
        ARCHIVO_SALIDA,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\nArchivo generado:")
    print(ARCHIVO_SALIDA)

    print("\n" + "=" * 70)
    print("PREPROCESAMIENTO FINALIZADO CORRECTAMENTE")
    print("=" * 70)


# ============================================================
# PUNTO DE ENTRADA
# ============================================================

if __name__ == "__main__":
    main()

