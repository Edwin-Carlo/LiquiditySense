"""
predict.py
----------

Script de inferencia del modelo LiquiditySense.

Carga:
    - Modelo Random Forest entrenado.
    - Preprocesador ajustado durante entrenamiento.
    - Lista de variables predictoras finales.

Realiza:
    - Validación de artefactos.
    - Carga de un registro real de la base final.
    - Validación de las variables originales de entrada.
    - Transformación mediante el preprocesador persistido.
    - Predicción de probabilidad.
    - Clasificación mediante threshold.
    - Validación del resultado de inferencia.

Uso:
    python scripts/predict.py
"""

from pathlib import Path
import sys

import joblib
import pandas as pd


# ============================================================
# CONFIGURACIÓN
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data" / "processed"

ARCHIVO_MODELO = (
    MODELS_DIR / "random_forest_liquidez.pkl"
)

ARCHIVO_PREPROCESADOR = (
    MODELS_DIR / "preprocesador_liquidez.pkl"
)

ARCHIVO_VARIABLES = (
    MODELS_DIR / "variables_modelo.pkl"
)

ARCHIVO_BASE_FINAL = (
    DATA_DIR / "base_liquidez_final.csv"
)

THRESHOLD = 0.50


# ============================================================
# IMPORTACIÓN DE MÓDULOS
# ============================================================

SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from liquiditysense.preprocessing import convertir_booleanos


# ============================================================
# FUNCIÓN DE PREDICCIÓN
# ============================================================

def realizar_prediccion(
    datos: pd.DataFrame,
    modelo,
    preprocesador,
    variables_modelo: list,
    threshold: float = 0.50,
) -> pd.DataFrame:
    """
    Ejecuta inferencia sobre nuevos registros.

    Parameters
    ----------
    datos:
        DataFrame que contiene las variables originales
        requeridas por el preprocesador.

    modelo:
        Modelo de Machine Learning entrenado.

    preprocesador:
        Preprocesador previamente ajustado con los datos
        de entrenamiento.

    variables_modelo:
        Lista de variables finales utilizadas por el modelo.

    threshold:
        Umbral para convertir la probabilidad en clase.

    Returns
    -------
    pd.DataFrame
        Datos originales junto con probabilidad y predicción.
    """

    datos = datos.copy()

    # --------------------------------------------------------
    # 1. Obtener variables originales de entrada
    # --------------------------------------------------------
    #
    # IMPORTANTE:
    # variables_modelo contiene los nombres finales generados
    # por el preprocesador, por ejemplo:
    #
    # numericas__assets
    # numericas__cash
    #
    # En cambio, feature_names_in_ contiene las variables
    # originales que deben entregarse al preprocesador:
    #
    # assets
    # cash
    #
    # --------------------------------------------------------

    if not hasattr(
        preprocesador,
        "feature_names_in_"
    ):
        raise AttributeError(
            "El preprocesador no contiene el atributo "
            "'feature_names_in_'."
        )

    variables_entrada = list(
        preprocesador.feature_names_in_
    )

    # --------------------------------------------------------
    # 2. Validación de variables originales
    # --------------------------------------------------------

    faltantes = [
        variable
        for variable in variables_entrada
        if variable not in datos.columns
    ]

    if faltantes:
        raise ValueError(
            "Faltan variables originales requeridas "
            "por el preprocesador: "
            f"{faltantes}"
        )

    # --------------------------------------------------------
    # 3. Selección exacta de variables de entrada
    # --------------------------------------------------------

    X = datos[variables_entrada].copy()

    # --------------------------------------------------------
    # 4. Conversión de variables booleanas
    # --------------------------------------------------------

    X = convertir_booleanos(X)

    # --------------------------------------------------------
    # 5. Validación final del esquema
    # --------------------------------------------------------

    columnas_esperadas = list(
        preprocesador.feature_names_in_
    )

    columnas_actuales = list(
        X.columns
    )

    faltantes_preprocesador = [
        columna
        for columna in columnas_esperadas
        if columna not in columnas_actuales
    ]

    if faltantes_preprocesador:
        raise ValueError(
            "El registro de inferencia no contiene "
            "todas las variables esperadas por el "
            f"preprocesador: {faltantes_preprocesador}"
        )

    # --------------------------------------------------------
    # 6. Orden exacto de las variables
    # --------------------------------------------------------

    X = X[columnas_esperadas]

    # --------------------------------------------------------
    # 7. Transformación
    # --------------------------------------------------------

    X_transformado = preprocesador.transform(X)

    # --------------------------------------------------------
    # 8. Validación del número de variables transformadas
    # --------------------------------------------------------

    if X_transformado.shape[1] != len(
        variables_modelo
    ):
        raise ValueError(
            "El número de variables transformadas "
            "no coincide con las variables utilizadas "
            "por el modelo. "
            f"Transformadas: {X_transformado.shape[1]}; "
            f"Esperadas: {len(variables_modelo)}"
        )

    # --------------------------------------------------------
    # 9. Predicción de probabilidad
    # --------------------------------------------------------

    probabilidades = modelo.predict_proba(
        X_transformado
    )[:, 1]

    # --------------------------------------------------------
    # 10. Clasificación mediante threshold
    # --------------------------------------------------------

    predicciones = (
        probabilidades >= threshold
    ).astype(int)

    # --------------------------------------------------------
    # 11. Resultado
    # --------------------------------------------------------

    resultado = datos.copy()

    resultado[
        "probabilidad_deterioro"
    ] = probabilidades

    resultado[
        "prediccion"
    ] = predicciones

    return resultado


# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================

def main() -> None:
    """
    Carga los artefactos y ejecuta una predicción
    utilizando un registro real de la base final.
    """

    print("=" * 70)
    print("LIQUIDYSENSE - PREDICCIÓN")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Validar artefactos
    # --------------------------------------------------------

    print(
        "\n[1/5] Validando artefactos..."
    )

    archivos_modelo = [
        ARCHIVO_MODELO,
        ARCHIVO_PREPROCESADOR,
        ARCHIVO_VARIABLES,
    ]

    for archivo in archivos_modelo:

        if not archivo.exists():

            raise FileNotFoundError(
                "No se encontró el archivo requerido:\n"
                f"{archivo}"
            )

    if not ARCHIVO_BASE_FINAL.exists():

        raise FileNotFoundError(
            "No se encontró la base final requerida "
            "para la inferencia:\n"
            f"{ARCHIVO_BASE_FINAL}"
        )

    print(
        "Artefactos validados correctamente."
    )

    # --------------------------------------------------------
    # 2. Cargar modelo y preprocesador
    # --------------------------------------------------------

    print(
        "\n[2/5] Cargando modelo y preprocesador..."
    )

    modelo = joblib.load(
        ARCHIVO_MODELO
    )

    preprocesador = joblib.load(
        ARCHIVO_PREPROCESADOR
    )

    variables_modelo = joblib.load(
        ARCHIVO_VARIABLES
    )

    print(
        "Modelo cargado correctamente."
    )

    print(
        "Preprocesador cargado correctamente."
    )

    print(
        f"Variables finales del modelo: "
        f"{len(variables_modelo)}"
    )

    # --------------------------------------------------------
    # Validación de variables del modelo
    # --------------------------------------------------------

    if len(variables_modelo) != 71:

        raise ValueError(
            "El número de variables finales del modelo "
            "no coincide con el esperado. "
            f"Encontradas: {len(variables_modelo)}"
        )

    # --------------------------------------------------------
    # Variables originales de entrada
    # --------------------------------------------------------

    if not hasattr(
        preprocesador,
        "feature_names_in_"
    ):

        raise AttributeError(
            "El preprocesador no contiene "
            "'feature_names_in_'."
        )

    variables_entrada = list(
        preprocesador.feature_names_in_
    )

    print(
        f"Variables originales de entrada: "
        f"{len(variables_entrada)}"
    )

    # --------------------------------------------------------
    # Validación de consistencia
    # --------------------------------------------------------

    if len(variables_entrada) != 71:

        raise ValueError(
            "El número de variables originales de entrada "
            "no coincide con las 71 variables esperadas. "
            f"Encontradas: {len(variables_entrada)}"
        )

    # --------------------------------------------------------
    # 3. Cargar registro real de inferencia
    # --------------------------------------------------------

    print(
        "\n[3/5] Preparando registro de inferencia..."
    )

    base = pd.read_csv(
        ARCHIVO_BASE_FINAL,
        low_memory=False,
    )

    print(
        f"Base cargada: "
        f"{base.shape[0]:,} registros"
    )

    # --------------------------------------------------------
    # Validar variables de entrada en la base
    # --------------------------------------------------------

    faltantes_base = [
        variable
        for variable in variables_entrada
        if variable not in base.columns
    ]

    if faltantes_base:

        raise ValueError(
            "La base final no contiene todas las "
            "variables requeridas por el preprocesador: "
            f"{faltantes_base}"
        )

    # --------------------------------------------------------
    # Seleccionar un registro real
    # --------------------------------------------------------
    #
    # Se selecciona el último registro que tenga
    # información completa en las variables de entrada.
    #
    # No se utiliza variables_modelo aquí porque sus nombres
    # corresponden a las variables transformadas.
    #
    # --------------------------------------------------------

    registro = (
        base
        .dropna(
            subset=variables_entrada
        )
        .tail(1)
        .copy()
    )

    if registro.empty:

        raise ValueError(
            "No se encontró un registro válido "
            "para realizar la inferencia."
        )

    print(
        "Registro de inferencia seleccionado "
        "correctamente."
    )

    # --------------------------------------------------------
    # Información del registro seleccionado
    # --------------------------------------------------------

    if "id_rssd" in registro.columns:

        print(
            "Banco: "
            f"{registro.iloc[0]['id_rssd']}"
        )

    if "date" in registro.columns:

        print(
            "Fecha: "
            f"{registro.iloc[0]['date']}"
        )

    # --------------------------------------------------------
    # 4. Ejecutar predicción
    # --------------------------------------------------------

    print(
        "\n[4/5] Ejecutando predicción..."
    )

    resultado = realizar_prediccion(
        datos=registro,
        modelo=modelo,
        preprocesador=preprocesador,
        variables_modelo=variables_modelo,
        threshold=THRESHOLD,
    )

    # --------------------------------------------------------
    # Extraer resultado
    # --------------------------------------------------------

    probabilidad = resultado.iloc[
        0
    ]["probabilidad_deterioro"]

    prediccion = resultado.iloc[
        0
    ]["prediccion"]

    # --------------------------------------------------------
    # Resultado de inferencia
    # --------------------------------------------------------

    print(
        "\nRESULTADO DE INFERENCIA"
    )

    print(
        "-" * 40
    )

    print(
        f"Probabilidad de deterioro: "
        f"{probabilidad:.6f}"
    )

    print(
        f"Threshold utilizado: "
        f"{THRESHOLD:.2f}"
    )

    print(
        f"Predicción: "
        f"{prediccion}"
    )

    # --------------------------------------------------------
    # Interpretación
    # --------------------------------------------------------

    if prediccion == 1:

        print(
            "Interpretación: "
            "Se clasifica el registro como "
            "deterioro de liquidez."
        )

    else:

        print(
            "Interpretación: "
            "No se clasifica el registro como "
            "deterioro de liquidez."
        )

    # --------------------------------------------------------
    # 5. Validación final
    # --------------------------------------------------------

    print(
        "\n[5/5] Validando resultado..."
    )

    # --------------------------------------------------------
    # Validar probabilidad
    # --------------------------------------------------------

    if not 0 <= probabilidad <= 1:

        raise ValueError(
            "La probabilidad generada está fuera "
            "del intervalo [0, 1]."
        )

    # --------------------------------------------------------
    # Validar predicción
    # --------------------------------------------------------

    if prediccion not in [0, 1]:

        raise ValueError(
            "La predicción generada no es binaria."
        )

    # --------------------------------------------------------
    # Validar cantidad de variables
    # --------------------------------------------------------

    if len(variables_modelo) != 71:

        raise ValueError(
            "La cantidad de variables del modelo "
            "no coincide con el esquema esperado."
        )

    print(
        "Validación de inferencia: OK."
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "PREDICCIÓN FINALIZADA CORRECTAMENTE"
    )

    print(
        "=" * 70
    )


# ============================================================
# PUNTO DE ENTRADA
# ============================================================

if __name__ == "__main__":

    main()