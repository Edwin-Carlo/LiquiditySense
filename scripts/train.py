
"""
train.py
--------

Script de entrenamiento del modelo LiquiditySense.

Flujo:
    1. Carga la base final.
    2. Construye la variable objetivo.
    3. Prepara el dataset de modelado.
    4. Identifica las 71 variables predictoras.
    5. Realiza la división temporal.
    6. Ajusta el preprocesador exclusivamente con Train.
    7. Entrena el Random Forest optimizado.
    8. Evalúa el modelo sobre Test.
    9. Guarda el modelo, preprocesador y variables.
    10. Registra parámetros y métricas en MLflow.

Uso:
    python scripts/train.py
"""


from pathlib import Path
import sys
import joblib

import pandas as pd


# ============================================================
# CONFIGURACIÓN
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"

ARCHIVO_ENTRADA = (
    DATA_PROCESSED_DIR / "base_liquidez_final.csv"
)

ARCHIVO_MODELO = (
    MODELS_DIR / "random_forest_liquidez.pkl"
)

ARCHIVO_PREPROCESADOR = (
    MODELS_DIR / "preprocesador_liquidez.pkl"
)

ARCHIVO_VARIABLES = (
    MODELS_DIR / "variables_modelo.pkl"
)

# Umbral final seleccionado en 04_modelado.ipynb
THRESHOLD = 0.50


# ============================================================
# IMPORTACIÓN DE MÓDULOS REUTILIZABLES
# ============================================================

SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from liquiditysense.data import (
    cargar_base,
    crear_target_liquidez,
    dividir_temporalmente,
)

from liquiditysense.preprocessing import (
    identificar_variables_predictoras,
    preparar_variables_modelado,
    validar_datos_transformados,
)

from liquiditysense.modeling import (
    crear_modelo_final,
    entrenar_modelo,
)

from liquiditysense.evaluation import (
    evaluar_modelo,
)

from liquiditysense.tracking import (
    configurar_mlflow,
    iniciar_run,
    registrar_parametros,
    registrar_metricas,
    registrar_tags,
    registrar_artefactos,
)


# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================

def main() -> None:
    """Ejecuta el entrenamiento completo del modelo."""

    print("=" * 70)
    print("LIQUIDITYSENSE - ENTRENAMIENTO")
    print("=" * 70)

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # 1. CARGAR BASE FINAL
    # ========================================================

    print("\n[1/8] Cargando base final...")

    if not ARCHIVO_ENTRADA.exists():

        raise FileNotFoundError(
            f"No se encontró la base final:\n"
            f"{ARCHIVO_ENTRADA}\n\n"
            "Debe generarse previamente desde el proceso de EDA."
        )

    base = cargar_base(
        ARCHIVO_ENTRADA,
        convertir_fecha=True,
    )

    print(
        f"Registros: {len(base):,}"
    )

    print(
        f"Variables: {base.shape[1]:,}"
    )

    # ========================================================
    # 2. CONSTRUCCIÓN DEL TARGET
    # ========================================================

    print("\n[2/8] Construyendo variable objetivo...")

    base, umbral_deterioro = crear_target_liquidez(
        base
    )

    print(
        "Umbral de deterioro utilizado: "
        f"{umbral_deterioro:.6f}"
    )

    # ========================================================
    # 3. PREPARACIÓN DEL DATASET
    # ========================================================

    print(
        "\n[3/8] Preparando dataset para modelado..."
    )

    # --------------------------------------------------------
    # Eliminar registros sin target
    # --------------------------------------------------------

    base_modelo = base[
        base["target"].notna()
    ].copy()

    print(
        f"Registros modelables: "
        f"{len(base_modelo):,}"
    )

    # --------------------------------------------------------
    # Identificación de variables predictoras
    # --------------------------------------------------------

    variables_predictoras, variables_excluir = (
        identificar_variables_predictoras(
            base_modelo
        )
    )

    # --------------------------------------------------------
    # Construcción de X e y
    # --------------------------------------------------------

    X = base_modelo[
        variables_predictoras
    ].copy()

    y = base_modelo[
        "target"
    ].astype(int)

    print(
        f"Variables predictoras iniciales: "
        f"{X.shape[1]:,}"
    )

    # --------------------------------------------------------
    # Validación de la cantidad esperada
    # --------------------------------------------------------

    if X.shape[1] != 71:

        raise ValueError(
            "\nLa cantidad de variables predictoras "
            "no coincide con la metodología de "
            "04_modelado.ipynb.\n\n"
            f"Variables encontradas: {X.shape[1]}\n"
            "Variables esperadas: 71\n\n"
            "Revise la selección de variables o la "
            "estructura de base_liquidez_final.csv."
        )

    # ========================================================
    # 4. DIVISIÓN TEMPORAL
    # ========================================================

    print(
        "\n[4/8] Realizando división temporal..."
    )

    splits = dividir_temporalmente(
        X=X,
        y=y,
        base_modelo=base_modelo,
        fecha_train_fin="2020-12-31",
        fecha_validacion_fin="2022-12-31",
    )

    X_train = splits["X_train"]
    X_valid = splits["X_validacion"]
    X_test = splits["X_test"]

    y_train = splits["y_train"]
    y_valid = splits["y_validacion"]
    y_test = splits["y_test"]

    print(
        f"Train:      {len(X_train):,}"
    )

    print(
        f"Validation: {len(X_valid):,}"
    )

    print(
        f"Test:       {len(X_test):,}"
    )

    # ========================================================
    # 5. PREPROCESAMIENTO
    # ========================================================

    print(
        "\n[5/8] Ajustando preprocesador..."
    )

    resultado_preprocesamiento = (
        preparar_variables_modelado(
            X_train=X_train,
            X_validacion=X_valid,
            X_test=X_test,
        )
    )

    preprocesador = (
        resultado_preprocesamiento[
            "preprocesador"
        ]
    )

    variables_numericas = (
        resultado_preprocesamiento[
            "variables_numericas"
        ]
    )

    X_train_transformado = (
        resultado_preprocesamiento[
            "X_train_transformado"
        ]
    )

    X_valid_transformado = (
        resultado_preprocesamiento[
            "X_validacion_transformado"
        ]
    )

    X_test_transformado = (
        resultado_preprocesamiento[
            "X_test_transformado"
        ]
    )

    nombres_variables = (
        resultado_preprocesamiento[
            "nombres_variables"
        ]
    )

    print(
        "Variables utilizadas por el modelo: "
        f"{len(nombres_variables)}"
    )

    # --------------------------------------------------------
    # Validación de datos transformados
    # --------------------------------------------------------

    validar_datos_transformados(
        X_train_transformado,
        "X_train_transformado",
    )

    validar_datos_transformados(
        X_valid_transformado,
        "X_valid_transformado",
    )

    validar_datos_transformados(
        X_test_transformado,
        "X_test_transformado",
    )

    print(
        "Validación del preprocesamiento: OK"
    )

    print(
        f"Shape Train: {X_train_transformado.shape}"
    )

    print(
        f"Shape Validation: {X_valid_transformado.shape}"
    )

    print(
        f"Shape Test: {X_test_transformado.shape}"
    )

    # ========================================================
    # 6. ENTRENAMIENTO
    # ========================================================

    print(
        "\n[6/8] Entrenando Random Forest optimizado..."
    )

    # La configuración del modelo se encuentra centralizada
    # en modeling.py.
    modelo = crear_modelo_final()

    modelo = entrenar_modelo(
        modelo,
        X_train_transformado,
        y_train,
    )

    print(
        "Modelo entrenado correctamente."
    )

    # ========================================================
    # 7. EVALUACIÓN SOBRE TEST
    # ========================================================

    print(
        "\n[7/8] Evaluando modelo sobre Test..."
    )

    # --------------------------------------------------------
    # Probabilidades
    # --------------------------------------------------------

    y_prob_test = modelo.predict_proba(
        X_test_transformado
    )[:, 1]

    # --------------------------------------------------------
    # Evaluación
    # --------------------------------------------------------

    resultados_test = evaluar_modelo(
        y_true=y_test,
        y_prob=y_prob_test,
    )

    # --------------------------------------------------------
    # Mostrar métricas
    # --------------------------------------------------------

    print("\nMÉTRICAS TEST")
    print("-" * 40)

    metricas_principales = [
        "auc",
        "accuracy",
        "precision",
        "recall",
        "f1",
    ]

    for metrica in metricas_principales:

        if metrica in resultados_test:

            valor = resultados_test[metrica]

            print(
                f"{metrica}: {valor:.6f}"
            )

    # --------------------------------------------------------
    # Evaluación con threshold = 0.50
    # --------------------------------------------------------

    from liquiditysense.evaluation import calcular_metricas

    metricas_threshold = calcular_metricas(
        y_true=y_test,
        y_prob=y_prob_test,
        umbral=THRESHOLD,
    )

    print(
        "\nMÉTRICAS TEST - THRESHOLD 0.50"
    )

    print("-" * 40)

    for metrica in [
        "accuracy",
        "precision",
        "recall",
        "f1",
    ]:

        if metrica in metricas_threshold:

            valor = metricas_threshold[
                metrica
            ]

            print(
                f"{metrica}: {valor:.6f}"
            )

    # ========================================================
    # 8. GUARDADO DE ARTEFACTOS
    # ========================================================

    print(
        "\n[8/8] Guardando artefactos..."
    )

    # --------------------------------------------------------
    # Modelo
    # --------------------------------------------------------

    joblib.dump(
        modelo,
        ARCHIVO_MODELO,
    )

    # --------------------------------------------------------
    # Preprocesador
    # --------------------------------------------------------

    joblib.dump(
        preprocesador,
        ARCHIVO_PREPROCESADOR,
    )

    # --------------------------------------------------------
    # Variables
    # --------------------------------------------------------

    joblib.dump(
        nombres_variables,
        ARCHIVO_VARIABLES,
    )

    print(
        f"Modelo:         {ARCHIVO_MODELO}"
    )

    print(
        f"Preprocesador:  {ARCHIVO_PREPROCESADOR}"
    )

    print(
        f"Variables:      {ARCHIVO_VARIABLES}"
    )

    # ========================================================
    # MLFLOW
    # ========================================================

    ruta_mlflow = (
        ROOT_DIR / "mlflow.db"
    )

    try:

        print(
            "\nRegistrando experimento en MLflow..."
        )

        configurar_mlflow(
            ruta_tracking=ruta_mlflow,
            nombre_experimento="LiquiditySense",
        )

        # ----------------------------------------------------
        # Parámetros
        # ----------------------------------------------------

        parametros = {
            "model": "RandomForestClassifier",
            "n_estimators": 400,
            "max_depth": 12,
            "min_samples_leaf": 30,
            "max_features": "sqrt",
            "class_weight": "balanced",
            "random_state": 42,
            "threshold": THRESHOLD,
            "n_features": len(nombres_variables),
            "train_rows": len(X_train),
            "validation_rows": len(X_valid),
            "test_rows": len(X_test),
        }

        # ----------------------------------------------------
        # Métricas
        # ----------------------------------------------------

        metricas = {
            "auc_test": resultados_test[
                "auc"
            ],
            "accuracy_test": metricas_threshold[
                "accuracy"
            ],
            "precision_test": metricas_threshold[
                "precision"
            ],
            "recall_test": metricas_threshold[
                "recall"
            ],
            "f1_test": metricas_threshold[
                "f1"
            ],
        }

        # ----------------------------------------------------
        # Tags
        # ----------------------------------------------------

        tags = {
            "project": "LiquiditySense",
            "objective": (
                "Predicción de deterioro de liquidez"
            ),
            "model": "Random Forest",
            "split": "Temporal",
            "dataset": (
                "Call Reports / Balance Sheets"
            ),
            "evaluation": "Test",
            "threshold": str(THRESHOLD),
        }

        # ----------------------------------------------------
        # Registro del experimento
        # ----------------------------------------------------

        with iniciar_run(
            nombre_run=(
                "modelo_final_random_forest"
            )
        ):

            registrar_parametros(
                parametros
            )

            registrar_metricas(
                metricas
            )

            registrar_tags(
                tags
            )

            registrar_artefactos(
                {
                    "model": ARCHIVO_MODELO,
                    "preprocessor": (
                        ARCHIVO_PREPROCESADOR
                    ),
                    "variables": ARCHIVO_VARIABLES,
                }
            )

        print(
            "Experimento registrado correctamente."
        )

    except Exception as error:

        print(
            "\nADVERTENCIA: no fue posible registrar "
            "el experimento en MLflow."
        )

        print(
            f"Detalle: {error}"
        )

    # ========================================================
    # FINALIZACIÓN
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "ENTRENAMIENTO FINALIZADO CORRECTAMENTE"
    )

    print(
        "=" * 70
    )


# ============================================================
# PUNTO DE ENTRADA
# ============================================================

if __name__ == "__main__":
    main()

