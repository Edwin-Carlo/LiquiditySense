# LiquiditySense

## Predicción anticipada del riesgo de liquidez bancario mediante Machine Learning

---

## 1. Descripción del proyecto

**LiquiditySense** es un proyecto de Machine Learning orientado al análisis y
predicción anticipada del deterioro del riesgo de liquidez en entidades
bancarias.

El proyecto busca identificar señales tempranas de deterioro de la posición
de liquidez mediante el análisis de información financiera histórica de
entidades bancarias y la aplicación de técnicas de análisis exploratorio,
ingeniería de variables y Machine Learning.

La propuesta busca pasar de un análisis principalmente descriptivo del riesgo
de liquidez hacia un enfoque predictivo que permita identificar patrones
previos asociados a un posible deterioro.

---

## 2. Problema

El riesgo de liquidez representa la posibilidad de que una entidad financiera
no pueda cumplir oportunamente con sus obligaciones debido a una insuficiencia
de recursos líquidos o a dificultades para obtener financiamiento.

La identificación temprana de señales de deterioro constituye un elemento
importante para la gestión del riesgo financiero.

Sin embargo, el análisis tradicional de indicadores financieros puede
concentrarse en observar el comportamiento histórico una vez que el deterioro
ya se ha producido.

Por ello, **LiquiditySense** plantea desarrollar un enfoque predictivo que
permita identificar anticipadamente condiciones financieras asociadas con un
mayor riesgo de deterioro de liquidez.

---

## 3. Objetivo general

Desarrollar un modelo de Machine Learning capaz de identificar
anticipadamente señales de deterioro del riesgo de liquidez en entidades
bancarias utilizando información financiera histórica.

---

## 4. Objetivos específicos

- Analizar información financiera histórica de entidades bancarias.
- Identificar las principales variables relacionadas con la posición de
  liquidez.
- Reducir y optimizar una base de datos financiera de gran volumen.
- Construir indicadores relevantes para el análisis del riesgo de liquidez.
- Realizar un análisis exploratorio de los datos.
- Definir una variable objetivo asociada al deterioro de liquidez.
- Desarrollar modelos de Machine Learning para la predicción.
- Comparar diferentes algoritmos y estrategias de modelamiento.
- Evaluar el desempeño predictivo mediante métricas apropiadas.
- Interpretar los principales factores asociados a las predicciones.
- Registrar y comparar experimentos mediante MLflow.
- Desarrollar una visualización de resultados mediante un dashboard.

---

## 5. Hipótesis

La información financiera histórica de las entidades bancarias contiene
patrones que permiten identificar anticipadamente condiciones asociadas al
deterioro del riesgo de liquidez.

Se plantea que un modelo de Machine Learning puede capturar relaciones no
lineales y combinaciones de variables que complementen el análisis
tradicional de indicadores financieros.

---

## 6. Datos

La fuente principal del proyecto corresponde a información histórica de
reportes financieros de entidades bancarias.

El dataset original contiene información trimestral de múltiples entidades
bancarias y diferentes períodos históricos.

La base original presenta un volumen considerable de información, por lo que
se realizará un proceso de selección y reducción de variables antes de la
etapa de modelamiento.

El objetivo de esta reducción será conservar la información relevante para
el análisis de liquidez y disminuir el volumen de datos innecesarios para el
modelo.

### Variables consideradas inicialmente

Entre las variables seleccionadas para el análisis se encuentran:

- `id_rssd` — Identificador de la entidad
- `date` — Fecha del reporte
- `assets` — Activos totales
- `cash` — Efectivo
- `securities` — Valores e inversiones
- `deposits` — Depósitos
- `demand_deposits` — Depósitos a la vista
- `time_deposits` — Depósitos a plazo
- `brokered_dep` — Depósitos intermediados
- `insured_deposits` — Depósitos asegurados
- `ln_tot` — Cartera total
- `ln_tot_gross` — Cartera bruta
- `ln_re` — Cartera inmobiliaria
- `ln_ci` — Cartera comercial e industrial
- `ln_cons` — Cartera de consumo
- `ln_agr` — Cartera agrícola
- `ln_rre` — Cartera inmobiliaria residencial
- `npl_tot` — Cartera deteriorada
- `liab_tot` — Pasivos totales
- `equity` — Patrimonio
- `subdebt` — Deuda subordinada

> Las variables definitivas serán determinadas después del análisis
> exploratorio y de la evaluación de su relevancia para el problema.

---

## 7. Alcance temporal preliminar

Para el desarrollo inicial del proyecto se utilizará información comprendida
entre los años **2000 y 2025**, sujeto a la disponibilidad y calidad de los
datos.

Este período permite trabajar con una cantidad suficiente de observaciones
históricas y mantener una ventana temporal consistente para el análisis y
modelamiento.

---

## 8. Metodología

El proyecto seguirá un flujo de trabajo estructurado en las siguientes
etapas:

```text
Datos financieros originales
          │
          ▼
Selección de variables relevantes
          │
          ▼
Filtrado temporal
          │
          ▼
Limpieza y transformación
          │
          ▼
Construcción de indicadores de liquidez
          │
          ▼
Dataset final
          │
          ▼
Análisis exploratorio (EDA)
          │
          ▼
Definición de variable objetivo
          │
          ▼
Preparación para Machine Learning
          │
          ▼
Entrenamiento de modelos
          │
          ▼
Evaluación e interpretación
          │
          ▼
MLflow + Dashboard