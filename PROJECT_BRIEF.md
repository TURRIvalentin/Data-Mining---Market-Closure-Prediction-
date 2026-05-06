# Trabajo Final Integrador — Especialización en Explotación de Datos y Descubrimiento del Conocimiento (UBA)

## Contexto

Soy estudiante de la Especialización y estoy haciendo el Trabajo Final Integrador. Necesito que me asistas durante todo el proyecto: desde la definición del problema hasta la entrega del informe final. Buscás cumplir bien los requisitos sin sobre-ingeniería: un trabajo sólido, prolijo, reproducible, con varios modelos comparados honestamente.

## Requisitos del trabajo (no negociables)

- Elaboración individual.
- Debe **integrar metodologías de varias materias del primer año**: estadística descriptiva e inferencial, análisis exploratorio, reducción de dimensionalidad, clustering, clasificación supervisada, validación cruzada, manejo de clases desbalanceadas.
- El informe final debe contener:
  1. Resumen ejecutivo
  2. Presentación del problema
  3. Solución propuesta (síntesis)
  4. Descripción del dataset
  5. Análisis exploratorio
  6. Pruebas y variantes que justifiquen la elección del modelo
  7. Descripción de tareas y problemas encontrados
  8. Exposición detallada de la solución
  9. Conclusiones
  10. Trabajo futuro
  11. Bibliografía y referencias
  12. Anexos
- Calidad de "informe profesional".

## Tema del trabajo

**Predicción temprana del outcome de mercados de predicción en Polymarket: clasificación binaria del resultado final de un mercado a partir de su comportamiento durante los primeros días de operación.**

> **Período de estudio (actualizado en Fase 2):** Q3 2025 – Q2 2026 (período de máxima actividad de Polymarket). El universo accesible de mercados binarios resueltos con ≥30 días de duración se concentra en este intervalo. Ver DECISIONES.md sección 1.

### Pregunta de investigación principal

¿Es posible predecir, con accuracy significativamente mejor que la probabilidad implícita inicial del mercado, el outcome final (SÍ/NO) de un mercado binario de Polymarket utilizando únicamente información de sus primeros N días de actividad (precio, volumen, volatilidad, número de traders, etc.)?

### Hipótesis iniciales a explorar

1. La probabilidad implícita inicial del mercado es un baseline difícil de superar (los mercados de predicción están razonablemente bien calibrados).
2. La volatilidad temprana del precio aporta información predictiva por encima del precio promedio.
3. El volumen total del mercado (proxy de liquidez) correlaciona con su predictibilidad: mercados de mayor volumen estarán mejor calibrados y serán más fáciles de predecir. *(Hipótesis original: "volumen y número de traders únicos en los primeros 7 días". Revisada en Fase 1 porque esos datos no están disponibles vía API pública sin autenticación — ver DECISIONES.md sección 11.)*
4. Distintas categorías (política, deportes, cripto) presentan diferente predictibilidad y requieren posiblemente modelos separados. *(Actualización Fase 2: no existe campo `category` en la API; la categoría se deriva heurísticamente del texto de la pregunta — ver DECISIONES.md sección 6.)*

### Restricciones de alcance (scope)

- **Solo mercados binarios** (sí/no), no multi-outcome.
- **Solo mercados ya resueltos** con outcome conocido.
- **Ventana de observación fija**: primeros N=7 días de actividad (a confirmar en EDA si N=7 es razonable o conviene N=14).
- **Duración mínima del mercado**: ≥30 días (para que los primeros 7 sean realmente tempranos).
- **Volumen mínimo**: a definir tras EDA (descartar mercados con liquidez insuficiente para que el precio sea señal).
- **Categorías**: incluir todas en el análisis, pero comparar performance por categoría.

### Métricas de éxito

- **Métrica principal**: ROC-AUC sobre el conjunto de test.
- **Baseline a superar**: predecir usando solo la probabilidad implícita al final del día N (ej: si día 7 el precio del SÍ es 0.65, predecir SÍ con prob 0.65).
- **Métricas secundarias**: accuracy, F1, log-loss, Brier score.
- **Análisis de calibración**: reliability diagram del modelo final.

## Stack y entorno

- Python 3.11+
- pandas, numpy, scikit-learn, statsmodels, scipy
- matplotlib, seaborn para visualización (plotly opcional para exploración interactiva)
- imbalanced-learn para técnicas de balanceo
- requests / httpx para consumir APIs de Polymarket
- Jupyter notebooks para exploración + scripts .py para código reusable
- Git desde el día 1
- Outputs reproducibles (random seeds fijos, requirements.txt pinneado)

## Estructura del proyecto

```
trabajo_final_polymarket/
├── README.md
├── PROJECT_BRIEF.md          # Este documento
├── PROGRESS.md               # Bitácora actualizada después de cada sesión
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/                  # Datos crudos de la API, NO modificar
│   ├── interim/              # Datos limpios intermedios
│   └── processed/            # Features finales para modelado
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modeling_baseline.ipynb
│   ├── 05_modeling_advanced.ipynb
│   └── 06_results_and_figures.ipynb
├── src/
│   ├── __init__.py
│   ├── data/                 # Scripts de descarga y limpieza
│   ├── features/             # Feature engineering
│   ├── models/               # Entrenamiento y evaluación
│   └── visualization/        # Funciones de plotting reutilizables
├── reports/
│   ├── figures/              # Figuras finales para el informe
│   └── informe_final.md      # Informe en markdown (luego export a PDF/docx)
└── tests/                    # Tests mínimos de funciones críticas
```

## Cómo quiero que trabajes conmigo

1. **No te apures a programar.** Antes de cada fase, discutimos el plan. Vos proponés, yo confirmo.
2. **Justificá técnicamente cada decisión.** Si elegís random forest sobre XGBoost, decime por qué. Si descartás una feature, explicame.
3. **Pensá en el informe final desde el día 1.** Cada análisis y cada figura tiene que poder ir al informe. Guardá figuras en `reports/figures/` con nombres descriptivos.
4. **Sé crítico con los resultados.** Si un modelo da AUC 0.95 en datos financieros, sospechá leakage antes de festejar. Validá que las features no contengan información del futuro.
5. **Integrá técnicas de varias materias** explícitamente: estadística descriptiva en EDA, ACP en feature engineering, clustering exploratorio si aporta, clasificación con varios algoritmos, validación cruzada (con respeto al orden temporal si aplica), técnicas de balanceo si las clases lo requieren.
6. **Documentá supuestos y limitaciones** a medida que aparecen, no al final.
7. **Reproducibilidad ante todo:** seeds fijos en todo lugar donde haya aleatoriedad, requirements pinneados, scripts ejecutables de punta a punta.
8. **Manejá los datos con cuidado:** raw es sagrado, no se modifica. Cada transformación queda en código, no en celdas de Jupyter modificadas a mano.

## Plan de trabajo en fases

Al terminar cada fase me hacés un resumen de qué se hizo, qué se encontró, y cuál es el siguiente paso. Actualizás `PROGRESS.md`.

### Fase 0 — Setup (1-2 horas)
- Inicializar repositorio git
- Crear estructura de carpetas
- Setup de entorno (venv + requirements.txt inicial)
- README inicial con el plan
- `.gitignore` apropiado (excluir `data/raw/` del repo si pesa mucho, datos sensibles, etc.)

### Fase 1 — Exploración de la API y diseño del dataset (4-6 horas)
- Investigar Gamma API y CLOB API de Polymarket
- Identificar endpoints relevantes: listado de mercados, detalle de mercado, histórico de precios, trades
- Hacer requests exploratorios pequeños para entender la estructura de los datos
- Decidir granularidad (precio diario? horario?) y horizonte (¿desde qué fecha?)
- Documentar limitaciones de la API (rate limits, paginación, datos faltantes)

### Fase 2 — Recolección de datos (4-8 horas, mucho de espera por rate limits)
- Implementar scripts de descarga modulares y reanudables
- Persistir datos crudos en `data/raw/` con timestamp de descarga
- Manejar rate limits con backoff exponencial
- Documentar el dataset resultante: cantidad de mercados, periodo cubierto, variables disponibles, distribución por categoría
- Validar integridad: no debe haber duplicados, los IDs deben ser consistentes entre tablas

### Fase 3 — EDA (6-8 horas)
- Análisis univariado de todas las variables (distribuciones, missing, outliers)
- Análisis bivariado: relación entre features y outcome
- Análisis temporal: ¿hay drift? ¿cambia la distribución de mercados con el tiempo?
- Análisis por categoría: ¿son comparables?
- Visualizaciones que vayan directo al informe (no descartables)
- Lista de "hallazgos sorprendentes" o decisiones de scope que se ajustan según lo encontrado

### Fase 4 — Limpieza y feature engineering (6-8 horas)
- Tratamiento documentado de nulos y outliers (con justificación, no defaults)
- Construcción de features de los primeros N días:
  - Estadísticas de precio: media, mediana, desvío, rango, último precio del día N
  - Estadísticas de volumen: total, daily mean, tendencia
  - Volatilidad: desvío de retornos diarios, rango intra-día si hay datos
  - Actividad: número de trades, traders únicos
  - Features categóricas: tipo de mercado, encoders apropiados
- Si aplica: ACP sobre el bloque de features numéricas correlacionadas (precios día 1...7) para reducir dimensionalidad
- Split train/test con criterio temporal estricto: el test set son los mercados resueltos más recientes. Nunca shuffle aleatorio sobre series temporales sin pensarlo.
- Validar AUSENCIA DE LEAKAGE: ninguna feature puede usar información posterior al día N.

### Fase 5 — Modelado baseline (3-4 horas)
- Baseline 0: predecir siempre la clase mayoritaria (piso absoluto).
- Baseline 1: predecir según probabilidad implícita al final del día N (este es el baseline relevante: ¿el modelo agrega valor sobre lo que el mercado ya dice?).
- Modelo simple: regresión logística con regularización, hiperparámetros por CV.
- Comparación contra los baselines en AUC, log-loss, Brier score.

### Fase 6 — Modelado avanzado (8-10 horas)
- Random Forest con hyperparameter tuning (RandomizedSearchCV)
- Gradient Boosting (sklearn GBM o XGBoost/LightGBM si tiene sentido)
- KNN como referencia (más para integración con materias del programa que por performance esperada)
- Tratamiento explícito de clases desbalanceadas si aplica:
  - class_weight='balanced'
  - undersampling, oversampling, SMOTE
  - Comparación honesta de cada estrategia
- Comparación final con curvas ROC superpuestas, matriz de confusión, calibración (reliability diagram + Brier score)
- Ensamble simple (voto promedio de probabilidades) si aporta

### Fase 7 — Análisis de resultados (4-6 horas)
- Interpretabilidad: feature importance del mejor modelo. SHAP si el modelo lo amerita.
- Análisis de errores: ¿en qué tipos de mercados falla? ¿categorías? ¿rangos de precio inicial?
- Validación de hipótesis iniciales: ¿cuáles se confirmaron, cuáles no?
- Análisis de calibración del modelo final.
- Hallazgos accionables: ¿en qué condiciones el modelo bate al mercado?

### Fase 8 — Informe final (10-15 horas)
- Redacción siguiendo la estructura requerida.
- Figuras pulidas con títulos, ejes y leyendas en español, formato consistente.
- Tablas formateadas (no screenshots de Jupyter).
- Resumen ejecutivo de 1 página.
- Trabajo futuro **concreto** (no genérico): qué features faltaron, qué modelos no se probaron y por qué podrían ayudar, qué validaciones adicionales corresponden.
- Bibliografía con citas en formato académico (papers de mercados de predicción, calibración, ML aplicado a finanzas).
- Anexos con resultados detallados que no entran en el cuerpo principal.

## Para empezar AHORA

Antes de tocar código, quiero que:

1. Me hagas 5-10 preguntas críticas sobre el alcance que detecten ambigüedades o decisiones que conviene cerrar antes. Cosas que un data scientist senior preguntaría para no perder el tiempo después.
2. Investigues (puede ser con web search si tenés acceso, o pidiéndome que lo haga yo) qué APIs concretas de Polymarket conviene usar y qué endpoints son relevantes. Necesito una lista de 3-5 endpoints específicos con qué devuelven.
3. Identifiques los principales **riesgos** del proyecto:
   - Datos no disponibles o limitados
   - Leakage temporal
   - Imbalance extremo entre clases
   - Distribución no estacionaria (drift)
   - Volumen insuficiente de mercados que cumplan los filtros
4. Estimes esfuerzo en horas para cada fase y total, para poder planificar.

NO empieces a programar hasta que hayamos discutido los puntos anteriores y yo te dé luz verde explícita.

Cuando arranquemos a programar, seguí siempre el patrón: explicar qué vas a hacer → escribir el código → mostrar el output → interpretar el resultado → siguiente paso. Después de cada paso esperás mi confirmación.

¿Listo? Empezá con las preguntas críticas y la investigación de la API.