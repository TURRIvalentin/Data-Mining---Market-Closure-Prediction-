# DECISIONES DE DISEÑO — Trabajo Final Integrador

> Documento vivo. Cada decisión de scope, metodológica o técnica relevante
> se registra aquí con su justificación. Es la primera defensa ante preguntas
> del evaluador sobre "¿por qué hiciste X?".

---

## 1. Definición del problema

| Decisión | Valor elegido | Justificación |
|---|---|---|
| Tipo de mercado | Solo binarios (SÍ/NO) | Outcome bien definido, comparación directa con precio implícito |
| Estado del mercado | Solo resueltos (outcome conocido) | Variable objetivo disponible |
| Duración mínima | ≥ 30 días | Garantiza que los primeros 7 días sean realmente "tempranos" |
| Ventana de observación | N = 7 días | A confirmar en EDA; N = 14 queda como análisis de sensibilidad |
| Mercados cancelados/N/A | Descartar | Outcome ambiguo; documentar cantidad descartada y si correlaciona con categoría o volumen |
| Período histórico | 2024-01-01 hasta hoy | Volumen y calibración de Polymarket más consistentes desde 2024; si quedan < 1500 mercados tras filtros, extender a 2023-01-01 |

---

## 2. Decisión crítica sobre leakage (Interpretación A)

**El modelo recibe `precio_día_1` ... `precio_día_7` como features, incluyendo el precio del último día.**

El baseline relevante es el **predictor simple = `precio_día_7`** (lo que el mercado ya dice al final de la ventana de observación).

### Justificación

Esto no constituye leakage porque:
1. Toda la información del modelo corresponde a los primeros N días, que son anteriores a la resolución del mercado.
2. El precio del día 7 está disponible al momento de hacer la predicción (no es información futura respecto a la pregunta de investigación).
3. El modelo agrega valor si aprende combinaciones no lineales de la trayectoria de precios que superen a observar solo el valor final.

La pregunta de investigación es: *"¿puede un modelo aprender patrones en la trayectoria de los primeros 7 días que sean más predictivos que el simple precio implícito del día 7?"*

### Lo que SÍ sería leakage (y está explícitamente prohibido)
- Cualquier feature construida con datos posteriores al día 7.
- El precio de resolución final o cualquier dato del período post-ventana.
- Features de volumen o actividad de días 8 en adelante.

---

## 3. Split temporal train / validation / test

| Conjunto | Proporción | Criterio |
|---|---|---|
| Train | 70% | Mercados resueltos más antiguos |
| Validation | 10% | Usada para hyperparameter tuning; no shuffle |
| Test | 20% | Mercados resueltos más recientes; **se toca una sola vez al final** |

**Criterio de orden:** fecha de resolución del mercado (no fecha de creación).

**Cross-validation en train+validation:** usar `TimeSeriesSplit` de scikit-learn (o `GroupKFold` con grupos por mes de resolución) para respetar el orden temporal. Nunca `KFold` con shuffle sobre datos temporales.

---

## 4. Métricas

| Métrica | Rol |
|---|---|
| ROC-AUC | Métrica principal de comparación entre modelos |
| Log-loss | Evalúa calibración de probabilidades |
| Brier score | Combina discriminación y calibración |
| Accuracy / F1 | Métricas secundarias; menos relevantes si hay desbalance |
| Reliability diagram | Calibración del modelo final |

**Baseline a superar:** AUC del predictor `precio_día_7` sobre el test set.

---

## 5. Modelos a entrenar (en orden)

1. **Baseline 0:** predecir siempre clase mayoritaria (piso absoluto)
2. **Baseline 1:** `precio_día_7` como predictor único (umbral 0.5)
3. Regresión logística con regularización L2 (CV sobre `C`)
4. Random Forest (`RandomizedSearchCV` sobre n_estimators, max_depth, min_samples_leaf)
5. Gradient Boosting (sklearn GBM o LightGBM)
6. KNN (integración metodológica con materias del programa)
7. Ensamble por promedio de probabilidades (si aporta sobre el mejor individual)

---

## 6. Feature engineering — resumen

### Features a construir sobre los primeros N=7 días

**Precio (del token SÍ):**
- `precio_inicio`: precio del día 1
- `precio_fin`: precio del día 7 (= baseline)
- `precio_media`, `precio_mediana`, `precio_std`
- `precio_rango`: max - min del período
- `precio_tendencia`: pendiente de regresión lineal días 1..7

**Volumen:**
- `volumen_total_7d`
- `volumen_media_diaria`
- `volumen_tendencia`: pendiente lineal del volumen diario

**Actividad (de `/trades`):**
- `n_trades_7d`: número de trades
- `n_traders_unicos`: estimación si la API lo permite

**Volatilidad:**
- `volatilidad_retornos`: desvío estándar de retornos diarios (log)
- `precio_intraday_range` (si disponible con granularidad < 24h)

**Categoría:**
- Encoding: `OrdinalEncoder` o `TargetEncoder` según cardinalidad

### Reducción de dimensionalidad
PCA sobre el bloque de precios diarios (`precio_día_1` ... `precio_día_7`) para visualizar correlación y posiblemente reducir antes de modelos lineales.

---

## 7. Clustering en EDA

K-means sobre el perfil de precios `[precio_día_1, ..., precio_día_7]` (normalizados).

- Determinar k con elbow + silhouette score.
- Interpretar clusters cualitativamente (ej: "mercados que arrancan alto y bajan", "mercados estables", "mercados volátiles").
- En la fase de resultados: cruzar clusters con performance del modelo final para identificar en qué "tipos" de mercado el modelo agrega más valor.

---

## 8. Granularidad y APIs

| API | Uso | Advertencia |
|---|---|---|
| Gamma API `/markets` | Listado y metadatos de mercados | **Header `Deprecation: true`** — investigar endpoint reemplazante en Fase 1 antes de descarga masiva |
| CLOB API `/prices-history` | Histórico de precios diarios (fidelity=1440) | Solo granularidad ≥12h para mercados resueltos |
| CLOB API `/trades` | Volumen y actividad por mercado | Sin limitación de granularidad conocida |

Código de descarga parametrizado por `fidelity` para poder cambiar a granularidad horaria sin reescribir.

---

## 9. Riesgos y mitigaciones

| Riesgo | Probabilidad | Impacto | Mitigación |
|---|---|---|---|
| Dataset < 1500 mercados tras filtros | Media | Alto | Extender período a 2023-01-01 |
| Granularidad de precios insuficiente (7 puntos) | Baja | Medio | Complementar con features de `/trades`; dejar como trabajo futuro la granularidad horaria |
| Leakage no detectado | Baja | Muy alto | Checklist explícito en cada feature: ¿usa datos post-día 7? |
| Imbalance severo de clases | Media | Medio | EDA determina; aplicar `class_weight='balanced'`, SMOTE, comparar estrategias |
| Drift temporal entre 2024 y 2025 | Media | Medio | Análisis de distribución por trimestre en EDA; reportar si es significativo |
| API deprecada rompe descarga | Media | Alto | Investigar endpoints actuales en Fase 1 antes de escribir el script de descarga masiva |
| AUC irrazonablemente alto (> 0.85) | Baja | Alto | Sospechar leakage; auditar features una a una |

---

## 10. Estimación de esfuerzo

| Fase | Descripción | Horas estimadas |
|---|---|---|
| 0 | Setup: git, estructura, venv, README | 1–2h |
| 1 | Exploración API, endpoints definitivos, requests manuales | 3–4h |
| 2 | Scripts de descarga reanudables, validación de integridad | 5–8h |
| 3 | EDA completo con figuras para el informe | 7–9h |
| 4 | Limpieza, feature engineering, PCA, split temporal | 6–8h |
| 5 | Baselines + regresión logística | 3–4h |
| 6 | RF, GBM, KNN, balanceo, curvas ROC, calibración | 8–12h |
| 7 | Interpretabilidad, análisis de errores, validación hipótesis | 4–6h |
| 8 | Informe final (redacción, figuras pulidas, bibliografía) | 12–16h |
| **Total** | | **~50–70h** |

El rango depende principalmente del tiempo de descarga (rate limits), del tamaño del dataset, y de la profundidad del informe.

---

## Próximos 3 pasos (en orden)

1. **[Fase 0]** Setup del repositorio: estructura de carpetas, venv, requirements.txt inicial, .gitignore, README con el plan. *(~1-2h)*

2. **[Fase 1 — primer paso]** Investigar el endpoint reemplazante del `/markets` deprecado de Gamma API. Hacer requests manuales exploratorios: listar 20 mercados resueltos del período, inspeccionar estructura de respuesta, identificar los campos necesarios (condition_id, tokenIds, outcome, fechas, categoría, volumen). *(~1h)*

3. **[Fase 1 — segundo paso]** Con un mercado de ejemplo, probar `/prices-history` y `/trades` y documentar exactamente qué devuelven: campos, tipos, valores nulos, rate limits reales observados. *(~1h)*

> **LUZ VERDE REQUERIDA** antes de ejecutar cualquiera de estos pasos.
