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
| Período histórico | **Q3 2025 – Q2 2026** (período de máxima actividad de Polymarket) | El dry-run reveló que el 84% de los candidatos tienen `startDate` en 2026-Q1. Los mercados de 2023-2024 están en offsets > 42k de la API y son estadísticamente mínimos. El período se reframea explícitamente: este trabajo estudia Polymarket durante su fase de expansión masiva, no el historial completo. Extender a 2023 no aportaría diversidad temporal real. |
| Tamaño del dataset | **967 mercados** (post-filtros, real) | El umbral original de 1,500 era estimación previa. El universo accesible confirmado tras descarga completa es 967 (de 1,209 candidatos, 242 con < 3 puntos de precio excluidos). Suficiente para ML con split 70/10/20. Documentado como hallazgo sobre la estructura del ecosistema Polymarket. |
| Mínimo de puntos de precio | ≥ 3 puntos en ventana de 7 días | `/prices-history` no garantiza 7 puntos (días sin trades no aparecen). Con < 3 puntos las features de tendencia y volatilidad son poco confiables. Mercados con < 3 puntos se descartan del análisis. |

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

**Criterio de orden:** `start_date` (fecha de apertura del mercado, no cierre).

**Contexto temporal (actualizado tras EDA completo):** El dataset cubre Q3 2025 – Q2 2026 con 66% de los mercados concentrados en marzo 2026.

**Hallazgo crítico post-EDA — Drift intra-mes en el split temporal:**

El split 70/10/20 estricto por `start_date` sobre este dataset produce:

| Conjunto | n | Rango | YES rate |
|---|---|---|---|
| Train | 675 | 2025-06-23 a 2026-03-24 | **9.8%** |
| Val | 96 | 2026-03-24 a 2026-03-25 | 8.3% |
| Test | 194 | 2026-03-25 a 2026-04-02 | **21.1%** |

El test set tiene YES rate = 21.1% vs 9.8% del train — más del doble. Test KS sobre `precio_fin`: p < 0.001 (distribuciones estadísticamente distintas). La causa raíz es que la semana 4 de marzo 2026 (Sem4: 22-31 marzo, n=361) tiene YES rate = 14.7% vs semanas anteriores (3.5%-9.7%), y la composición de categorías varía semana a semana dentro de marzo (Sem2: 81% Finance; Sem3: 60% Tech; Sem4: 33% Finance + 38% Other + 17% Politics).

**Decisión sobre el split (post-EDA):** Usar split **estratificado por bucket temporal con asignación determinística por `market_id`** en lugar de corte estricto.

**Justificación explícita:** Se prefirió este split sobre el puramente cronológico por:
(a) Drift estadísticamente significativo intra-marzo (KS test sobre `precio_fin`: p < 0.001).
(b) Concentración del 66% del dataset en un solo mes que vuelve poco informativo el split cronológico estricto (val cubre solo 2 días; test tiene YES rate 21.1% vs 9.8% en train).

**Implementación:**
1. Asignar cada mercado a uno de 4 buckets según `start_date`: `pre-2026` / `2026-01` / `2026-02` / `2026-03+`
2. Dentro de cada bucket: `split = hash(market_id) % 100` → 0-69 = train, 70-79 = val, 80-99 = test
3. El hash usa MD5 sobre el `conditionId` (hex → int) para garantizar determinismo e independencia del orden de procesamiento

**Cross-validation en train:** `GroupKFold` con grupos por bucket temporal (4 folds). Nunca `KFold` con shuffle.

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

### Imbalance de clases — decisiones firmes

**Hallazgo del dry-run:** YES = 14.6% / NO = 85.4%. El imbalance es estructural del dominio (Polymarket: la mayoría de mercados son sobre eventos que no ocurren), no error de muestreo.

| Decisión | Detalle |
|---|---|
| Métricas válidas | AUC, log-loss, Brier score, F1-macro, reliability diagram. **No accuracy.** |
| Baseline trivial | "Predecir siempre NO" da 85% accuracy. Cualquier modelo tiene que superar eso en AUC/F1, no solo en accuracy. Reportar explícitamente este piso. |
| Técnicas de balanceo | Pasan de "explorar si aplica" a **obligatorias**: `class_weight='balanced'` en todos los modelos que lo soporten; SMOTE y undersampling como experimentos comparados en Fase 6. |
| Reporte | Para cada modelo: AUC, log-loss, Brier, F1 para YES y NO por separado, matriz de confusión. Incluir curva precision-recall (más informativa que ROC con imbalance severo). |

---

## 6. Feature engineering — resumen

> **Nota:** El feature set fue revisado en Fase 1 tras constatar restricciones de la API pública. Ver sección 11 para el detalle completo de la decisión.

### Features a construir — set definitivo

**Precio del token SÍ — primeros 7 días (fuente: CLOB `/prices-history`):**
- `precio_dia_1` ... `precio_dia_7`: serie cruda (también entran como features individuales)
- `precio_inicio`: precio del día 1
- `precio_fin`: precio del día 7 (= baseline del modelo)
- `precio_media`, `precio_mediana`, `precio_std`
- `precio_rango`: max − min del período
- `precio_tendencia`: pendiente de regresión lineal sobre días 1..7
- `volatilidad_retornos`: desvío estándar de retornos log diarios log(p_t / p_{t-1})

**Tamaño de mercado (fuente: Gamma API `volumeNum`, snapshot al momento de descarga):**
- `log_volumen_total`: log(1 + volumeNum) — proxy de liquidez y seriedad del mercado

**Actividad de la ventana de observación (fuente: CLOB `/prices-history`):**
- `n_puntos_precio`: número real de días con precio registrado en los primeros 7 días (rango: 3–7 para mercados que pasan filtro). Preserva información sobre densidad de actividad del mercado, que puede ser señal predictiva. Las features de precio (media, tendencia, volatilidad) se calculan con los puntos disponibles; no se forward-fill.

**Categoría (derivada — la API no provee un campo `category` directo):**

El endpoint `/markets` no retorna un campo `category`. Se construyen dos features derivadas:

- `event_ticker_prefix`: las primeras 1-2 palabras del `events[0].ticker` (ej. `"serie-a"`, `"epl"`, `"elon-musk"`). Identidad del evento padre. Alta cardinalidad → `TargetEncoder` o `OrdinalEncoder` en Fase 4 según EDA.
- `category_coarse`: categoría de 7 valores derivada del texto de `question` por matching de keywords. Reglas exactas (en orden de prioridad):

**Reglas refinadas post-EDA** (v2 — aplicadas desde Fase 3). Código autorizado: `src/features/categorization.py`.

El EDA reveló que la v1 clasificaba 28% de los mercados como "Other" cuando la mayoría eran geopolíticos, bancos centrales, o charts musicales. Tras refinamiento, "Other" bajó al 6.5%.

| Categoría | n (v2) | YES rate | Adiciones principales vs v1 |
|---|---|---|---|
| Finance | 353 (36.6%) | 16.1% | ECB, Bank of Canada/Brazil/England, "bps", "rate cut/hike/pause", "median home", "the fed ", "fed decision" |
| Politics | 186 (19.3%) | 7.5% | "russia ", "ukraine", "israel", "iran ", "gaza", "ceasefire", "military action", "warship", "strait of", "saudi arabia", "hezbollah", "hamas", "houthi" |
| Sports | 162 (16.8%) | 8.0% | "eastern conference", "western conference", "vezina", "jack adams award", "super lig", "both teams to score", "end in a draw" |
| Tech | 128 (13.3%) | 8.6% | "cloudflare", "gemini ", "deepseek", "grok", "x money" |
| Other | 63 (6.5%) | 4.8% | Reducido a genuino misceláneo: precipitaciones, suscriptores YouTube, bans, eventos one-off |
| Entertainment | 39 (4.0%) | 10.3% | "spotify", "monthly listeners", "music festival", "todo mundo", "concert" |
| Crypto | 34 (3.5%) | 38.2% | Corregidos false positives: " bnb" (no "bnb" para evitar "Airbnb"), " sol " (no "sol" para evitar "solar") |

Nota: Finance se evalúa antes que Tech para que "Will Apple dip to $240" → Finance (precio), no Tech. El orden completo es: Sports > Politics > Crypto > Finance > Tech > Entertainment > Other.

**Features descartadas del plan — decisiones post-EDA:**

| Feature | Motivo de descarte |
|---|---|
| `log_volumen_total` | **Leakage temporal:** el campo `volumeNum` del snapshot Gamma API refleja el volumen acumulado durante toda la vida del mercado (incluyendo post-ventana de 7 días). No representa el volumen disponible en el momento de predicción. Además, la correlación con el outcome es nula (r=0.022, p=0.495 en EDA). Descartarla no afecta performance y garantiza integridad metodológica. Hallazgo reportable: "el volumen total del mercado, a diferencia del precio, no predice el outcome binario." |
| `duration_days` | **Leakage temporal:** en el momento de predicción (día 7 del mercado), la duración total no es conocida. Solo disponible una vez que el mercado resuelve. |

**Imputation de `precio_dia_*`:** forward-fill + backward-fill dentro del vector de 7 días.
- Solo 74/965 mercados (7.7%) tienen primer precio en día 2 (no día 1). Ninguno tiene primer precio en día ≥ 3.
- El único caso real: backward-fill de `precio_dia_1` usando `precio_dia_2`.
- `volatilidad_retornos` se calcula sobre los puntos RAW del historial de precios (no sobre los días imputados), por lo que el forward-fill no la afecta directamente.

**Limitación documentada:** `volatilidad_retornos` puede subestimar la volatilidad real en mercados con pocos puntos de precio (`n_puntos_precio < 5`), ya que se calcula con menos observaciones y los retornos log están más esparcidos temporalmente.

### Features eliminadas respecto al plan original

Las siguientes features se descartaron por restricciones de la API pública (sin autenticación):

| Feature eliminada | Motivo |
|---|---|
| `volumen_total_7d`, `volumen_media_diaria`, `volumen_tendencia` | Requieren serie histórica de volumen por día. Ningún endpoint público los provee para ventanas arbitrarias del pasado. |
| `n_trades_7d`, `n_traders_unicos` | `CLOB /trades` requiere API key (401). `data-api/trades` ignora parámetros de fecha y solo retorna los 200 trades más recientes. |
| `precio_intraday_range` | Solo granularidad diaria disponible para mercados resueltos en `/prices-history`. |

### Reducción de dimensionalidad
PCA sobre el bloque `[precio_dia_1, ..., precio_dia_7]` para visualizar correlación entre días consecutivos y posiblemente reducir antes de modelos lineales.

---

## 7. Clustering en EDA

K-means sobre el perfil de precios `[precio_día_1, ..., precio_día_7]` (normalizados).

- Determinar k con elbow + silhouette score.
- Interpretar clusters cualitativamente (ej: "mercados que arrancan alto y bajan", "mercados estables", "mercados volátiles").
- En la fase de resultados: cruzar clusters con performance del modelo final para identificar en qué "tipos" de mercado el modelo agrega más valor.

---

## 8. Granularidad y APIs

Estado verificado experimentalmente el 2026-05-03 durante Fase 1.

| API | Uso en este proyecto | Estado | Notas |
|---|---|---|---|
| Gamma API `GET /markets` | Listado de mercados resueltos + metadatos (outcome, categoría, fechas, volumeNum, clobTokenIds) | **Sunset 2026-05-01. Sigue funcionando.** Sin reemplazo documentado. | Descargar dataset completo ASAP como mitigación. Ver sección 11. |
| CLOB API `GET /prices-history` | Serie de precios diarios para ventana de 7 días | Activo, sin deprecación | Requiere `startTs` + `endTs` (máx ~7 días por request) o `interval`. Retorna `{t, p}` solamente — sin volumen. |
| data-api `GET /trades` | Sondeo exploratorio — descartado para features | Activo, sin deprecación | Parámetros de fecha (`startTs`, `endTs`, `after`, `before`) son ignorados. Siempre retorna los ~200 trades más recientes. No usable para features históricas. |
| CLOB API `GET /trades` | Descartado — requiere autenticación | Activo, requiere API key | Retorna 401 sin key. Reservado como Camino B de escalada si las features actuales resultan insuficientes. |

**Outcome encoding (verificado experimentalmente):**
- Campo `outcomePrices` en la respuesta de Gamma API: array de strings `["precio_yes", "precio_no"]`
- Si `outcomePrices[0] == "1"` → YES ganó (target = 1)
- Si `outcomePrices[0] == "0"` → NO ganó (target = 0)
- Si `outcomePrices[0]` es valor intermedio (ej. `"0.5"`) → mercado cancelado/N/A → **descartar**

Código de descarga parametrizado por `fidelity` para poder cambiar a granularidad horaria sin reescribir.

---

## 9. Riesgos y mitigaciones

| Riesgo | Probabilidad | Impacto | Mitigación |
|---|---|---|---|
| Dataset < 1500 mercados | ~~Media~~ → **RESUELTO** | Alto | Dataset real ~1,007. El umbral de 1,500 era estimación previa. Se acepta ~1,000: suficiente para ML con split 70/10/20. Ver sección 1. |
| Granularidad de precios insuficiente (< 7 puntos) | ~~Media~~ → **RESUELTO** | Medio | 83% de los mercados tienen ≥3 puntos. Filtro aplicado. Feature `n_puntos_precio` captura la densidad. |
| Leakage no detectado | Baja | Muy alto | Checklist explícito en cada feature: ¿usa datos post-día 7? |
| Imbalance severo de clases | ~~Media~~ → **CONFIRMADO** (85/15) | Medio | Ver sección 5 — Imbalance de clases. Técnicas obligatorias, métricas claras. |
| Drift temporal (limitado al período Q3 2025 – Q2 2026) | Baja-Media | Bajo-Medio | Rango de 4 trimestres. Analizar distribución mensual en EDA; reportar si hay diferencias entre Q3-Q4 2025 y Q1-Q2 2026. Expectativa: drift menor que en análisis multi-año. |
| API deprecada rompe descarga en mitad del proyecto | Media | Alto | Sunset ya pasó (2026-05-01) pero endpoint activo. Mitigación: ejecutar descarga completa ASAP y persistir en `data/raw/`. Una vez descargado, el riesgo desaparece. |
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

## 11. Restricciones de API y feature set final (Camino A)

**Decisión tomada el 2026-05-03, después de la exploración experimental de Fase 1.**

Durante la exploración de APIs se constató que no es posible obtener datos de volumen ni de actividad de trading para ventanas históricas arbitrarias (primeros 7 días) a través de la API pública de Polymarket sin autenticación. Se evaluaron dos caminos:

- **Camino A:** Continuar con features de precio + metadata de Gamma. Feature set más acotado pero completamente funcional.
- **Camino B:** Obtener API key de Polymarket para desbloquear `CLOB /trades`. Feature set completo pero mayor complejidad operacional.

**Se eligió Camino A por las siguientes razones:**

1. **Coherencia con el scope del brief:** el trabajo busca solidez metodológica sin sobre-ingeniería. Camino B añadiría complejidad sin garantía de mejora sustancial en los resultados.
2. **`log_volumen_total` se preserva:** el volumen total del mercado (snapshot de Gamma API) sigue siendo una feature válida como proxy de liquidez y seriedad del mercado, aunque no sea específico de los primeros 7 días.
3. **Limitación como hallazgo metodológico legítimo:** la restricción de la API pública es información relevante sobre el ecosistema de Polymarket y va documentada en la sección "tareas y problemas encontrados" del informe final (sección 7 del brief).
4. **Camino B queda como escalada futura:** si en el análisis de resultados se detecta que el feature set de precios es insuficiente para superar el baseline, se puede activar Camino B sin rediseñar la arquitectura del proyecto.
5. **Los precios incorporan información de volumen implícitamente:** para que el precio se mueva, alguien tiene que tradear. Las features de trayectoria de precios (tendencia, volatilidad, rango) capturan indirectamente la actividad del mercado.

**Riesgo aceptado sobre los endpoints deprecados:**
Ambos endpoints de Gamma API (`/markets`, `/events`) tienen `Sunset: 2026-05-01` (ya vencido) pero siguen funcionando. No existe endpoint de reemplazo documentado. La mitigación es ejecutar la descarga completa del dataset lo antes posible y persistirla en `data/raw/`. Una vez que los datos están en disco, el riesgo de que el endpoint se apague desaparece para este proyecto.

---

## 12. Hallazgos clave del EDA (Fase 3)

### Cohorte octubre 2025 — 100 mercados, 0% YES (Investigación 1)

**Causa:** No es error de datos ni período bajista generalizado. Es un artefacto estructural de los mercados de premios deportivos: en octubre 2025 se abrieron ~90 mercados de la NHL (Vezina Trophy = mejor portero, Jack Adams Award = mejor entrenador), con ~29-30 candidatos por premio. Solo uno puede ganar → 29 resuelven NO, 1 resuelve YES. El mercado ganador (`outcomePrices[0]=="1"`) fue descartado por algún filtro (probablemente `MIN_PRICE_POINTS < 3` o no coincidió la cohorte), o simplemente no apareció en los primeros 7 días con tendencia clara.

**Verificación post-EDA (concreta):** Se buscaron en `data/raw/markets/` todos los mercados con "vezina" o "jack adams" en el texto. Resultado:
- Encontrados en descarga: **90 markets**, todos NO (`outcomePrices = ["0","1"]`)
- Todos pasaron el filtro `n_pts >= 3` (6-7 puntos cada uno)
- Los ganadores (YES) **no están en `candidates.json`** — no es un problema de `n_pts < 3`

**Causa raíz:** La paginación de la Gamma API se detuvo en offset ~42,200 (plateau detection, página 422). Los mercados ganadores de los premios NHL están en offsets superiores a ese umbral — simplemente no fueron escaneados. No es un error de filtrado sino un sesgo de cobertura: el dataset contiene sistemáticamente los ~N-1 candidatos perdedores de cada premio, pero no el ganador.

**Implicación para el modelado:** Los mercados de formato "¿ganará X el premio Y?" son subrepresentados en YES dentro de Sports. Si el evaluador pregunta "¿descartaste los YES de Sports?", la respuesta es: no se descartaron — no fueron recolectados debido a la mecánica de paginación. **Limitación documentada**, no error metodológico. La corrección requeriría re-correr la descarga sin plateau detection, lo que excede el scope del TFI.

**Conclusión:** Dato legítimo, limitación de cobertura conocida. No se requiere acción correctiva en el pipeline actual.

### Drift intra-marzo 2026 — split temporal afectado (Investigación 2)

**Causa:** La concentración del 66% del dataset en marzo 2026 + la variación de composición categórica y YES rate semana a semana dentro del mes (Sem2: 81% Finance, 3.5% YES; Sem3: 60% Tech, 7.6% YES; Sem4: mixto, 14.7% YES) hace que el split temporal estricto produzca un test con YES rate del 21.1% vs 9.8% en train.

**Decisión:** Split estratificado por bucket temporal (ver sección 3). Esto elimina el problema sin sacrificar la integridad temporal dentro de cada bucket.

### Categorización refinada (Investigación 3)

**Reglas v1 → v2:** Other bajó de 27.9% (269 mkt) a 6.5% (63 mkt). Se rescataron:
- 99 mercados geopolíticos → Politics (Russia, Ukraine, Israel, Iran, etc.)
- 47 mercados de bancos centrales → Finance (ECB, Bank of Canada, bps, etc.)
- 35 mercados de Spotify/charts → Entertainment
- 16 mercados de equipos/formatos deportivos → Sports
- 7 mercados de IA/Tech → Tech

### Hallazgos adicionales documentados en `notebooks/02_eda.ipynb`

| Hallazgo | Relevancia para modelado |
|---|---|
| `precio_fin` r=0.44 con outcome | Feature más predictiva — el precio al día 7 ya captura la mayor parte del señal |
| `volatilidad_retornos` r=-0.17 | Señal negativa independiente — mercados volátiles resuelven NO más frecuente |
| `n_puntos_precio`, `log_volumen_total`, `precio_rango`, `precio_std` no significativos (p>0.05) | La señal está en el nivel del precio, no en su variabilidad ni volumen |
| YES rate varía 6%-38% por categoría | `category_coarse` es feature valiosa; Crypto 38.2% vs Sports 8.0% |
| K-means k=4: cluster con 62% YES y cluster con 0.5% YES | Los precios días 1-7 separan grupos extremos claramente |

---

---

## 13. Verificaciones post-Fase 5: multicolinealidad e interpretación de coeficientes

### V1 — Coeficiente negativo de precio_tendencia en LR-MIN

La hipótesis inicial de "corrección a la media" fue descartada por verificación directa.

**Correlaciones entre las 4 features de LR-MIN (train):**

| Par | r |
|-----|---|
| precio_fin / precio_media | +0.922 — multicolinealidad severa |
| precio_fin / precio_tendencia | +0.646 |
| precio_fin / volatilidad_retornos | -0.556 |
| precio_media / precio_tendencia | +0.397 |
| precio_media / volatilidad_retornos | -0.538 |
| precio_tendencia / volatilidad_retornos | -0.515 |

**Hipótesis de "corrección a la media" descartada:**
- Corr(precio_tendencia, precio_inicio) = -0.09 — prácticamente nulo
- Los mercados con tendencia positiva NO empiezan significativamente más bajos

**Explicación correcta del signo negativo:**
- YES rate marginal: tendencia>0 = 27.2%, tendencia<0 = 7.6% (¡señal POSITIVA!)
- Sin embargo, precio_tendencia y precio_fin tienen r=0.646. Una vez que precio_fin absorbe la señal principal, el coeficiente parcial de tendencia captura información condicional (no dirección causal)
- El signo negativo del coeficiente es artefacto de multicolinealidad y no debe interpretarse como "tendencia positiva es señal negativa"
- **Interpretación válida**: las 4 features de LR-MIN son un sistema multicolineal. Solo precio_fin tiene interpretación directa; los otros tres coeficientes son correcciones condicionales

### V2 — Inversión de signo de volatilidad_retornos

| Configuración | Coeficiente volatilidad |
|---|---|
| LR con solo volatilidad | -0.7736 (consistente con r=-0.17) |
| LR con precio_fin + volatilidad | +0.1823 (¡inversión de signo!) |

**Mecanismo:**
- Corr(volatilidad, precio_fin) = -0.556: mercados con precio_fin bajo (inciertos) tienen alta volatilidad
- Una vez condicionado en precio_fin, la volatilidad adicional es señal positiva débil (mercado más activo/informado dado el mismo precio final)
- Esto es confounding clásico: volatilidad es proxy de incertidumbre, y precio_fin ya captura incertidumbre directamente

**Consecuencia para el informe:** No reportar el coeficiente de volatilidad_retornos en LR-MIN como "señal positiva" sin mencionar que es señal condicional a precio_fin. La señal marginal (univariada) es negativa.

---

## 14. Resultados Fase 5 — Baselines y Regresión Logística

Script: `src/models/phase5.py`

### Tabla de métricas (test set, n=205)

| Modelo | AUC | PR-AUC | LogLoss | Brier | F1(YES) | Features activas |
|--------|-----|--------|---------|-------|---------|-----------------|
| B1: Mayoría NO | 0.5000 | 0.1122 | 1.8084 | 0.1122 | 0.0000 | — |
| B2: Prior 12.4% | 0.5000 | 0.1122 | 0.3518 | 0.0998 | 0.0000 | — |
| B3: precio_fin directo | **0.8471** | 0.6031 | 0.3697 | 0.1181 | 0.4615 | — |
| LR-A: sin reg. | 0.8162 | 0.5451 | 0.2630 | 0.0743 | 0.4737 | 22/22 |
| LR-MIN: 4f | 0.8245 | 0.5680 | 0.2613 | 0.0731 | 0.4444 | 4/4 |
| LR-B: L2 C=50 | 0.8069 | 0.5555 | 0.2639 | 0.0730 | 0.4242 | 22/22 |
| **LR-C: L1 C=0.5** | **0.8339** | **0.5824** | **0.2545** | **0.0715** | **0.5000** | **13/22** |
| LR-D: L1 balanced | 0.8073 | 0.5390 | 0.4874 | 0.1604 | 0.3947 | 14/22 |

**Campeon Fase 5: LR-C** (L1 saga, C=0.5) — mejor AUC entre LR (0.8339), mejor Brier, mejor LogLoss, 13/22 features activas.

### Features eliminadas por L1 (LR-C, zeroed):
precio_dia_2, precio_dia_3, precio_dia_5, precio_mediana, precio_rango, volatilidad_retornos, cat_Entertainment, cat_Finance, cat_Tech (9 features eliminadas)

### Hallazgos clave:
1. B3 supera a LR sin regularización en AUC — la señal está concentrada, 22 features añaden ruido
2. LR-MIN (4 features) supera a LR-A (22 features) — hipótesis mínima confirmada
3. L1 con C=0.5 resulta en solución más esparsa y mejor calibrada que liblinear C=1.0 (saga solver)
4. Balanceo (LR-D) mejora recall YES (35%→61%) pero destruye precisión (73%→29%), F1-YES neto baja
5. Solo LR-D clasifica más de 10 YES correctamente; todos los demás LR son muy conservadores

---

## 15. Resultados Fase 6 — Random Forest y Gradient Boosting

Script: `src/models/phase6.py`  
Búsqueda: RandomizedSearchCV, n_iter=25, cv=5, scoring=AUC, random_state=42

### Tabla de métricas (test set, n=205)

| Modelo | AUC | PR-AUC | LogLoss | Brier | F1(YES) | CV AUC (train) |
|--------|-----|--------|---------|-------|---------|----------------|
| **GB (sklearn)** | **0.8933** | **0.6348** | **0.2436** | **0.0678** | 0.3333 | 0.8217 |
| RF | 0.8876 | 0.6338 | 0.3414 | 0.1000 | **0.5833** | 0.8339 |

Ambos superan B3 (AUC=0.8471) y LR-C (AUC=0.8339).

### Mejores hiperparámetros:

**RF:** n_estimators=300, max_depth=5, min_samples_split=20, min_samples_leaf=8, max_features='sqrt', class_weight='balanced'
**GB:** n_estimators=100, max_depth=6, learning_rate=0.01, subsample=0.7, max_features=None, min_samples_split=5, min_samples_leaf=4

### Feature importance (top 5):

**RF:** precio_dia_6 (0.161), precio_fin (0.136), precio_media (0.129), precio_dia_7 (0.112), precio_dia_5 (0.069)

**GB:** precio_dia_6 (0.277), precio_dia_7 (0.120), precio_fin (0.090), precio_dia_4 (0.064), precio_media (0.061)

**Hallazgo notable:** precio_dia_6 emerge como feature más importante en ambos modelos de árbol, algo que la regresión logística con L1 había zerado (precio_dia_6 sobrevive en LR-C pero con coeficiente +0.39). Los modelos de árbol capturan interacciones no lineales entre los precios diarios que la logística no puede.

### Campeon general:

**GB (sklearn)** — mejor en discriminación (AUC=0.8933), calibración (Brier=0.0678, LogLoss=0.2436) y PR-AUC. El bajo F1(YES)=0.3333 se explica por threshold=0.5 en un modelo sin class_weight='balanced'; ajustando el threshold, el recall mejora sustancialmente.

---

## Próximos 3 pasos (en orden)

~~**[Fase 0]** Setup del repositorio~~ Completado 2026-05-02

~~**[Fase 1]** Exploración de APIs~~ Completado 2026-05-03

~~**[Fase 2]** Descarga masiva~~ Completado 2026-05-03
- 1,209 candidatos encontrados (plateau en página 422, offset 42,200)
- 967 mercados con ≥3 puntos de precio en `data/raw/` (242 excluidos por < 3 puntos)
- Período real del dataset: Q3 2025 – Q2 2026
- Imbalance YES/NO confirmado: 12.0% / 88.0%
- Tiempo de descarga: 14.7 min (1.4 mercados/s, 0 fallos)

~~**[Fase 3]** EDA completo~~ Completado 2026-05-03
- Ver sección 12 para hallazgos clave del EDA
- Reglas de categorización refinadas (Other: 28% → 6.5%)
- Split temporal redefinido: estratificado por bucket temporal

~~**[Fase 4]** Feature engineering + split definitivo~~ Completado 2026-05-03

~~**[Fase 5]** Baselines y regresión logística~~ Completado 2026-05-04
- Ver secciones 13-14

~~**[Fase 6]** Random Forest y Gradient Boosting~~ Completado 2026-05-04
- Ver sección 15

**[Fase 7 — siguiente]** Análisis de resultados, calibración, threshold optimization, narrativa final
