# CONTEXT.md — Polymarket ML Predictor

> Generado: 2026-07-16 (~10:20, hora local de la máquina de generación; los commits del repo usan offset `-03:00`, no verificado que coincidan exactamente).
> Reemplaza una versión previa de este mismo archivo (generada 2026-07-04), que nunca estuvo versionada. Este documento sí está versionado desde ahora — ver [Sección 2.4](#24-documentos-preexistentes-y-su-estado).
> Objetivo: que cualquier persona o IA sin contexto previo pueda entender el proyecto completo leyendo solo este archivo y el código.

Repo remoto: `https://github.com/TURRIvalentin/Data-Mining---Market-Closure-Prediction-.git`, branch `master`.

---

## 1. Propósito del proyecto

**Qué hace la app en una frase:** predice si un mercado binario (SÍ/NO) de [Polymarket](https://polymarket.com) va a resolver en SÍ, usando únicamente la trayectoria de precio del token "YES" durante sus primeros 7 días de actividad, y expone esas predicciones en un dashboard Streamlit público.

**Qué problema resuelve:** es el Trabajo Final Integrador (TFI) de la Especialización en Explotación de Datos y Descubrimiento del Conocimiento (UBA). La pregunta de investigación central: *¿puede un modelo de ML, mirando solo la primera semana de la trayectoria de precio de un mercado de predicción, anticipar mejor el resultado final que simplemente leer el precio implícito al cierre de esa semana?* En Polymarket, cada mercado binario tiene un token "YES" cuyo precio (entre 0 y 1) es la probabilidad implícita de que el evento ocurra según el mercado. El proyecto contrasta un modelo entrenado sobre la trayectoria completa de 7 días contra el baseline trivial "el precio del día 7 tal cual es la predicción".

El repositorio contiene dos piezas conceptualmente distintas que conviven en el mismo directorio de trabajo:

1. **El TFI académico** (la parte principal, versionada desde el primer commit): pipeline de ciencia de datos completo — descarga de mercados históricos ya resueltos, EDA, feature engineering, modelado supervisado (regresión logística, Random Forest, Gradient Boosting) y un dashboard Streamlit de scoring de mercados abiertos como demo pública del modelo entrenado.
2. **`polymarket-bot/`** — un subproyecto **no versionado** (excluido explícitamente en `.gitignore`) que vive en el mismo directorio pero es conceptualmente independiente: un bot de paper-trading/backtesting para mercados Polymarket de plazo muy corto ("¿BTC estará arriba/abajo de $X?"), que no usa los modelos de este TFI. No fue re-auditado en profundidad para este documento (ver [Sección 3.1](#31-árbol-completo-con-descripciones), nota al pie).

**Estado actual (verificado contra el código y `git log`, no contra documentación vieja):**

| Componente | Estado |
|---|---|
| Descarga de dataset histórico (`src/data/download.py`) | Completo, ejecutado. Datos crudos NO están en el repo (`data/raw/` gitignored) pero sí existen localmente en esta máquina. |
| Feature engineering + split (`src/data/make_dataset.py`) | Completo. Outputs versionados en `data/processed/`. |
| Modelado (Fases 5-6: LR, RF, GB) | Completo. `models/best_lr.pkl` y `models/best_tree_model.pkl` versionados y en uso activo. |
| Análisis de resultados (Fase 7: threshold, calibración) | Completo. `reports/fase7_analysis.json` versionado. |
| Informe final académico (Fase 8) | **No existe en el repo.** Se escribió, corrigió dos veces y se convirtió a Word, y luego **se borró** (commits `0facb83`, `6a57a65`). Ver [Sección 13](#13-historial-de-decisiones-importantes). Si la entrega académica ya se realizó, el documento vive fuera de este repositorio. |
| Dashboard Streamlit (`dashboard.py`) + scoring (`score_markets.py`) | Completo y en uso. Rediseñado hoy (2026-07-16, commit `4492036`). |
| Tests automatizados | **No existen.** `tests/__init__.py` está vacío, sin ningún `test_*.py`. |
| CI/CD | No hay ningún workflow (`no existe carpeta .github/`). |
| Deploy | Streamlit Community Cloud, según badge en `README.md` (URL no verificada activamente en esta sesión — ver [Sección 9](#9-deployment)). |
| `PROGRESS.md` (bitácora) | Abandonada tras la Sesión 1 (2026-05-02), pese a que tanto `PROJECT_BRIEF.md` como el propio encabezado de `PROGRESS.md` piden actualizarla al final de cada sesión. |

---

## 2. Arquitectura general

El proyecto **no es una aplicación tradicional** (no hay backend/frontend separados, no hay base de datos). Es un **pipeline de datos batch (ejecutado una vez, offline, para producir dataset y modelos) + un dashboard de solo lectura/scoring on-demand** (ejecutado repetidamente, online), típico de un proyecto de ciencia de datos aplicada llevado a producto.

### 2.1 Diagrama de flujo

**Pipeline offline (investigación — se corre una sola vez para producir el dataset y los modelos):**

```
┌──────────────────────────────────────────────────────────────────────────┐
│ FASE 2 — Descarga (src/data/download.py)                                 │
│   Gamma API GET /markets (paginado, orden desc, plateau detection)       │
│     → data/raw/candidates.json (mercados que pasan passes_filters())     │
│   CLOB API GET /prices-history (por mercado, ventana de 7 días)          │
│     → data/raw/markets/{conditionId}.json  (metadata)                    │
│     → data/raw/prices/{conditionId}.json   (serie de precios)            │
└──────────────────────────────┬─────────────────────────────────────────-─┘
                                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ FASE 3 — EDA (notebooks/02_eda.ipynb)                                    │
│   Estadística descriptiva, correlaciones, clustering K-means,            │
│   detección de drift temporal, refinamiento de categorización v1 → v2    │
└──────────────────────────────┬─────────────────────────────────────────-─┘
                                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ FASE 4 — Feature engineering + split (src/data/make_dataset.py)          │
│   Lee data/raw/{markets,prices}/*.json                                   │
│   → construye 22 features (7 precios diarios + 8 agregados de precio     │
│     + 1 de actividad + 6 dummies de categoría) + target "outcome"        │
│   → split estratificado por bucket temporal (hash MD5 determinístico)    │
│   → StandardScaler fiteado en train                                      │
│   → data/processed/{train,val,test}.parquet + scaler.pkl +               │
│     feature_columns.json + split_stats.json                              │
└──────────────────────────────┬─────────────────────────────────────────-─┘
                                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ FASE 5 — Baselines + Regresión Logística (src/models/phase5.py)          │
│ FASE 6 — Random Forest + Gradient Boosting (src/models/phase6.py)        │
│   → models/best_lr.pkl, models/best_tree_model.pkl                       │
│   → reports/fase5_*.{md,json}, reports/fase6_*.{md,json}                 │
└──────────────────────────────┬─────────────────────────────────────────-─┘
                                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ FASE 7 — Análisis de resultados (src/models/phase7.py)                   │
│   Threshold optimization, calibración (ECE/MCE), feature importance,     │
│   análisis de errores por categoría                                      │
│   → reports/fase7_analysis.json, reports/figures/*.png                   │
└────────────────────────────────────────────────────────────────────────-─┘
```

**Sistema en producción (dashboard, uso interactivo/online):**

```
Usuario final
   │  abre el dashboard Streamlit (local o deployado)
   ▼
dashboard.py (Streamlit)
   │
   ├──(botón "🔄 Actualizar scores")──▶ subprocess: python score_markets.py --top 200
   │                                       │
   │                                       ├──▶ Gamma API GET /markets (active=true, closed=false)
   │                                       ├──▶ filtra: binario, con clobTokenIds, abierto ≥ 7 días
   │                                       ├──▶ CLOB API GET /prices-history (primeros 7 días)
   │                                       ├──▶ build_features() — MISMA lógica que make_dataset.py
   │                                       │      (duplicada, no importada — ver §11)
   │                                       ├──▶ escala con data/processed/scaler.pkl (fit en Fase 4)
   │                                       ├──▶ predict_proba() con best_tree_model.pkl (GB, champion)
   │                                       │                     y best_lr.pkl (LR-C)
   │                                       └──▶ escribe scoring_output.csv (gitignored)
   │
   ├──(lee, @st.cache_data ttl=300s)──▶ scoring_output.csv
   │
   └──(al seleccionar un mercado)──▶ fetch_price_history(condition_id)  [@st.cache_data ttl=600s]
                                          │
                                          ├──▶ Gamma API GET /markets?conditionId=... (clobTokenIds)
                                          └──▶ CLOB API GET /prices-history (histórico completo)
                                          └──▶ gráfico Plotly: precio + umbral 0.25 + ventana 7d
```

No hay base de datos: todo el estado persistente vive en archivos (JSON, parquet, pickle, CSV). El dashboard es esencialmente stateless salvo por el CSV cacheado en disco.

### 2.2 Stack tecnológico y versiones

De [`requirements.txt`](requirements.txt) (raíz del repo, dependencias mínimas para el dashboard):

```
streamlit
plotly
pandas==3.0.2
numpy==2.4.4
scikit-learn==1.8.0
joblib
requests
pyarrow
```

De [`runtime.txt`](runtime.txt): `3.11` — versión de Python leída por Streamlit Community Cloud para elegir el runtime del contenedor de despliegue.

Solo `pandas`, `numpy` y `scikit-learn` están pineados a versión exacta; `streamlit`, `plotly`, `joblib`, `requests` y `pyarrow` no tienen versión fija. Esto es intencional (ver commit `1b5e650`, "Sin versiones fijas en requirements para compatibilidad con Python 3.14") pero deja el pineo inconsistente: si Streamlit Cloud resuelve una versión de `joblib` o `streamlit` incompatible con Python 3.11 en el futuro, el build podría romperse sin aviso.

**Nota sobre dependencias faltantes:** `src/models/phase5.py`, `phase6.py` y `phase7.py` importan `matplotlib.pyplot`, `matplotlib.gridspec`, `sklearn.calibration`, `sklearn.inspection.permutation_importance` — ninguno de esos requiere paquetes extra salvo `matplotlib`, que **no está en `requirements.txt`**. Confirmado por grep de imports: los tres scripts de fase importan `matplotlib` sin que aparezca en ningún requirements versionado. Quien quiera re-ejecutar el pipeline de investigación completo (no solo el dashboard) necesita instalarlo manualmente. `scipy`, `statsmodels` e `imbalanced-learn`, mencionados como stack en `PROJECT_BRIEF.md`, no aparecen como import directo en `phase5/6/7.py`; su uso real, si lo hubo, habría sido dentro de `notebooks/02_eda.ipynb` (no auditado celda por celda en esta sesión) — marcado como **por confirmar**.

**Infraestructura de desarrollo:**
- Dev Container: [`.devcontainer/devcontainer.json`](.devcontainer/devcontainer.json) — imagen `mcr.microsoft.com/devcontainers/python:1-3.11-bookworm`. `updateContentCommand` instala `requirements.txt` y además `streamlit` explícitamente (redundante, ya está en `requirements.txt` — deuda técnica menor). `postAttachCommand` lanza automáticamente `streamlit run dashboard.py --server.enableCORS false --server.enableXsrfProtection false`, puerto 8501 con `onAutoForward: "openPreview"`. Pensado para abrir el proyecto en GitHub Codespaces sin setup manual.
- No hay `Pipfile`, `poetry.lock` ni `pyproject.toml` — gestión de dependencias es `pip` + `requirements.txt` plano.

### 2.3 Por qué estas tecnologías

- **Streamlit:** permite un dashboard interactivo completo sin separar backend/frontend ni escribir HTML/JS a mano — coherente con un proyecto de una sola persona que necesita una demo pública rápida a partir de un modelo ya entrenado. La razón explícita no está comentada en el código; se infiere del uso (`st.cache_data`, `st.session_state`, `subprocess` hacia un script CLI) y de que el propio `PROJECT_BRIEF.md` pide una entrega de "calidad profesional" con salida visual. **No documentado explícitamente en ningún `.md`.**
- **scikit-learn:** stack estándar de ML enseñado en la especialización UBA (`PROJECT_BRIEF.md`, sección "Stack y entorno" lista `pandas, numpy, scikit-learn, statsmodels, scipy`). Elegido por sobre alternativas como XGBoost/LightGBM explícitamente: `DECISIONES.md` sección 5 dice "Gradient Boosting (sklearn GBM o LightGBM)" como opción abierta, y terminó usándose el GBM nativo de sklearn — no hay justificación explícita de por qué no LightGBM, se infiere que fue por simplicidad y para no agregar una dependencia adicional a un feature set pequeño (22 features, ~965 filas).
- **Polymarket Gamma API + CLOB API:** son la única fuente de datos posible para este dominio — no hay elección real, es el objeto de estudio. La documentación explícita de por qué se usa cada endpoint específico está en `DECISIONES.md` sección 8 (Gamma para metadata/listado, CLOB para históricos de precio, ambos sin autenticación).
- **Plotly (dashboard) vs. matplotlib (reportes/notebooks):** Plotly se usa únicamente en `dashboard.py` para el gráfico interactivo de precio histórico (zoom, hover); matplotlib con backend `Agg` se usa en `phase5/6/7.py` para generar las figuras estáticas de `reports/figures/*.png`. Separación razonable: interactividad en el producto, reproducibilidad/estática en el informe. No comentado explícitamente en el código, inferido del uso.
- **pyarrow / parquet:** formato columnar para `data/processed/`, más eficiente que CSV para el tipo de dato (floats + pocos miles de filas). No hay comentario explícito; es una elección estándar en pipelines de features tabulares.

### 2.4 Documentos preexistentes y su estado

Antes de este `CONTEXT.md` existían otros documentos de contexto en la raíz del repo. Tabla de qué contiene cada uno, si está versionado, y qué tan confiable es:

| Documento | Versionado | Contenido | Confiabilidad / relación con este documento |
|---|---|---|---|
| **`CONTEXT.md` (versión anterior, reemplazada por este archivo)** | **No** (estaba listado en `.gitignore` hasta esta sesión) | Análisis automático exhaustivo del repo, generado 2026-07-04. Muy detallado: stack, arquitectura, estructura de carpetas, flujos, convenciones. | Alto nivel de detalle y en general preciso **para la fecha en que se escribió**. Su falla principal detectada: cita hashes de commit (`6bbfcf0`, `be99812`, `ca7ebf5`, `d31d8bc`, etc.) que **eran reales al momento de escribirse** (2026-07-04) pero dejaron de existir como commits alcanzables después de una reescritura de historial con `git filter-branch` ocurrida entre el 2026-07-13 y el 2026-07-16 (ver [§11](#11-deuda-técnica-y-known-issues) y [§16](#16-inconsistencias-detectadas-durante-la-generación)). No es que el documento haya inventado los hashes — quedaron obsoletos por un evento posterior a su escritura. Todo el contenido verificable de esa versión fue re-chequeado contra el código real antes de incorporarse acá; lo no verificable se marca explícitamente como tal en las secciones correspondientes. |
| **`DECISIONES.md`** | Sí | Documento vivo de decisiones metodológicas y de scope con justificación, organizado en 15 secciones numeradas cronológicamente. Es la pieza de documentación **más rica y confiable** del repo — registra hallazgos del EDA, métricas exactas de cada fase de modelado, y decisiones sobre leakage, split temporal y balanceo de clases con su razonamiento. | Muy alta. Es fuente primaria para las secciones 5, 6 y 13 de este documento. Su única limitación: la última entrada ("Próximos 3 pasos") quedó en "Fase 7 — siguiente", es decir, **nunca se actualizó** para reflejar que las Fases 7 y 8 sí se completaron (confirmado por `git log`: commits `2023998`, `93d5d15`, `c290a29`, `c4efa9f`, `9e4be89`, todos posteriores a la fecha implícita del documento). |
| **`PROJECT_BRIEF.md`** | Sí | Brief original del TFI: contexto académico, pregunta de investigación, hipótesis iniciales, restricciones de scope, stack planeado, estructura de carpetas planeada, plan de fases con horas estimadas, y las instrucciones de cómo el autor quiere trabajar con el asistente de IA. | Alta como fuente del "por qué" original del proyecto, pero es un **documento de intención**, no de estado: varias cosas planeadas ahí no se concretaron así (notebooks `01/03/04/06` nunca se crearon, `reports/informe_final.md` fue borrado, el requirements.txt final es mucho más chico que el stack planeado). Útil para entender el contrato original, no para saber el estado actual. |
| **`PROGRESS.md`** | Sí | Bitácora de sesiones, pensada para actualizarse "al final de cada sesión de trabajo". | Baja como fuente de estado actual: **solo tiene la Sesión 1** (Fase 0, 2026-05-02). Todo el trabajo posterior (Fases 1 a 8, pivote a dashboard, rediseño de UI) no está documentado ahí. Sirve únicamente como confirmación de la fecha de arranque del proyecto. |
| **`README.md`** | Sí | Presentación breve orientada a un lector externo (evaluador o visitante del repo): pregunta de investigación, estructura de carpetas planeada, setup con conda, badge del dashboard deployado. | Media — es correcto pero superficial y con la misma estructura de carpetas "planeada" (no real) que `PROJECT_BRIEF.md`. No documenta el pivote a dashboard ni el estado real de `data/`. |
| **`CLAUDE.md`** | Sí (agregado en el mismo commit que el rediseño de UI, `4492036`, 2026-07-16) | Una única regla operativa sobre cómo deben verse los commits del repo (ver contenido literal en [§10](#10-convenciones-del-proyecto)). | Alta, es la fuente de verdad para convenciones de commit. Ver §11/§13 sobre su relación temporal con la reescritura de historial del mismo día. |

**Conclusión práctica:** para entender decisiones metodológicas del TFI (features, splits, modelos), `DECISIONES.md` sigue siendo la referencia primaria y este documento no la reemplaza — la resume y la cruza contra el código, pero no repite sus 15 secciones completas. Para entender el estado actual del repo, el dashboard, el deploy y las convenciones de trabajo, este `CONTEXT.md` es la referencia más actualizada.

---

## 3. Estructura de carpetas

### 3.1 Árbol completo con descripciones

```
Polymarket ML Predictor/                    ← raíz del repo git
│
├── README.md                    Presentación breve del TFI + badge del dashboard
├── PROJECT_BRIEF.md             Brief original: objetivos, plan de fases, cómo trabajar con IA
├── DECISIONES.md                Decisiones metodológicas documentadas — LEER ANTES de tocar
│                                 features, splits o modelos (ver §2.4)
├── PROGRESS.md                  Bitácora de sesiones — desactualizada tras la Sesión 1
├── CONTEXT.md                   Este documento
├── CLAUDE.md                    Regla de estilo de commits (ver §10)
├── requirements.txt             Dependencias mínimas para correr el dashboard (§2.2)
├── runtime.txt                  "3.11" — versión de Python para Streamlit Community Cloud
├── dataset.xlsx                 Export en Excel del dataset de análisis (§4.4) — 965 filas,
│                                 29 columnas, incluye columnas descartadas del feature set
│                                 final por leakage (log_volumen_total, duration_days, cluster)
├── dashboard.py                 App Streamlit — punto de entrada del dashboard (§7)
├── score_markets.py             Script CLI de scoring de mercados abiertos (§8)
├── probe_exclusions.py          Script exploratorio ad-hoc (Fase 1/2) para depurar por qué
│                                 se excluían candidatos de la API — no forma parte del
│                                 pipeline productivo, es una herramienta de diagnóstico puntual
├── scoring_output.csv           Output de score_markets.py — gitignored, se regenera en
│                                 cada corrida, existe localmente
│
├── .streamlit/
│   └── config.toml              Theme dark del dashboard (§7.5)
│
├── .devcontainer/
│   └── devcontainer.json        Config de Codespaces/Dev Container (§2.2)
│
├── data/
│   ├── raw/                     Datos crudos descargados de la API — GITIGNORED (no versionado,
│   │   ├── markets/               "no modificar, no commitear" según DECISIONES.md §11).
│   │   ├── prices/                Existe localmente en esta máquina. Un JSON por mercado en
│   │   ├── candidates.json        cada subcarpeta (markets/ y prices/).
│   │   └── download_run.json    candidates.json cachea la lista de candidatos tras paginar
│   │                             Gamma API; download_run.json resume metadata de la corrida.
│   ├── interim/
│   │   ├── .gitkeep
│   │   └── eda_dataset.parquet   Versionado. Dataset intermedio usado en notebooks/02_eda.ipynb.
│   └── processed/                Versionado — a diferencia de data/raw/, estos SÍ están en git.
│       ├── .gitkeep
│       ├── train.parquet (n=684) Features finales listas para modelar.
│       ├── val.parquet (n=76)
│       ├── test.parquet (n=205)
│       ├── scaler.pkl            StandardScaler fiteado en train (Fase 4) — se reusa en
│       │                         score_markets.py para escalar mercados en vivo.
│       ├── feature_columns.json  Metadata: numeric / binary_onehot / all_features / target
│       └── split_stats.json      n, YES%, distribución por bucket temporal y categoría
│
├── notebooks/                    Solo 2 de los 6 notebooks planeados en PROJECT_BRIEF.md
│   ├── 02_eda.ipynb                existen; el resto del trabajo terminó en scripts .py
│   └── 05_analisis_resultados.ipynb  dentro de src/, no en notebooks separados.
│
├── src/                           Código reusable (no notebooks), todo versionado
│   ├── config.py                  Seeds y constantes globales centralizadas (§5, §6)
│   ├── data/
│   │   ├── download.py (~765 l.)    Pipeline de descarga Fase 2 (§4, §5.1)
│   │   └── make_dataset.py (~292 l.) Feature engineering + split (§5)
│   ├── features/
│   │   └── categorization.py (~115 l.) Reglas heurísticas de categorización (§5.2)
│   ├── models/
│   │   ├── phase5.py (~553 l.)      Baselines + regresión logística (§6)
│   │   ├── phase6.py (~381 l.)      Random Forest + Gradient Boosting (§6)
│   │   ├── phase7.py (~401 l.)      Threshold optimization, calibración, errores (§6)
│   │   └── create_notebook_fase7.py  Genera notebooks/05_analisis_resultados.ipynb
│   │                                  programáticamente
│   └── visualization/               Paquete vacío (solo __init__.py) — el plan original
│                                     preveía funciones de plotting reutilizables acá; en la
│                                     práctica todo el plotting quedó inline en phase5/6/7.py
│
├── models/                         Versionados — los .pkl están en el repo, no se regeneran
│   ├── best_lr.pkl (881 B)           LogisticRegression L1/saga/C=0.5 ("LR-C")
│   └── best_tree_model.pkl (~396 KB) GradientBoostingClassifier — el "champion"
│
├── reports/                        Versionado (35 archivos entre .md, .json y figures/)
│   ├── fase5_baselines_logistica.md
│   ├── fase5_metrics_checkpoint.json
│   ├── fase5_metrics_full.json
│   ├── fase6_rf_gb.md
│   ├── fase6_metrics_full.json
│   ├── fase7_analysis.json         Threshold sweep, calibración ECE/MCE, feature importance,
│   │                                análisis de errores por categoría — fuente primaria de §6
│   └── figures/                    ~25 PNG generados por phase5/6/7.py (ROC, matrices de
│                                    confusión, calibración, importances, clustering, etc.)
│   (No existe reports/informe_final.md ni .docx — borrados, ver commits 0facb83/6a57a65)
│
├── tests/
│   └── __init__.py                 Vacío. Sin tests implementados.
│
└── polymarket-bot/                 SUBPROYECTO NO VERSIONADO (gitignored: "polymarket-bot/")
    ├── .env                        Credenciales — nunca en git
    ├── requirements.txt            Propio, distinto del de la raíz
    ├── main.py                     CLI (subcomandos, no auditados en profundidad esta sesión)
    ├── debug_*.py                  Scripts de debugging ad-hoc
    ├── data/
    │   ├── binance/, polymarket/
    │   ├── backtest_results.csv
    │   └── live_trades.json
    └── src/
        ├── data/, strategy/, backtest/, live/, dashboard/
```

*Nota sobre `polymarket-bot/`: en esta sesión confirmé por listado de directorio que estas carpetas y archivos existen en disco (comando `find`), pero no leí su contenido línea por línea. La descripción funcional detallada de este subproyecto (modelo de probabilidad log-normal, motor de backtest, paper trading) proviene de la versión anterior de `CONTEXT.md` y **no fue re-verificada contra el código en esta sesión** — tratarla como "según CONTEXT.md previo, no verificable en esta sesión" hasta confirmarla.*

### 3.2 Qué está en `.gitignore` y por qué

Contenido completo de [`.gitignore`](.gitignore) al momento de este documento:

```gitignore
# Datos crudos — no van al repo (pueden pesar GBs y contienen snapshots de la API)
data/raw/

# Entornos virtuales
venv/
.venv/
env/
.env/

# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
.eggs/

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# Variables de entorno (credenciales, tokens)
.env
.env.local

# IDEs
.vscode/
.idea/
*.code-workspace

# OS
.DS_Store
Thumbs.db

# Outputs temporales (las figuras finales van en reports/figures/ y SÍ van al repo)
*.log

# Ajenos a este proyecto / generados
CONTEXT.md
polymarket-bot/
scoring_output.csv
```

Esta lista es la del último commit (`4492036`); **como parte de esta actualización se retira la línea `CONTEXT.md`** para que este documento quede versionado (ver instrucción explícita del autor — este archivo debe tener valor para cualquier lector futuro, no solo local).

Razones de cada exclusión:
- `data/raw/`: datos crudos de la API, potencialmente pesados (miles de JSONs), y considerados "sagrados" — no se versionan ni se modifican manualmente (`DECISIONES.md` §11). Reproducibles re-corriendo `src/data/download.py`.
- `venv/`, `.venv/`, `env/`, `.env/`, `__pycache__/`, etc.: exclusiones estándar de entorno Python.
- `.env`, `.env.local`: credenciales. El TFI principal no usa ninguna (APIs públicas sin auth), pero la regla está para prevenir el caso — y es directamente relevante para `polymarket-bot/.env`, que si no estuviera ya cubierto por la exclusión completa de `polymarket-bot/`, expondría credenciales de Binance/Polymarket.
- `polymarket-bot/`: subproyecto entero excluido — nunca se decidió versionarlo junto con el TFI.
- `scoring_output.csv`: output regenerable en cada corrida de `score_markets.py`, no tiene sentido versionarlo (cambiaría en cada ejecución y generaría diffs de ruido).
- `CONTEXT.md`: **exclusión retirada en esta actualización** — antes era generado y consultado solo localmente; a partir de ahora se versiona.

---

## 4. Fuentes de datos

### 4.1 Gamma API — listado y metadata de mercados

Base: `https://gamma-api.polymarket.com` (constante `GAMMA_API_BASE` en [`src/config.py`](src/config.py)).

**Uso 1 — descarga histórica** (`src/data/download.py`): pagina `GET /markets` ordenado por `closedTime` descendente, buscando mercados ya **resueltos**. Implementa "plateau detection": si las últimas `PLATEAU_WINDOW=20` páginas suman menos de `PLATEAU_MIN_NEW=2` candidatos nuevos que pasen los filtros, se asume que la paginación entró en territorio anterior a `DATE_START` y se corta (evita escanear todo el historial de Polymarket). Pausa entre requests: `API_RATE_LIMIT_PAUSE=1.0s` (constante en `config.py`, para este endpoint específico, más conservadora que la de CLOB).

**Uso 2 — mercados abiertos en vivo** (`score_markets.py::fetch_open_markets`): pagina `GET /markets` con parámetros `active=true`, `closed=false`, `limit=100`, cursor-based (`next_cursor`). Filtra a mercados binarios (`len(tokens|outcomes) == 2`) con `clobTokenIds` presente, y descarta los que llevan menos de `OBS_DAYS=7` días abiertos (compara `startDate`/`createdAt` contra `now - 7 días`).

**Uso 3 — detalle puntual** (`dashboard.py::fetch_price_history`): `GET /markets?conditionId=<id>` para obtener `clobTokenIds` de un mercado específico al mostrar su detalle.

**Endpoint deprecado, sigue funcionando:** según `DECISIONES.md` §8, `/markets` tiene header `Sunset: 2026-05-01` (ya vencido a la fecha de este documento) pero continúa respondiendo. No hay endpoint de reemplazo documentado. Riesgo aceptado, mitigado descargando el dataset completo antes de que el endpoint se apagara.

### 4.2 CLOB API — historial de precios

Base: `https://clob.polymarket.com` (constante `CLOB_API_BASE`).

`GET /prices-history` con parámetros `market=<clobTokenId>`, `fidelity=1440` (granularidad diaria, en minutos), y opcionalmente `startTs`/`endTs` (unix timestamp) para acotar la ventana. Respuesta:

```json
{"history": [{"t": 1717027200, "p": 0.62}, {"t": 1717113600, "p": 0.58}, ...]}
```

`t` = timestamp unix, `p` = precio del token YES (0 a 1). **Sin volumen** — limitación documentada de la API que llevó a descartar features de volumen/actividad por día (`DECISIONES.md` §6, "Features eliminadas respecto al plan original"). No requiere autenticación para este endpoint específico; `GET /trades` del mismo CLOB sí la requiere (401 sin API key) y por eso fue descartado como fuente de features.

### 4.3 Dataset histórico de entrenamiento (`data/processed/*.parquet`)

Generado por `src/data/make_dataset.py` a partir de `data/raw/{markets,prices}/*.json`. Esquema (una fila = un mercado, 22 features + target + metadata):

| Columna | Tipo | Descripción |
|---|---|---|
| `precio_dia_1` … `precio_dia_7` | float (escalado) | Precio del token YES en cada día 1-7, con forward-fill + backward-fill |
| `precio_inicio` | float (escalado) | Primer precio real observado (sin fill) |
| `precio_fin` | float (escalado) | Último precio real observado en la ventana — feature más predictiva individualmente y baseline B3 |
| `precio_media`, `precio_mediana`, `precio_std`, `precio_rango` | float (escalado) | Estadísticos descriptivos de la serie de precios cruda |
| `precio_tendencia` | float (escalado) | Pendiente de `np.polyfit(t, prices, 1)` sobre los puntos reales |
| `volatilidad_retornos` | float (escalado) | Desvío estándar de retornos log diarios `log(p_t / p_{t-1})` |
| `n_puntos_precio` | int (escalado) | Cantidad de observaciones reales en la ventana (rango 3-7) |
| `cat_Crypto`, `cat_Entertainment`, `cat_Finance`, `cat_Politics`, `cat_Sports`, `cat_Tech` | binario (0/1) | One-hot de `category_coarse` — "Other" es la categoría de referencia, omitida |
| `outcome` | binario (0/1) | **Target.** 1 = YES resolvió, 0 = NO resolvió |
| `condition_id`, `split`, `bucket`, `category_coarse` | — | Metadata, no entran al vector de features del modelo |

Tamaños reales (de [`data/processed/split_stats.json`](data/processed/split_stats.json), verificado): **train n=684 (12.4% YES), val n=76 (9.2% YES), test n=205 (11.2% YES)** — total 965. `DECISIONES.md` menciona 967 como total post-filtro `MIN_PRICE_POINTS`; la suma real de los tres splits es 965. Diferencia de 2 registros no explicada por el código auditado — **por confirmar** (posibles candidatas: mercados con `startDate` no parseable, o sin archivo de precios pese a tener JSON de metadata, ambos casos son `continue` silenciosos en `make_dataset.py::main()`).

### 4.4 `dataset.xlsx`

Verificado directamente con `pandas.read_excel()` en esta sesión: **965 filas × 29 columnas**. Superset de las columnas del dataset de modelado: incluye todas las 22 features + `outcome`, más `question`, `event_ticker`, `start_date`, `closed_date`, `duration_days`, `log_volumen_total`, `start_month`, `closed_month`, `start_quarter`, y `cluster` (resultado de K-means, ver `DECISIONES.md` §7).

**Importante:** `duration_days`, `log_volumen_total` y `cluster` son columnas que **fueron descartadas del feature set final por leakage** (`duration_days` no se conoce hasta la resolución; `log_volumen_total` es un snapshot acumulado de toda la vida del mercado) o son subproducto exploratorio (`cluster`). Su presencia en `dataset.xlsx` confirma que este archivo es un **export de análisis/auditoría** (probablemente para revisión manual o como anexo del informe), no el dataset que efectivamente entra al modelo — ese es `data/processed/{train,val,test}.parquet`. No hay ningún `.md` que documente explícitamente el propósito de `dataset.xlsx`; se infiere del contenido de sus columnas. **Por confirmar** con el autor si tuvo un uso puntual (ej. anexo del informe borrado) o si sigue teniendo alguna función activa.

### 4.5 `scoring_output.csv`

Output de `score_markets.py`. Gitignored, se regenera en cada corrida. Columnas confirmadas (leyendo el archivo generado localmente):

```
question,category,start_date,prob_yes_gb,prob_yes_lr,pred_gb,pred_lr,condition_id
```

Ejemplo de fila real:
```
Will MegaETH perform an airdrop by June 30? ,Crypto,2025-06-26,0.1973224975599783,0.09387893747395365,0,0,0xe459d1b598...
```

`prob_yes_gb`/`prob_yes_lr` son probabilidades float sin redondear (el redondeo a 1 decimal para mostrar en el dashboard ocurre en `dashboard.py::load_csv()`, no en `score_markets.py`). `pred_gb`/`pred_lr` son 0/1, resultado de aplicar `THRESHOLD=0.25` a cada probabilidad respectivamente — aunque en la práctica el dashboard solo usa `pred_gb` para la columna "Señal".

### 4.6 Frecuencia de actualización y mecanismo de refresh

No hay cron ni scheduler. El refresh es **manual y on-demand**: el usuario del dashboard presiona "🔄 Actualizar scores" en el sidebar, lo que dispara un `subprocess.run([sys.executable, "score_markets.py", "--top", "200"], ...)` desde `dashboard.py::run_scorer()`. Si el proceso falla (`returncode != 0`), el `stderr` (últimos 2000 caracteres) se guarda en `st.session_state["scorer_error"]` y se muestra como error en la UI; si tiene éxito, se limpia el cache (`st.cache_data.clear()`) y se fuerza un `st.rerun()`. No hay reintentos automáticos ni backoff.

---

## 5. Pipeline de datos

### 5.1 De raw a feature vector, paso a paso

1. **Descarga** (`src/data/download.py`): pagina Gamma API con plateau detection → filtra con `passes_filters()` (cerrado, duración ≥ `MIN_MARKET_DURATION_DAYS=30`, `startDate ≥ DATE_START="2023-01-01"`, outcome no ambiguo, binario) → cachea candidatos en `data/raw/candidates.json` → descarga precios por candidato vía CLOB (`GET /prices-history`), un JSON por mercado en `data/raw/prices/{conditionId}.json`, **reanudable** (si el archivo ya existe, se saltea).
2. **Construcción de features** (`src/data/make_dataset.py::build_price_features()`): por cada par `(markets/{cid}.json, prices/{cid}.json)`, mapea cada observación de precio a un día 1-7 relativo a `startDate`, aplica forward-fill + backward-fill sobre el vector de 7 días, calcula agregados (media, mediana, std, rango, tendencia por regresión lineal, volatilidad de retornos log) sobre los **puntos reales** (no sobre los imputados), y cuenta `n_puntos_precio`.
3. **Target:** `outcome = int(outcomePrices[0] == "1")`, leído del JSON de metadata Gamma. Si `outcomePrices[0]` es un valor intermedio (mercado cancelado), el mercado ya fue descartado en el paso de filtros de descarga.
4. **Categorización:** `infer_category_coarse(question)` (ver §5.2) sobre el texto de la pregunta.
5. **Bucket temporal:** `temporal_bucket(start_dt)` asigna uno de 4 buckets (`pre-2026`, `2026-01`, `2026-02`, `2026-03+`) según `startDate`.
6. **Split:** `assign_split(condition_id, bucket)` — determinístico vía `hash(MD5(conditionId)) % 100` (0-69 train, 70-79 val, 80-99 test), calculado **dentro de cada bucket** para evitar el drift intra-mes detectado en el EDA (ver §13).
7. **Escalado:** `StandardScaler` fiteado exclusivamente en train, aplicado a los tres splits.
8. **Persistencia:** `data/processed/{train,val,test}.parquet`, `scaler.pkl`, `feature_columns.json`, `split_stats.json`. El script imprime además una tabla de verificación de no-leakage feature por feature (ver bloque de código en §5.3).

### 5.2 Categorización de mercados (`src/features/categorization.py`)

7 categorías mutuamente excluyentes, asignadas por **keyword matching** (no un clasificador entrenado) sobre el texto de `question` en minúsculas. El orden de evaluación importa — primera coincidencia gana:

```python
_CAT_RULES: list[tuple[str, list[str]]] = [
    ("Sports", [...]),        # nba, nhl, nfl , mlb, ufc, playoffs, vezina, ...
    ("Politics", [...]),      # election, president, russia , ukraine, gaza, ...
    ("Crypto", [...]),        # bitcoin, btc, ethereum, " sol ", " bnb", nft, ...
    ("Finance", [...]),       # stock, nasdaq, fed rate, ecb, bank of canada, ...
    ("Tech", [...]),          # openai, nvidia, apple, tesla, gemini , deepseek, ...
    ("Entertainment", [...]), # oscar, grammy, spotify, monthly listeners, ...
]
# Sin match → "Other"
```

Orden completo: **Sports > Politics > Crypto > Finance > Tech > Entertainment > Other**. Ejemplo de por qué el orden importa (documentado en el docstring del archivo): `"Finance"` se evalúa antes que `"Tech"` para que `"Will Apple dip to $240"` clasifique como Finance (pregunta sobre precio de acción), no Tech. Algunos keywords incluyen espacios deliberadamente para evitar falsos positivos: `"nfl "` (no `"nfl"`) para no capturar `"nflx"` (ticker de Netflix); `" bnb"` (no `"bnb"`) para no capturar `"airbnb"`.

**Reutilización exacta en producción:** `score_markets.py` importa la misma función (`from src.features.categorization import infer_category_coarse`), no hay una versión duplicada — a diferencia del resto del feature engineering (ver §11).

Las reglas actuales son la "v2" (refinadas post-EDA); según `DECISIONES.md` §6 y §12, la v1 clasificaba 28% de los mercados como "Other", bajado a 6.5% en v2 al rescatar mercados geopolíticos → Politics, bancos centrales → Finance, charts musicales → Entertainment, etc.

### 5.3 Feature engineering: las 22 features

Definidas canónicamente en `src/data/make_dataset.py` (y duplicadas en `score_markets.py`, ver §11):

```python
CAT_DUMMIES = ["Crypto", "Entertainment", "Finance", "Politics", "Sports", "Tech"]
PRICE_DAY_COLS  = [f"precio_dia_{d}" for d in range(1, 8)]                       # 7
PRICE_AGG_COLS  = ["precio_inicio", "precio_fin", "precio_media", "precio_mediana",
                   "precio_std", "precio_rango", "precio_tendencia",
                   "volatilidad_retornos"]                                       # 8
ACTIVITY_COLS   = ["n_puntos_precio"]                                            # 1
CAT_COLS        = [f"cat_{c}" for c in CAT_DUMMIES]                              # 6
ALL_FEATURE_COLS = PRICE_DAY_COLS + PRICE_AGG_COLS + ACTIVITY_COLS + CAT_COLS    # 22 total
```

Todas las features de precio y actividad vienen de CLOB `/prices-history` dentro de la ventana de 7 días (por diseño, no puede haber leakage temporal: nada usa datos posteriores al día 7). Las 6 dummies de categoría vienen del texto de `question`, disponible desde el momento en que el mercado abre. El propio `make_dataset.py` imprime una tabla de auto-verificación al final de la corrida:

```
VERIFICACION DE NO-LEAKAGE
  precio_dia_1..7               Solo usa precios del historial de precios en la ventana start_date + 7 dias. OK.
  precio_inicio/fin/...         Calculados sobre history dentro de la ventana de 7 dias. OK.
  volatilidad_retornos          Log-returns calculados sobre prices[] dentro de la ventana. OK.
  n_puntos_precio                Cuenta de observaciones en la ventana. OK.
  cat_*                          Derivado del texto de la pregunta (disponible al abrir el mercado). OK.
  log_volumen_total              EXCLUIDA (leakage: snapshot incluye volumen post-ventana).
  duration_days                  EXCLUIDA (leakage: duracion solo conocida tras resolucion).
```

Este patrón (documentar por qué cada feature es segura, dentro del propio script) es la convención a seguir si se agregan features nuevas — ver `DECISIONES.md` §11, punto 1.

### 5.4 Feature engineering — detalle de cálculo (`build_price_features`)

```python
# src/data/make_dataset.py
def build_price_features(history: list, start_ts: int) -> dict | None:
    history = sorted(history, key=lambda x: x["t"])
    n_pts = len(history)
    if n_pts < MIN_PRICE_POINTS:        # 3
        return None
    prices = [h["p"] for h in history]
    ...
    row["precio_tendencia"] = float(np.polyfit(t_norm, prices, 1)[0]) if n_pts >= 2 else 0.0
    if n_pts >= 3:
        safe = [(prices[i], prices[i-1]) for i in range(1, n_pts)
                if prices[i] > 0 and prices[i-1] > 0]
        log_rets = [np.log(a / b) for a, b in safe]
        row["volatilidad_retornos"] = float(np.std(log_rets, ddof=1)) if len(log_rets) >= 2 else 0.0
    ...
```

Detalle de imputación (`DECISIONES.md` §6): solo 74/965 mercados (7.7%) tienen su primer precio real en el día 2 en vez del día 1; ninguno tiene su primer precio en día ≥ 3. El único caso de backward-fill real en la práctica es completar `precio_dia_1` con `precio_dia_2`. La volatilidad se calcula **sobre los puntos crudos**, no sobre el vector ya imputado, por lo que el forward-fill no la contamina.

---

## 6. Modelos ML

Tres modelos comparados contra dos baselines triviales y un baseline no-trivial (B3). Todas las métricas de esta sección vienen de `reports/fase5_metrics_full.json`, `reports/fase6_metrics_full.json` y `reports/fase7_analysis.json`, verificadas contra el contenido real de esos archivos en esta sesión.

### 6.1 Baselines (Fase 5, `src/models/phase5.py`)

| Modelo | AUC (test) | Descripción |
|---|---|---|
| B1 | 0.5000 | Predice siempre la clase mayoritaria (NO) |
| B2 | 0.5000 | Predice el prior de clase (12.4% YES) constante |
| **B3** | **0.8471** | `precio_fin` tal cual, sin modelo — el baseline "serio" a superar |

### 6.2 Regresión logística — LR-C (champion de Fase 5)

`sklearn.linear_model.LogisticRegression(penalty="l1", solver="saga", C=0.5)`. Serializada en [`models/best_lr.pkl`](models/best_lr.pkl) (881 bytes — solución esparsa, coherente con L1). Sobre 5 variantes de LR probadas (sin regularización, 4-features mínimo, L2 C=50, L1 C=0.5, L1 balanced), LR-C ganó por mejor combinación de AUC (0.8339), Brier (0.0715), LogLoss (0.2545) con solo **13 de 22 features activas** (L1 zeró 9: `precio_dia_2/3/5`, `precio_mediana`, `precio_rango`, `volatilidad_retornos`, `cat_Entertainment/Finance/Tech`).

### 6.3 Gradient Boosting — champion general (Fase 6)

`sklearn.ensemble.GradientBoostingClassifier`, hiperparámetros ganadores tras `RandomizedSearchCV(n_iter=25, cv=5, scoring="roc_auc", random_state=42)`:

```python
GradientBoostingClassifier(
    n_estimators=100, max_depth=6, learning_rate=0.01,
    subsample=0.7, max_features=None,
    min_samples_split=5, min_samples_leaf=4,
)
```

Serializado en [`models/best_tree_model.pkl`](models/best_tree_model.pkl) (~396 KB). Espacio de búsqueda explorado (`phase6.py`):

```python
param_dist = {
    "n_estimators":  [100, 150, 200, 250, 300],
    "max_depth":     [2, 3, 4, 5, 6],
    "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.15, 0.2],
    # + subsample, min_samples_split, min_samples_leaf, max_features (no capturados
    #   en el grep de esta sesión, valores ganadores arriba confirmados por reports/fase6_rf_gb.md)
}
```

### 6.4 Random Forest (evaluado, no productivizado como default)

```python
RandomForestClassifier(
    n_estimators=300, max_depth=5, min_samples_split=20,
    min_samples_leaf=8, max_features="sqrt", class_weight="balanced",
)
```

RF obtuvo AUC=0.8876 (test) — muy cerca de GB (0.8933) y con mejor F1(YES)=0.5833 vs 0.3333 de GB — pero **no se serializó ningún `best_rf_model.pkl`**; el repo solo tiene `best_tree_model.pkl` (GB). El criterio de selección del champion general fue AUC + calibración (Brier, LogLoss), no F1, lo cual explica por qué GB ganó pese a peor F1(YES) a threshold 0.5 — ver §6.5 sobre cómo el threshold 0.25 corrige esto en producción.

### 6.5 Tabla comparativa de métricas (test set, n=205)

| Modelo | AUC | PR-AUC | LogLoss | Brier | F1(YES) @ 0.5 |
|---|---|---|---|---|---|
| B1 (mayoría) | 0.5000 | 0.1122 | 1.8084 | 0.1122 | 0.0000 |
| B2 (prior) | 0.5000 | 0.1122 | 0.3518 | 0.0998 | 0.0000 |
| B3 (`precio_fin`) | 0.8471 | 0.6031 | 0.3697 | 0.1181 | 0.4615 |
| LR-C (champion LR) | 0.8339 | 0.5824 | 0.2545 | 0.0715 | 0.5000 |
| RF | 0.8876 | 0.6338 | 0.3414 | 0.1000 | **0.5833** |
| **GB (champion general)** | **0.8933** | **0.6348** | **0.2436** | **0.0678** | 0.3333 |

Feature importance top-5 de GB (MDI, `reports/fase7_analysis.json`): `precio_dia_6` (0.277), `precio_dia_7` (0.120), `precio_fin` (0.090), `precio_dia_4` (0.064), `precio_media` (0.061). Hallazgo notable (`DECISIONES.md` §15): `precio_dia_6` es la feature más importante para ambos modelos de árbol (RF y GB), algo que la regularización L1 de LR-C casi había eliminado (sobrevive con coeficiente +0.39) — evidencia de que los árboles capturan interacciones no lineales entre precios diarios consecutivos que la regresión logística lineal no puede.

### 6.6 Threshold de decisión = 0.25

Derivado formalmente en Fase 7 (`src/models/phase7.py`) barriendo thresholds de 0.05 a 0.50 sobre las probabilidades de GB en test, maximizando F1-macro. Tabla completa (de `reports/fase7_analysis.json`):

| Threshold | Precision | Recall | F1(YES) | F1(macro) | Accuracy | n_pred_YES |
|---|---|---|---|---|---|---|
| 0.05 | 0.114 | 1.000 | 0.205 | 0.124 | 0.132 | 201 |
| 0.10 | 0.328 | 0.870 | 0.476 | 0.671 | 0.785 | 61 |
| 0.15 | 0.471 | 0.696 | 0.561 | 0.745 | 0.878 | 34 |
| 0.20 | 0.636 | 0.609 | 0.622 | 0.788 | 0.917 | 22 |
| **0.25** | **0.722** | **0.565** | **0.634** | **0.797** | **0.927** | **18** |
| 0.30 | 0.688 | 0.478 | 0.564 | 0.759 | 0.917 | 16 |
| 0.40 | 0.692 | 0.391 | 0.500 | 0.726 | 0.912 | 13 |
| 0.50 | 0.714 | 0.217 | 0.333 | 0.640 | 0.902 | 7 |

`0.25` es el punto de máximo F1-macro (0.7967) y también de máximo F1(YES) (0.6341) en esta corrida — ambos criterios coinciden en el mismo threshold, lo que refuerza la elección. **Este es el mismo valor, literal, que se hardcodea como `THRESHOLD = 0.25`** tanto en `score_markets.py` como en `dashboard.py` — la investigación decide el umbral de negocio y producción lo reutiliza sin reajuste.

Calibración (ECE = Expected Calibration Error, MCE = Maximum Calibration Error): GB tiene ECE=0.0614/MCE=0.5429, LR-C tiene ECE=0.0294/MCE=0.3471 (mejor calibrado que GB pero con peor AUC), B3 tiene ECE=0.1836/MCE=0.4125 (peor calibrado que ambos modelos).

### 6.7 Cómo se cargan y usan los `.pkl` en runtime

```python
# score_markets.py
gb     = joblib.load(MODELS_DIR / "best_tree_model.pkl")
lr     = joblib.load(MODELS_DIR / "best_lr.pkl")
scaler = pickle.load(open(PROCESSED_DIR / "scaler.pkl", "rb"))
...
df[NUM_COLS] = scaler.transform(df[NUM_COLS])     # mismas 16 columnas numéricas del entrenamiento
X = df[ALL_FEATURES].values                        # orden exacto de las 22 columnas
df["prob_yes_gb"] = gb.predict_proba(X)[:, 1]
df["prob_yes_lr"] = lr.predict_proba(X)[:, 1]
df["pred_gb"]     = (df["prob_yes_gb"] >= THRESHOLD).astype(int)
```

Nótese la mezcla de `joblib.load` (modelos) y `pickle.load` (scaler) — ambos son formatos pickle-compatibles pero se usan dos APIs distintas sin razón funcional aparente; probablemente accidente de cómo se serializó cada objeto en su script de origen (`phase5/6.py` para modelos, `make_dataset.py` para el scaler). El orden de columnas en `X` importa: si `ALL_FEATURES` en `score_markets.py` alguna vez se desincroniza del orden usado al entrenar (`ALL_FEATURE_COLS` en `make_dataset.py`), las predicciones serían silenciosamente incorrectas sin ningún error — ambas listas están hoy en el mismo orden pero no hay ningún test que lo garantice (ver §11).

### 6.8 Compatibilidad de versiones: por qué `scikit-learn==1.8.0` y `numpy==2.4.4`

Objetos `.pkl`/`.joblib` de scikit-learn son sensibles a la versión exacta de sklearn/numpy con la que se serializaron — cargarlos con una versión distinta puede fallar silenciosamente (predicciones erróneas) o directamente lanzar excepción. Los modelos en `models/*.pkl` fueron entrenados y serializados el 2026-05-04 (fecha de modificación del archivo, confirmada por `ls -la`). El commit `052105f` ("Fix: pinear Python 3.11 y versiones sklearn/numpy para compatibilidad de modelos pkl", 2026-06-20) fija `scikit-learn==1.8.0`, `numpy==2.4.4` y `runtime.txt=3.11` explícitamente para que el entorno de Streamlit Community Cloud reconstruya el mismo entorno de serialización y los `.pkl` carguen sin error. Antes de ese commit, el requirements no tenía versiones fijas (commit `1b5e650`), lo cual en algún momento debió romper el deploy — es la explicación más directa de por qué existe este pineo (no hay un comentario explícito en el código que lo diga, pero el mensaje de commit es autoexplicativo).

---

## 7. Dashboard (Streamlit)

### 7.1 Estructura de `dashboard.py`

Punto de entrada: `streamlit run dashboard.py`. Organización del archivo (511 líneas):

1. **Config** (líneas 19-39): constantes (`SCORING_CSV`, `CLOB_API`, `GAMMA_API`, `THRESHOLD=0.25`, `OBS_DAYS=7`), diccionario `CATEGORY_COLORS`, diccionario `MESES_ES` para formateo de fecha en español, `st.set_page_config(page_title="Polymarket Scorer", page_icon="📊", layout="wide")`.
2. **Iconos** (líneas 47-59): helper `lucide()` que genera SVGs inline (íconos [Lucide](https://lucide.dev), sin dependencia externa de paquete de íconos — se pegan como paths SVG crudos).
3. **Estilos base** (líneas 62-229): un único bloque `st.markdown(..., unsafe_allow_html=True)` con CSS completo (ver §7.2).
4. **Helpers** (líneas 232-322): `run_scorer()`, `load_csv()`, `fetch_price_history()`, `format_fecha_es()`, `badge_signal()`, `badge_category()`, `prob_bar()`.
5. **Layout — sidebar** (líneas 336-368): sección "Acciones" (botón actualizar + timestamp del último scoring) y sección "Filtros" (categoría, señal, prob. mínima).
6. **Tabla principal** (líneas 370-438): aplica filtros, calcula KPIs, renderiza tabla HTML custom.
7. **Detalle de mercado** (líneas 440-502): `st.selectbox` para elegir mercado, métricas, gráfico Plotly de historial de precio.
8. **Footer** (líneas 504-511): umbral del modelo y versión.

### 7.2 Sistema de estilos

Paleta de colores (variables CSS en `:root`, definidas en el bloque `st.markdown` de `dashboard.py`, replicadas en `.streamlit/config.toml`):

```css
:root {
    --bg: #0a0a0b;          /* fondo principal */
    --surface: #141416;     /* superficie de cards/tabla */
    --border: #27272a;      /* bordes */
    --text: #fafafa;        /* texto principal */
    --text-muted: #a1a1a6;  /* texto secundario */
    --accent: #3b82f6;      /* azul de acento (botones primarios, hline threshold no usa este) */
}
```

Colores adicionales usados fuera de las variables CSS (hardcodeados inline):
- Badges de señal: `.badge-yes { color: #22c55e; background: #22c55e26; }` (verde), `.badge-no { color: #a1a1a6; background: #71717a26; }` (gris).
- Colores de categoría (`CATEGORY_COLORS` en Python): `Sports #06b6d4` (cyan), `Politics #a855f7` (violeta), `Crypto #f97316` (naranja), `Entertainment #ec4899` (rosa); resto usa `CATEGORY_COLOR_DEFAULT = "#71717a"` (gris) — nótese que **Finance y Tech no tienen color propio** pese a ser categorías con más volumen de mercados (Finance es la categoría mayoritaria, 36.6% según `DECISIONES.md`), caen en el gris default.
- Barra de probabilidad (`prob_bar()`): gradiente por magnitud — `< 33% → #64748b` (gris azulado), `33-66% → #f59e0b` (ámbar), `≥ 66% → #22c55e` (verde).

Tipografía: `Inter` (Google Fonts, `@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap')`) — única dependencia externa vía CDN del dashboard (si el usuario no tiene conexión a `fonts.googleapis.com`, cae al fallback `sans-serif` declarado en el mismo `font-family`).

Componentes custom (todos HTML+CSS a mano, no widgets nativos de Streamlit):
- **KPI cards** (`.kpi-grid`, `.kpi-card`): grid responsive (`repeat(auto-fit, minmax(200px, 1fr))`), cada card con ícono Lucide, label y valor tabular (`font-variant-numeric: tabular-nums`).
- **Tabla** (`.mkt-table-wrap`, `.mkt-table`): tabla HTML pura (no `st.dataframe`), con header sticky, hover sutil, filas alternadas con tinte muy leve (`rgba(255,255,255,0.015)`), altura máxima 560px con scroll interno.
- **Badges** (`.badge`): pastilla con texto centrado, `border-radius: 999px`, usado tanto para señal como para categoría (color paramétrico vía `style="color:...;background:...26;"`, donde `26` es el sufijo hex de opacidad ~15%).
- **Footer** (`.app-footer`): flexbox `justify-content: space-between`, separador superior.

**Nota de seguridad:** el commit del rediseño (`4492036`) agregó `html.escape()` explícito sobre el texto de `question` y `category` antes de inyectarlos en HTML crudo (`badge_category()`, y el `<td>` de pregunta en la tabla). Dado que esos textos vienen de la API pública de Polymarket (contenido no controlado por el propio proyecto), esto es una mitigación correcta contra HTML/script injection si algún `question` contuviera markup malicioso — antes de este commit el texto se interpolaba sin escapar.

### 7.3 Interactividad

- **Botón "🔄 Actualizar scores"** (sidebar, `type="primary"`): limpia el cache de Streamlit y dispara `run_scorer(top=200)`, que corre `score_markets.py` como subproceso.
- **Selectbox "Categoría"**: opciones = `["Todas"] + sorted(categorías únicas del CSV)`.
- **Radio "Señal GB"** (horizontal): `Todas` / `YES` / `NO`, mapeado internamente a `None` / `"✅ YES"` / `"NO"` para filtrar la columna `señal` calculada.
- **Slider "Prob. mínima GB"**: 0-100, step 5, filtra `prob_yes_gb_pct >= min_prob`.
- **Selectbox "Ver detalle de mercado"**: lista todos los mercados filtrados y ordenados, formateados por los primeros 90 caracteres de la pregunta; al elegir uno, se despliega la sección de detalle con métricas y gráfico Plotly.
- El **filtro de señal solo aplica sobre GB**, no hay forma de filtrar por señal de LR en la UI — coherente con que LR es un modelo secundario/comparativo, GB es el "champion" mostrado como señal principal.

### 7.4 Sistema de caching

```python
@st.cache_data(ttl=300)   # 5 minutos
def load_csv() -> pd.DataFrame: ...

@st.cache_data(ttl=600)   # 10 minutos
def fetch_price_history(condition_id: str) -> pd.DataFrame: ...
```

`load_csv()` cachea la lectura de `scoring_output.csv` — evita releer el CSV en cada interacción de filtro (que en Streamlit re-ejecuta el script completo). `fetch_price_history()` cachea por `condition_id` — evita volver a pegarle a Gamma+CLOB cada vez que el usuario reselecciona el mismo mercado. El botón "Actualizar scores" invalida manualmente todo el cache con `st.cache_data.clear()` antes de correr el scorer, para no mostrar datos viejos tras la actualización.

### 7.5 `.streamlit/config.toml`

```toml
[theme]
base = "dark"
primaryColor = "#3b82f6"
backgroundColor = "#0a0a0b"
secondaryBackgroundColor = "#141416"
textColor = "#fafafa"
font = "sans serif"
```

Coincide exactamente con las variables CSS de `dashboard.py` (`--accent`, `--bg`, `--surface`, `--text`) — es la configuración de theme nativo de Streamlit que se aplica a los widgets no-custom (sliders, radios, selectbox, botones), mientras que el CSS inline controla los componentes 100% custom (tabla, KPIs, badges).

---

## 8. Scoring pipeline (`score_markets.py`)

### 8.1 Cómo se ejecuta

Dos formas:
1. **Desde el dashboard**, como subproceso (`dashboard.py::run_scorer()`, ver §7.1) — el flujo normal para el usuario final.
2. **Standalone / CLI**, para uso manual o debugging:
   ```bash
   python score_markets.py               # top 50 mercados por prob_yes_gb (default)
   python score_markets.py --top 20      # top 20
   python score_markets.py --all         # todos, sin recortar
   python score_markets.py --out mi.csv  # archivo de salida distinto
   ```
No hay cron ni scheduler configurado en ningún lado del repo — el refresh es 100% manual (ver §4.6).

### 8.2 Flujo paso a paso

```python
def main():
    gb     = joblib.load(MODELS_DIR / "best_tree_model.pkl")
    lr     = joblib.load(MODELS_DIR / "best_lr.pkl")
    scaler = pickle.load(open(PROCESSED_DIR / "scaler.pkl", "rb"))

    markets = fetch_open_markets()              # Gamma API, activos, ≥7 días abiertos
    rows = []
    for m in markets:
        sd = parse_dt(m.get("startDate") or m.get("createdAt"))
        start_ts = int(sd.replace(tzinfo=timezone.utc).timestamp())
        history = fetch_prices(m["_yes_token_id"], start_ts)   # CLOB, ventana de 7 días
        feats = build_features(history, start_ts)              # None si < MIN_PTS=3 puntos
        if feats is None: continue
        cat = infer_category_coarse(m.get("question", ""))
        # ... arma dummies de categoría, agrega metadata
        rows.append(feats)
        time.sleep(CLOB_PAUSE)                                  # 0.15s entre requests CLOB

    df = pd.DataFrame(rows)
    df[NUM_COLS] = scaler.transform(df[NUM_COLS])
    df["prob_yes_gb"] = gb.predict_proba(df[ALL_FEATURES].values)[:, 1]
    df["prob_yes_lr"] = lr.predict_proba(df[ALL_FEATURES].values)[:, 1]
    df["pred_gb"] = (df["prob_yes_gb"] >= THRESHOLD).astype(int)   # 0.25
    df = df.sort_values("prob_yes_gb", ascending=False)
    result = df[out_cols].head(args.top) if not args.all else df[out_cols]
    result.to_csv(args.out, index=False)
```

### 8.3 Manejo de errores y logging

- Logging estándar (`logging.basicConfig(level=logging.INFO)`), formato `HH:MM:SS  LEVEL  mensaje`, todo a `stdout`.
- `fetch_open_markets()`: `r.raise_for_status()` sin try/except propio — un error de red acá aborta todo el script (comportamiento "fail fast", razonable para un script CLI corto).
- `fetch_prices()`: sí tiene try/except; en caso de error, loggea a nivel `debug` (no visible con el nivel `INFO` por defecto) y retorna lista vacía — el mercado se descarta silenciosamente más adelante por no tener suficientes puntos (`build_features` devuelve `None` si `n_pts < MIN_PTS`).
- No hay reintentos ni backoff exponencial en ningún request — a diferencia de lo que pedía `PROJECT_BRIEF.md` para la Fase 2 original ("manejar rate limits con backoff exponencial"), que sí aplicaba al descargador histórico (`download.py`) pero no se replicó en el scoring en vivo.
- Si `fetch_open_markets()` no devuelve nada, o si ningún mercado pasa el filtro de features válidas, el script loguea un warning y termina sin escribir CSV (`return` temprano) — el dashboard maneja esto mostrando el CSV previo (si existe) o un mensaje de "sin datos".

### 8.4 Tiempo estimado de ejecución

No hay ninguna medición explícita en el código ni en la documentación para el scoring de mercados **abiertos** (`score_markets.py`). Sí está documentado el tiempo de la descarga **histórica** completa (`src/data/download.py`): 14.7 minutos para 1,209 candidatos (`DECISIONES.md`, "Próximos 3 pasos" → Fase 2). Para `score_markets.py`, con `CLOB_PAUSE=0.15s` por mercado y typicamente decenas a un par de cientos de mercados abiertos candidatos, un estimado de orden de magnitud sería de **varias decenas de segundos a pocos minutos**, dominado por la latencia de red de cada request CLOB, no por la pausa fija — **por confirmar** con una medición real, no se encontró ningún log de tiempos de ejecución en el repo.

---

## 9. Deployment

- **Plataforma:** Streamlit Community Cloud, según el badge de `README.md` (`polymarket-scorer.streamlit.app`). **No verificado activamente en esta sesión** (no se hizo ningún request HTTP a esa URL) — se documenta la afirmación tal como aparece en `README.md`, no como un hecho confirmado en vivo.
- **Archivos de configuración relevantes:** `runtime.txt` (`3.11`, versión de Python que Streamlit Cloud usa para el contenedor) y `requirements.txt` (dependencias mínimas, ver §2.2). No hay ningún archivo `Procfile`, `Dockerfile` ni configuración de Streamlit Cloud adicional en el repo (por ejemplo, no hay `secrets.toml` — coherente con que el proyecto no necesita credenciales).
- **Variables de entorno / secretos:** ninguno. Todas las APIs consumidas (Gamma, CLOB) son públicas y sin autenticación.
- **Proceso de deploy — qué se dispara con cada push:** no hay ningún workflow de CI/CD en el repo (no existe `.github/`). El comportamiento típico de Streamlit Community Cloud es redeploy automático al detectar un push a la rama conectada (`master`), pero esto se configura desde el panel de Streamlit Cloud, **no desde el repo** — no hay forma de confirmarlo solo leyendo el código. **Por confirmar** en el dashboard de Streamlit Cloud si el auto-deploy está activo y a qué rama apunta.
- El pineo estricto de `scikit-learn`/`numpy`/Python 3.11 (§6.8) es, en la práctica, la pieza de configuración de deploy más crítica del repo: sin ella, el redeploy en Streamlit Cloud podría levantar un entorno con versiones distintas a las que se usaron para serializar `models/*.pkl`, rompiendo la carga de los modelos.

---

## 10. Convenciones del proyecto

### 10.1 Estilo de commits

Contenido **literal completo** de [`CLAUDE.md`](CLAUDE.md) (agregado en el mismo commit que el rediseño de UI, `4492036`, 2026-07-16):

```markdown
# Git commits

Never add `Co-Authored-By` trailers or Claude Code attribution to commits in this project. No "🤖 Generated with Claude Code" line, no links to claude.com/code, no mention that a commit was AI-assisted. Commit messages should read as if written solely by the repo owner.
```

Esta regla no es solo prospectiva: en el mismo período (entre el commit `052105f` del 2026-06-20 y el commit `4492036` del 2026-07-16) el historial de git fue **reescrito con `git filter-branch`** para limpiar co-autoría de Claude de commits ya existentes — ver detalle completo en §11 y §13. Es decir, la regla de `CLAUDE.md` se aplicó también retroactivamente al historial preexistente, no solo hacia adelante.

Estilo observado en los mensajes de commit reales (`git log`, todos en español, imperativo/descriptivo, sin body extenso en la mayoría):
```
Rediseño UI: paleta oscura, KPI cards, tabla custom, sidebar agrupado
Fix: pinear Python 3.11 y versiones sklearn/numpy para compatibilidad de modelos pkl
Agrega dashboard Streamlit y script de scoring de mercados abiertos
Reduce requirements.txt a dependencias minimas del dashboard
```
Título corto descriptivo del cambio; algunos usan prefijo tipo `Fix:`, la mayoría no usa ningún prefijo de convención (no es Conventional Commits estricto).

### 10.2 Estructura de branches

Dos branches locales: `master` (activa, la única con historial "limpio" post-`filter-branch`) y `backup-antes-de-limpiar` (creada como resguardo antes de la limpieza de historial). **Hallazgo relevante:** al momento de este documento, `backup-antes-de-limpiar` apunta al mismo commit (`052105f`) que la porción vieja de `master` — es decir, **no preserva efectivamente el historial pre-limpieza** pese a su nombre. Ver detalle en §11. Remoto único: `origin` → `github.com/TURRIvalentin/Data-Mining---Market-Closure-Prediction-`. No hay evidencia de un flujo de feature branches / PRs — todo el trabajo histórico se hizo directamente sobre `master` (incluyendo algunos `pull --rebase` visibles en el reflog, lo que sugiere que en algún momento se trabajó desde más de una máquina/checkout).

### 10.3 Formato de nombres de columnas en dataframes

Mezcla deliberada de español e inglés, sin una regla 100% consistente pero con un patrón claro:
- **Español, snake_case:** todas las features de dominio del TFI — `precio_dia_1`, `precio_fin`, `precio_tendencia`, `volatilidad_retornos`, `n_puntos_precio`, `category_coarse` (esta última mixta: nombre en inglés, valores en categorías en inglés también — `Sports`, `Finance`, etc.).
- **Inglés:** columnas que reflejan directamente el vocabulario de la API de Polymarket o de scikit-learn — `condition_id`, `question`, `start_date`, `outcome`, `split`, `bucket`.
- **Prefijo `cat_`** para las dummies one-hot de categoría (`cat_Crypto`, `cat_Sports`, etc.) — inglés + valor en inglés.
- Columnas de salida del dashboard (`scoring_output.csv`): inglés (`prob_yes_gb`, `pred_gb`, `condition_id`), salvo la columna calculada en el propio `dashboard.py` para UI, `señal` (con eñe, español, valores con emoji: `"✅ YES"` / `"NO"`).

La convención implícita parece ser: **español para conceptos de negocio/features inventadas por el proyecto, inglés para todo lo que viene directo de una librería o API externa** — no está escrita en ningún documento, es inferida de la consistencia observada en el código.

---

## 11. Deuda técnica y known issues

Ítems obligatorios reportados por el autor (verificados/contextualizados contra el código donde fue posible) + hallazgos propios de esta sesión.

### 11.1 Columna "Apertura" del dashboard — posible confusión de fecha (reportado por el autor, confirmado a nivel de código)

**Síntoma reportado:** la columna "Apertura" de la tabla principal muestra fechas como "2 may 2025" o "2 jul 2025" para mercados que la propia tabla lista como abiertos en julio 2026.

**Origen a nivel de código, confirmado:** la columna se renderiza en `dashboard.py` así:
```python
f"<td>{format_fecha_es(r['start_date'])}</td>"
```
y `start_date` se genera en `score_markets.py::main()`:
```python
sd = parse_dt(m.get("startDate") or m.get("createdAt"))
...
feats["start_date"] = sd.strftime("%Y-%m-%d")
```
Es decir, `start_date` es efectivamente la **fecha de apertura al trading** del mercado (`startDate`, con fallback a `createdAt` si falta) — la misma fecha que se usa como ancla de la ventana de observación de 7 días para las features. **No es un bug de que la columna muestre la fecha equivocada según su propio nombre** ("Apertura" = fecha de apertura, y eso es lo que efectivamente muestra) — el problema real es que **no hay ninguna columna que muestre la fecha de cierre/resolución esperada**, lo que puede hacer pensar al usuario que un mercado que abrió hace más de un año "recién abrió". La observación del autor de mercados de mayo/julio de 2025 apareciendo como "abiertos" en julio 2026 es coherente con mercados de larga duración reales (Polymarket tiene mercados que están abiertos durante más de un año), no necesariamente con un error de parseo de fecha.

**Acción sugerida:** renombrar la columna a "Apertura" → mantener, pero agregar una columna adicional "Cierra" o "Resolución esperada" si esa fecha está disponible en la respuesta de Gamma API (`endDate` o similar, no verificado en esta sesión si el campo existe y con qué nombre). Marcado explícitamente como **por confirmar**: no se auditó el JSON crudo de un mercado abierto real en esta sesión para confirmar el nombre exacto del campo de fecha de cierre esperado.

### 11.2 `git filter-branch` deprecado, usado en la limpieza de historial

Confirmado por reflog (`git reflog`, entrada `filter-branch: rewrite`) que el historial fue reescrito con `git filter-branch` en algún momento entre el 2026-07-13 23:27 (creación de la rama `backup-antes-de-limpiar`, ver §11.3) y el 2026-07-16 09:53 (commit `4492036`). `git filter-branch` está oficialmente deprecado por el propio proyecto Git en favor de [`git-filter-repo`](https://github.com/newren/git-filter-repo) desde hace varios años, por ser significativamente más lento y con más casos borde peligrosos (permisos de archivos, refs no estándar). **Si en el futuro hace falta reescribir historial de nuevo (por ejemplo, para limpiar más co-autoría, secretos filtrados, etc.), usar `git-filter-repo` en su lugar.**

### 11.3 Historial reescrito el 2026-07-16 — hashes de commits pre-`4492036` cambiaron

Confirmado con `git reflog` y `git log backup-antes-de-limpiar`: antes de la limpieza, el commit "Fix: pinear Python 3.11..." tenía hash `6bbfcf0`; después de `git filter-branch`, el mismo contenido (mismo autor, mismo mensaje, mismo timestamp de autoría `2026-06-20 21:47 -0300`) quedó con hash `052105f`. Lo mismo aplica en cascada a **todos** los commits anteriores (`be99812`→`53e44a2`, `ca7ebf5`→`b5e72cb`, `d31d8bc`→`2023998`, etc. — la correspondencia completa vieja→nueva no fue reconstruida commit por commit en esta sesión, solo confirmada en los extremos).

**Consecuencia práctica:** cualquier fork o clone local hecho *antes* del 2026-07-16 tiene una historia divergente e incompatible con el remoto actual — un `git pull` normal fallaría o generaría duplicados; haría falta un `git fetch` + `reset --hard origin/master` (destructivo, perdería commits locales no pusheados) para resincronizar. Cualquier link a un commit específico por hash publicado antes de esa fecha (por ejemplo, en `CONTEXT.md` viejo, en un PR ya cerrado, o compartido por chat) **ya no resuelve** a un commit real en el remoto.

**El branch `backup-antes-de-limpiar` no cumple su función:** el reflog muestra que se creó apuntando al estado pre-limpieza (`6bbfcf0`), pero al verificar en esta sesión (`git rev-parse backup-antes-de-limpiar`), el branch apunta hoy a `052105f` — **el mismo commit post-reescritura que el ancestro de `master`**, no al `6bbfcf0` original. El objeto `6bbfcf0` todavía existe físicamente en `.git` como commit suelto (`git cat-file -t 6bbfcf0` → `commit`), recuperable **solo mientras no corra un `git gc`** que lo pode por no ser alcanzable desde ninguna ref.

**Decisión tomada el 2026-07-16:** no recrear el branch de backup. Ya cumplió su función durante la operación de limpieza. El commit `6bbfcf0` queda documentado acá por si en el futuro se quisiera recuperar (sujeto a que Git no lo haya recolectado con `git gc`). Un rollback futuro debería usar `git-filter-repo` en lugar de `git filter-branch` (ver §11.2), y excluir explícitamente el branch de backup del reescribido para que sí preserve el estado anterior.

### 11.4 Contributor "claude" en la UI de GitHub tras reescribir el historial

Reportado por el autor: después de reescribir el historial para quitar trailers de co-autoría de Claude, GitHub sigue mostrando "claude" como contributor en la UI del repo. Comportamiento conocido de GitHub: la lista de contributors se cachea agresivamente y no siempre se recalcula inmediatamente tras un force-push que reescribe historia — puede tardar horas o requerir que GitHub re-indexe el repo. **No verificable desde este entorno** (sin acceso a la UI de GitHub en esta sesión). Acción sugerida por el propio autor: verificar en 48hs desde la limpieza.

### 11.5 Feature engineering duplicado entre entrenamiento e inferencia

`build_price_features()` en `src/data/make_dataset.py` y `build_features()` en `score_markets.py` son **casi idénticas línea por línea** (mismo algoritmo de forward/backward-fill, mismos agregados, misma fórmula de volatilidad) pero están escritas dos veces de forma independiente, no como una función compartida importada. Confirmado comparando ambos bloques de código en esta sesión — son equivalentes hoy, pero cualquier cambio futuro al feature engineering (ej. agregar una feature nueva, cambiar la lógica de imputación) tiene que replicarse manualmente en los dos archivos o se rompe la paridad train/serving sin ningún error visible (el modelo simplemente empezaría a recibir un feature vector con semántica distinta a la de entrenamiento). Refactor sugerido: extraer a un módulo compartido, por ejemplo `src/features/price_features.py`, importado por ambos.

### 11.6 Sin tests, sin CI

`tests/__init__.py` existe pero está vacío; no hay ningún `test_*.py` en el repo. No hay carpeta `.github/` ni ningún otro workflow de CI. La paridad train/serving mencionada en 11.5, el orden de columnas en `ALL_FEATURES` (§6.7), y la compatibilidad de versiones de sklearn/numpy (§6.8) son las tres piezas más frágiles del proyecto y ninguna tiene un test de regresión que las proteja — por ejemplo, un test mínimo que cargue `models/best_tree_model.pkl` y verifique que predice sobre un feature vector sintético sin excepción, o un test que compare el output de `build_price_features()` vs `build_features()` sobre el mismo input, cubrirían gran parte del riesgo real.

### 11.7 `matplotlib` usado pero no declarado en `requirements.txt`

Ver §2.2 — confirmado por grep de imports en `phase5.py`, `phase6.py`, `phase7.py`. No rompe el dashboard (esos scripts no se ejecutan en producción), pero rompe la reproducibilidad de "correr el pipeline de investigación desde cero" solo con `pip install -r requirements.txt`.

### 11.8 `dataset.xlsx` con propósito no documentado

Ver §4.4. Existe, está versionado, tiene contenido coherente con un export de análisis, pero ningún `.md` explica para qué se generó ni si sigue en uso. Riesgo bajo (no afecta al pipeline productivo) pero genera la pregunta obvia a cualquier lector nuevo del repo.

### 11.9 Posible categorización desactualizada en `dataset.xlsx`

Al inspeccionar `dataset.xlsx` en esta sesión, la primera fila (`"Will Netflix (NFLX) close above $180 end of April?"`) tiene `category_coarse = "Sports"`. Aplicando manualmente las reglas actuales de `src/features/categorization.py` v2 a ese texto, ninguna keyword de la lista `Sports` (`"nba"`, `"nfl "`, `" nfl)"`, etc.) hace match contra `"will netflix (nflx) close above $180 end of april?"` en minúsculas — el resultado esperado según el código actual sería `"Finance"` (por `"above $"`) o eventualmente `"Other"`. Esto sugiere que **la columna `category_coarse` de `dataset.xlsx` podría reflejar una corrida de categorización anterior a la v2 actual**, o bien un bug puntual no identificado. **No se investigó exhaustivamente** (no se re-corrió `infer_category_coarse()` programáticamente sobre las 965 filas de `dataset.xlsx` para cuantificar cuántas filas divergen) — marcado como hallazgo puntual, **por confirmar** con una verificación sistemática antes de asumir que es un problema generalizado.

### 11.10 `DECISIONES.md` no refleja el estado final del proyecto

Ver §2.4. La sección final de `DECISIONES.md` ("Próximos 3 pasos") quedó parada en "Fase 7 — siguiente", pese a que Fases 7 y 8 sí se completaron según `git log` (commits del 2026-05-05 al 2026-05-08). Quien lea solo `DECISIONES.md` de punta a punta se queda con la impresión de que el proyecto se cortó antes del análisis de resultados, cuando en realidad avanzó hasta el informe final (luego borrado) y el pivote a producto.

### 11.11 `src/visualization/` — paquete vacío

Solo contiene `__init__.py` vacío. El plan original (`PROJECT_BRIEF.md`) preveía funciones de plotting reutilizables ahí; en la práctica todo el código de matplotlib quedó inline y duplicado (con variaciones) dentro de `phase5.py`, `phase6.py` y `phase7.py`. No es un bug, pero es dead code estructural (un paquete que nunca se usó) que podría eliminarse o, mejor, poblarse si se retoma el proyecto.

### 11.12 Sin TODO/FIXME en el código, pero sin comentarios de limitaciones inline tampoco

Búsqueda de `TODO|FIXME|XXX|HACK` en todo el árbol `.py` del repo: **cero resultados**. Esto no significa ausencia de deuda técnica (ver todos los puntos anteriores) — significa que las limitaciones conocidas del proyecto viven como decisiones documentadas en `DECISIONES.md`, no como comentarios inline en el código, lo cual es en general una práctica más prolija pero implica que este `CONTEXT.md` (y `DECISIONES.md`) son las únicas fuentes de esa información — no alcanza con leer el código solo.

---

## 12. Cómo correr el proyecto en local

### 12.1 Dashboard (uso más simple)

```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

No requiere ninguna variable de entorno. Requiere que `models/best_lr.pkl`, `models/best_tree_model.pkl` y `data/processed/scaler.pkl` existan — **están versionados en el repo**, así que funciona out-of-the-box tras el clone, sin necesidad de re-entrenar ni re-descargar nada. Si `scoring_output.csv` no existe todavía (primera corrida, o clone limpio), el dashboard arranca con un mensaje "Sin datos. Presioná Actualizar scores" hasta la primera ejecución de `score_markets.py`.

### 12.2 Scoring manual de mercados abiertos

```bash
python score_markets.py               # top 50 por prob_yes_gb
python score_markets.py --top 20
python score_markets.py --all
python score_markets.py --out mi.csv
```

### 12.3 Reproducir el pipeline de investigación completo desde cero

```bash
conda create -n polymarket python=3.11
conda activate polymarket
pip install -r requirements.txt
pip install matplotlib   # no está en requirements.txt, ver §11.7 — necesario para phase5/6/7.py

python -m src.data.download --dry-run     # ver plan sin descargar
python -m src.data.download               # descarga completa (~15 min, requiere red)
python -m src.data.make_dataset            # features + split → data/processed/
                                            # (redundante si solo se quiere correr el dashboard:
                                            #  data/processed/*.parquet ya está versionado)
python -m src.models.phase5                # baselines + regresión logística
python -m src.models.phase6                # Random Forest + Gradient Boosting
python -m src.models.phase7                # análisis de resultados, calibración
```

No hay un `Makefile` ni script único que encadene todo el pipeline — el orden se infiere de los nombres de archivo y de `DECISIONES.md`.

**Datos necesarios:** `data/raw/` (crudo, gitignored) **no está en el repo** — si se quiere re-correr `make_dataset.py` desde cero hay que re-descargar con `src.data.download` (requiere red y ~15 minutos). Si el objetivo es solo correr el dashboard o reentrenar sobre el dataset ya procesado, **no hace falta `data/raw/`** en absoluto: `data/processed/{train,val,test}.parquet` ya están versionados y son suficientes como input directo de `phase5.py`/`phase6.py`.

### 12.4 Tests

```bash
# No hay tests que correr — tests/__init__.py está vacío (ver §11.6)
```

### 12.5 Dev Container / Codespaces

Abrir el repo en GitHub Codespaces (o VS Code con la extensión Dev Containers) usando `.devcontainer/devcontainer.json` levanta automáticamente el dashboard en el puerto 8501 sin pasos manuales — ver §2.2.

---

## 13. Historial de decisiones importantes

### 13.1 Cronología de sesiones de trabajo (reconstruida de `git log` + `git reflog`, fechas y horas reales del repo, offset `-03:00`)

| Fecha | Commit(s) | Qué se hizo |
|---|---|---|
| 2026-05-02 13:47 | `32b85c3` | Pre-diseño cerrado: brief, decisiones iniciales de scope. Inicio formal del proyecto (coincide con `PROGRESS.md` Sesión 1). |
| 2026-05-05 21:13 | `1a7cc57` | **Commit único y grande**: Fases 0 a 6 completadas — descarga, EDA, feature engineering, modelos baseline y avanzados. Todo el trabajo de investigación central del TFI quedó comprimido en un solo commit (no hay granularidad por fase en el historial real, pese a que `DECISIONES.md` documenta las fases por separado con fechas propias como 2026-05-03/04). |
| 2026-05-05 22:14 – 23:34 | `2023998`, `93d5d15` | Fase 7 + 8: análisis final, informe completo en markdown, luego convertido a Word. |
| 2026-05-05 23:34 – 2026-05-07 23:43 | `c290a29`, `c4efa9f`, `9e4be89` | Rondas de corrección del informe: ajuste de conteo de features, revisión de la hipótesis de drift de marzo, sección 7.8 nueva, re-ejecución del notebook de EDA con categorización v2. |
| 2026-05-08 14:17 – 14:18 | `6a57a65`, `0facb83` | Se borran `reports/informe_final.docx` y `reports/informe_final.md` del repo — el informe final deja de existir en el árbol de archivos (ver §1, §2.4). |
| 2026-05-09 10:58 – 11:43 | `b5e72cb`, `53e44a2`, `1b5e650`, `f65178d` | **Pivote de "informe académico" a "producto"**: se agrega `dashboard.py` + `score_markets.py`; se reduce `requirements.txt` a lo mínimo para el dashboard; se sacan versiones fijas (más tarde revertido parcialmente, ver siguiente fila); se agrega el badge del dashboard al `README.md`. |
| 2026-06-05 09:24 | `7901c0a` | Se agrega `.devcontainer/` (Dev Container / Codespaces). |
| 2026-06-20 21:47 | `052105f` (hash original antes de la reescritura: `6bbfcf0`) | Se pinean `scikit-learn==1.8.0`, `numpy==2.4.4` y Python 3.11 explícitamente — fix de compatibilidad de los `.pkl` de modelos en el entorno de deploy (ver §6.8). |
| 2026-07-13 23:27 | (sin commit — `reset` en el reflog) | Se crea el branch `backup-antes-de-limpiar` apuntando al estado pre-limpieza del historial (aunque, como se documenta en §11.3, hoy ya no cumple esa función). |
| entre 2026-07-13 23:27 y 2026-07-16 09:53 | (`git filter-branch: rewrite` en el reflog, sin commit propio ni fecha de pared exacta) | **Reescritura de historial** con `git filter-branch` para limpiar trailers de co-autoría de Claude de commits preexistentes (ver §11.2, §11.3). Fecha exacta de esta operación **por confirmar** — el reflog no expone un timestamp de pared confiable para esta entrada específica. |
| 2026-07-16 09:53 | `4492036` | **Rediseño de UI** (paleta oscura, KPI cards, tabla HTML custom con badges y barras de probabilidad, sidebar agrupado en 2 secciones, `html.escape()` sobre texto de la API, footer) y, en el mismo commit, **creación de `CLAUDE.md`** con la regla de no incluir co-autoría de Claude en commits — ambos cambios llegaron juntos, consistente con que la limpieza de historial y la nueva regla de commits son parte de la misma iniciativa. |

### 13.2 Rediseño de UI (2026-07-16)

Un único commit (`4492036`), no un proceso incremental de varios commits — el mensaje de commit lista explícitamente 8 cambios aplicados juntos:
1. Paleta dark base con acento azul `#3b82f6` (`.streamlit/config.toml` + variables CSS).
2. Fuente Inter vía Google Fonts, jerarquía de título/subtítulo.
3. KPIs como cards con íconos Lucide SVG inline y valores tabulares.
4. Tabla HTML custom: badges de señal/categoría, barras de probabilidad con gradiente por magnitud, fechas en español, header sticky, hover sutil.
5. `st.selectbox` para elegir mercado (reemplaza la selección nativa por click de `st.dataframe`, que el proyecto dejó de usar).
6. Sidebar reorganizado en 2 secciones ("Acciones"/"Filtros") con separadores visuales.
7. Footer con info del modelo y versión.
8. `html.escape()` aplicado a campos de texto provenientes de la API (mitigación de HTML injection, ver §7.2).

Diff real: `+336 / -51` líneas sobre 4 archivos (`dashboard.py`, `.streamlit/config.toml`, `.gitignore`, `CLAUDE.md`). No hay evidencia en el repo de que este rediseño se haya planificado o ejecutado en "4 pasos" separados — es un solo commit atómico.

### 13.3 Migración/pineo de Python 3.11 y sklearn/numpy

Ver §6.8 y fila correspondiente en §13.1. Motivación inferida del propio mensaje de commit: los `.pkl` de `models/` se serializaron con una versión específica de scikit-learn/numpy sobre Python 3.11 (2026-05-04), y sin pinear esas versiones en `requirements.txt`/`runtime.txt`, el entorno de deploy de Streamlit Cloud podía resolver versiones distintas y romper la carga de los modelos. No hay in-code comment explicando el *por qué específico* más allá del mensaje de commit "Fix: pinear Python 3.11 y versiones sklearn/numpy para compatibilidad de modelos pkl" — se documenta tal cual, sin inventar detalle adicional no verificable (por ejemplo, no se pudo determinar en esta sesión cuál fue el error exacto de deploy que motivó el fix, si lo hubo).

### 13.4 Reescritura de historial para limpiar co-autoría de Claude

Ver §11.2, §11.3, §13.1. Es la decisión más reciente y la más delicada del repo desde el punto de vista de integridad de historial. Vale remarcar: la reescritura logró su objetivo funcional en `master` (los commits visibles hoy no tienen trailers de Claude), pero el mecanismo de backup pensado para resguardar el historial viejo (`backup-antes-de-limpiar`) no quedó apuntando donde debería — es una limpieza "casi completa" con un cabo suelto identificado en esta sesión.

---

## 14. Glosario

### 14.1 Términos de dominio (Polymarket / mercados de predicción)

- **Mercado de predicción (prediction market):** plataforma donde se puede comprar/vender contratos ("shares") cuyo valor final depende del resultado de un evento futuro incierto.
- **Mercado binario:** mercado con exactamente dos resultados posibles mutuamente excluyentes (SÍ/NO). Es el único tipo de mercado que este proyecto modela.
- **Token YES / NO shares:** en Polymarket, cada mercado binario emite dos tokens (YES y NO) que en conjunto valen siempre \$1 al resolver (uno vale \$1, el otro \$0). El **precio del token YES** (entre 0 y 1) es interpretable como la probabilidad implícita que el mercado le asigna a que el evento ocurra.
- **`condition_id` / `conditionId`:** identificador único de un mercado en el protocolo de Polymarket (basado en Gnosis Conditional Tokens Framework). Es la clave primaria usada en todo el pipeline de este proyecto (nombre de archivo en `data/raw/markets/` y `data/raw/prices/`, columna `condition_id` en los parquet).
- **`clobTokenIds`:** array con los IDs de los tokens YES/NO específicos del CLOB (Central Limit Order Book), necesarios para pedir el historial de precios de cada lado del mercado por separado. El proyecto siempre usa `clobTokenIds[0]` (el token YES).
- **Gamma API:** API pública de Polymarket para metadata de mercados/eventos (preguntas, fechas, outcomes, volumen). No es donde se ejecutan órdenes.
- **CLOB (Central Limit Order Book) API:** API pública de Polymarket para datos de trading (historial de precios, libro de órdenes, trades). El proyecto solo usa `/prices-history` de este API (sin autenticación); `/trades` requiere API key y fue descartado como fuente de features.
- **`outcomePrices`:** campo de la respuesta de Gamma API con el precio final de cada outcome tras la resolución — `["1","0"]` significa que YES ganó, `["0","1"]` que NO ganó, un valor intermedio indica mercado cancelado/N-A.
- **`resolution date` / fecha de resolución:** fecha en la que el mercado se cierra y se determina el outcome final. Distinta de `startDate` (apertura al trading) — ver §11.1 sobre la confusión potencial de estas fechas en la UI del dashboard.
- **USDC en Polygon:** la moneda con la que se opera en Polymarket (stablecoin USDC sobre la red Polygon). No es relevante para el pipeline de ML de este proyecto salvo como contexto de dominio — el proyecto no interactúa con wallets ni moneda real (eso sí ocurre en `polymarket-bot/`, subproyecto separado, donde `POLYMARKET_PRIVATE_KEY` en su `.env` sugiere capacidad de firmar transacciones reales — tratar esas credenciales con el mismo cuidado que cualquier clave de wallet).

### 14.2 Términos específicos del proyecto

- **`category_coarse`:** la categoría heurística de 7 valores (`Sports`, `Politics`, `Crypto`, `Finance`, `Tech`, `Entertainment`, `Other`) asignada a cada mercado por keyword matching sobre el texto de la pregunta (§5.2).
- **"Señal GB" / `pred_gb`:** la predicción binaria (0/1) del modelo Gradient Boosting tras aplicar el threshold de 0.25 a `prob_yes_gb`. Es la señal principal mostrada en el dashboard.
- **`prob_yes_gb_pct` / `prob_yes_lr_pct`:** las probabilidades de GB/LR expresadas como porcentaje con 1 decimal, calculadas en `dashboard.py::load_csv()` a partir de `prob_yes_gb`/`prob_yes_lr` del CSV (que están en escala 0-1).
- **"Champion model" (GB):** el modelo `GradientBoostingClassifier` serializado en `models/best_tree_model.pkl`, elegido como el mejor modelo general en Fase 6 por AUC + calibración (§6.3, §6.5). "LR-C" es el nombre interno del segundo modelo productivizado (regresión logística L1, §6.2).
- **Threshold 0.25:** el punto de corte de probabilidad aplicado a `prob_yes_gb`/`prob_yes_lr` para convertir la probabilidad continua en señal binaria YES/NO, derivado en Fase 7 maximizando F1-macro sobre el test set y reutilizado literalmente en producción (§6.6).
- **B1, B2, B3:** los tres baselines de Fase 5 — B1 (mayoría de clase), B2 (prior de clase), B3 (`precio_fin` tal cual, sin modelo) — usados como piso de comparación para todos los modelos entrenados.
- **Bucket temporal:** una de las 4 particiones (`pre-2026`, `2026-01`, `2026-02`, `2026-03+`) usadas para el split estratificado por fecha (§5.1, punto 5).
- **`n_puntos_precio`:** cantidad de observaciones reales de precio disponibles en la ventana de 7 días de un mercado (mínimo 3 para incluirlo en el dataset) — no confundir con los 7 valores de `precio_dia_1..7`, que siempre son 7 (con imputación) independientemente de `n_puntos_precio`.
- **`ALL_FEATURES` / `ALL_FEATURE_COLS`:** el nombre de la lista con las 22 columnas de features en su orden canónico, definida (duplicada, ver §11.5) en `make_dataset.py` y `score_markets.py`.

---

## 15. Cómo usar este documento

**Para un lector nuevo (humano o IA) que necesita entender el proyecto de punta a punta:** leer este documento completo primero (§1 a §14); después, si necesita el detalle metodológico fino de por qué se tomó cada decisión de modelado (no solo qué se decidió), leer `DECISIONES.md` completo — este documento resume sus conclusiones pero no reemplaza sus 15 secciones de razonamiento. Para el contrato original del TFI (qué pedía la especialización), leer `PROJECT_BRIEF.md`.

**Para alguien que va a modificar código:**
- Si va a tocar feature engineering, splits o modelos: leer §5, §6 y `DECISIONES.md` §1-6 y §11-15 antes de cambiar nada — hay decisiones no obvias (por qué el split no es cronológico estricto, por qué `log_volumen_total` se descartó) que un cambio ingenuo podría revertir sin darse cuenta.
- Si va a tocar `score_markets.py` o `make_dataset.py`: recordar la duplicación de feature engineering (§11.5) — cualquier cambio a la lógica de features tiene que replicarse en ambos archivos.
- Si va a tocar el dashboard: los estilos viven inline en `dashboard.py` (no hay archivo `.css` separado) — buscar el bloque `st.markdown("""<style>...""")` cerca de la línea 62.

**Cómo actualizar este documento cuando cambien cosas:** no regenerar desde cero salvo que haya pasado mucho tiempo o el repo haya cambiado sustancialmente — mejor, editar la sección puntual afectada y actualizar la fecha de generación en el encabezado. Si se reentrenan los modelos, actualizar §6 (métricas, hiperparámetros, hashes de archivos `.pkl`) y §6.8 (fecha de serialización). Si se agregan features, actualizar §5.3/§5.4 y verificar que la tabla de no-leakage en `make_dataset.py` se actualizó también. Si se hace otro rewrite de historial de git, actualizar §11.2/§11.3/§13.1 con las fechas y hashes nuevos, y **revisar si `backup-antes-de-limpiar` (o su reemplazo) apunta donde debería** antes de asumir que el backup es válido.

---

## 16. Inconsistencias detectadas durante la generación

1. **Hashes de commit en la versión anterior de `CONTEXT.md`:** citaba hashes (`6bbfcf0`, `be99812`, `ca7ebf5`, `d31d8bc`, `f75142b`, `5b7a423`, `1e0b984`, `be640e6`, `eed67a0`, etc.) que eran reales y verificables al momento en que ese documento se escribió (2026-07-04), pero que dejaron de ser alcanzables tras la reescritura de historial vía `git filter-branch` ocurrida entre el 2026-07-13 y el 2026-07-16 (ver §11.2, §11.3). No es una fabricación del documento viejo — es staleness causada por un evento posterior. Este documento usa exclusivamente los hashes reales de `git log` al momento de esta generación (2026-07-16).
2. **"Rediseño de UI en 4 pasos":** una instrucción previa de trabajo asumía que el rediseño de UI se había hecho en 4 commits/pasos separados. `git log` muestra que es **un solo commit** (`4492036`) con 8 cambios listados en su mensaje. Se documentó según lo que el propio commit dice, no según la premisa de "4 pasos" (ver §13.2).
3. **Columnas de `dataset.xlsx` descartadas del feature set final:** `dataset.xlsx` incluye `log_volumen_total`, `duration_days` y `cluster`, que `DECISIONES.md` y el código de `make_dataset.py` documentan explícitamente como excluidas del modelo por leakage o por ser subproducto exploratorio. El archivo Excel es un export de análisis, no el dataset de entrenamiento real (ver §4.4).
4. **Categorización potencialmente desactualizada en `dataset.xlsx`:** al menos un caso concreto (mercado de NFLX) tiene `category_coarse = "Sports"` en `dataset.xlsx` cuando las reglas actuales de `categorization.py` v2 no producirían ese resultado para ese texto. No se cuantificó cuántas filas del total (965) tienen esta discrepancia — marcado como hallazgo puntual, no como problema sistémico confirmado (ver §11.9).
5. **`CONTEXT.md` estaba gitignored antes de esta actualización:** la versión anterior de este documento nunca se versionó (`.gitignore` la excluía explícitamente); esta versión sí se versiona, por instrucción explícita del autor (ver §2.4, §3.2).
6. **Discrepancia de conteo del dataset:** `DECISIONES.md` reporta 967 mercados post-filtro `MIN_PRICE_POINTS`; la suma real de `train+val+test` en `split_stats.json` es 965. Diferencia de 2 registros no explicada por el código auditado en esta sesión (ver §4.3).
7. **`backup-antes-de-limpiar` no preserva el historial que su nombre promete:** creado aparentemente con la intención de resguardar el estado pre-limpieza del repo, pero al momento de esta generación apunta al mismo commit que la porción post-reescritura de `master`, no al commit `6bbfcf0` original (ver §11.3). Esto no es una inconsistencia entre documentos, sino un hallazgo directo sobre el estado del repo que vale la pena que el autor corrija.
8. **`DECISIONES.md` documentado como "vivo" pero con su última sección desactualizada:** su propio encabezado lo describe como "documento vivo", pero la sección final ("Próximos 3 pasos") no se actualizó después de completarse las Fases 7 y 8 (ver §11.10).

