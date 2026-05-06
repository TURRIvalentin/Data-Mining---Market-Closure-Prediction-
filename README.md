# Predicción temprana de outcomes en Polymarket

Trabajo Final Integrador — Especialización en Explotación de Datos y Descubrimiento del Conocimiento (UBA)

## Pregunta de investigación

¿Es posible predecir el outcome final (SÍ/NO) de un mercado binario de Polymarket con accuracy significativamente mejor que la probabilidad implícita del mercado, utilizando únicamente información de sus primeros 7 días de actividad?

## Estructura del proyecto

```
trabajo_final_polymarket/
├── data/
│   ├── raw/          # Datos crudos de la API — NO modificar, NO commitear
│   ├── interim/      # Datos limpios intermedios
│   └── processed/    # Features finales para modelado
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modeling_baseline.ipynb
│   ├── 05_modeling_advanced.ipynb
│   └── 06_results_and_figures.ipynb
├── src/
│   ├── config.py         # Seeds y constantes globales
│   ├── data/             # Scripts de descarga y limpieza
│   ├── features/         # Feature engineering
│   ├── models/           # Entrenamiento y evaluación
│   └── visualization/    # Funciones de plotting reutilizables
├── reports/
│   ├── figures/          # Figuras finales para el informe
│   └── informe_final.md
├── tests/
├── DECISIONES.md         # Decisiones de scope y metodológicas documentadas
├── PROGRESS.md           # Bitácora de sesiones de trabajo
└── requirements.txt
```

## Setup del entorno

```bash
# Crear entorno conda con Python 3.11
conda create -n polymarket python=3.11
conda activate polymarket

# Instalar dependencias
pip install -r requirements.txt

# Registrar kernel en Jupyter
python -m ipykernel install --user --name polymarket --display-name "Python (polymarket)"
```

## Reproducibilidad

- Todas las semillas aleatorias centralizadas en `src/config.py` (`RANDOM_SEED = 42`).
- Los datos crudos se descargan con los scripts de `src/data/` y se guardan en `data/raw/` con timestamp.
- `data/raw/` está excluido del repositorio (ver `.gitignore`).

## Período de datos

Mercados binarios resueltos de Polymarket: **2024-01-01 hasta la fecha de descarga**.
Filtros: duración ≥ 30 días, mercados no cancelados/N/A.

## Decisiones de diseño

Ver [DECISIONES.md](DECISIONES.md) para el registro completo de decisiones metodológicas, riesgos y justificaciones.
