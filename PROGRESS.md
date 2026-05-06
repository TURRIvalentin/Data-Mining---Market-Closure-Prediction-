# PROGRESS — Bitácora del proyecto

> Actualizar al final de cada sesión de trabajo.

---

## Sesión 1 — 2026-05-02

**Fase:** 0 — Setup

**Completado:**
- Definición completa del scope y decisiones de diseño (ver `DECISIONES.md`)
- Investigación de APIs de Polymarket: Gamma API y CLOB API
- Estructura de carpetas creada
- `.gitignore`, `requirements.txt`, `README.md` iniciales
- `src/config.py` con seeds y constantes globales centralizadas

**Hallazgos relevantes:**
- El endpoint `GET /markets` de Gamma API tiene header `Deprecation: true`. Investigar el endpoint actual en Fase 1 antes de escribir código de descarga masiva.
- El endpoint `/prices-history` de CLOB API solo devuelve granularidad ≥12h para mercados resueltos. Se usará granularidad diaria (fidelity=1440).
- Python disponible en el sistema: 3.9 (standalone) y **3.11.5 (miniconda)** — usar miniconda para el proyecto.

**Advertencias / decisiones pendientes:**
- Ninguna. El pre-diseño está cerrado.

**Próximo paso:**
- Fase 1: investigar endpoint actual que reemplaza `/markets` deprecado. Hacer requests manuales exploratorios con 20-30 mercados para entender la estructura de datos antes de escribir la descarga masiva.

---
