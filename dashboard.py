"""
Dashboard de scoring — Polymarket Prediction Tool
Correr con: streamlit run dashboard.py
"""

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────
SCORING_CSV  = Path("scoring_output.csv")
CLOB_API     = "https://clob.polymarket.com"
GAMMA_API    = "https://gamma-api.polymarket.com"
THRESHOLD    = 0.25
OBS_DAYS     = 7

st.set_page_config(
    page_title="Polymarket Scorer",
    page_icon="📊",
    layout="wide",
)

# ── Helpers ───────────────────────────────────────────────────────────────────
def run_scorer(top: int = 200):
    """Ejecuta score_markets.py y recarga el CSV."""
    with st.spinner("Descargando mercados y calculando scores..."):
        result = subprocess.run(
            [sys.executable, "score_markets.py", "--top", str(top)],
            capture_output=True, text=True
        )
    if result.returncode != 0:
        st.session_state["scorer_error"] = result.stderr[-2000:]
    else:
        st.session_state.pop("scorer_error", None)
        st.rerun()


@st.cache_data(ttl=300)
def load_csv() -> pd.DataFrame:
    if not SCORING_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(SCORING_CSV)
    df["prob_yes_gb_pct"] = (df["prob_yes_gb"] * 100).round(1)
    df["prob_yes_lr_pct"] = (df["prob_yes_lr"] * 100).round(1)
    df["señal"] = df["pred_gb"].map({1: "✅ YES", 0: "NO"})
    return df


@st.cache_data(ttl=600)
def fetch_price_history(condition_id: str) -> pd.DataFrame:
    """Trae el historial completo de precios del token YES."""
    try:
        # Buscar clobTokenId via Gamma API
        r = requests.get(f"{GAMMA_API}/markets",
                         params={"conditionId": condition_id}, timeout=15)
        markets = r.json()
        if not markets:
            return pd.DataFrame()
        m = markets[0] if isinstance(markets, list) else markets
        clob_ids = m.get("clobTokenIds", [])
        if isinstance(clob_ids, str):
            clob_ids = json.loads(clob_ids)
        if not clob_ids:
            return pd.DataFrame()

        r2 = requests.get(f"{CLOB_API}/prices-history",
                          params={"market": clob_ids[0], "fidelity": 1440},
                          timeout=20)
        history = r2.json().get("history", [])
        if not history:
            return pd.DataFrame()

        df = pd.DataFrame(history)
        df["fecha"] = pd.to_datetime(df["t"], unit="s", utc=True)
        df = df.rename(columns={"p": "precio_yes"})
        return df[["fecha", "precio_yes"]].sort_values("fecha")
    except Exception:
        return pd.DataFrame()


# ── Layout ────────────────────────────────────────────────────────────────────
st.title("📊 Polymarket — Scorer de mercados")

if "scorer_error" in st.session_state:
    st.error(f"Error al ejecutar scorer:\n```\n{st.session_state['scorer_error']}\n```")

# Sidebar
with st.sidebar:
    st.header("Controles")

    if st.button("🔄 Actualizar scores", use_container_width=True):
        st.cache_data.clear()
        run_scorer(top=200)

    if SCORING_CSV.exists():
        mtime = datetime.fromtimestamp(SCORING_CSV.stat().st_mtime)
        st.caption(f"Última actualización: {mtime.strftime('%d/%m/%Y %H:%M')}")
    else:
        st.info("No hay datos. Presioná Actualizar para descargar mercados.")

    st.divider()
    st.header("Filtros")

    df_full = load_csv()
    if not df_full.empty:
        categorias = ["Todas"] + sorted(df_full["category"].dropna().unique().tolist())
        cat_sel = st.selectbox("Categoría", categorias)

        señal_sel = st.radio("Señal GB", ["Todas", "✅ YES", "NO"], horizontal=True)

        min_prob = st.slider("Prob. mínima GB (%)", 0, 100, 0, step=5)

        st.divider()
        st.caption(f"Umbral del modelo: **{THRESHOLD}**")
        st.caption("Modelos: GB (champion) · LR-C")

# ── Tabla principal ───────────────────────────────────────────────────────────
df_full = load_csv()

if df_full.empty:
    st.warning("Sin datos. Presioná **Actualizar scores** en el panel izquierdo.")
    st.stop()

# Aplicar filtros
df = df_full.copy()
if cat_sel != "Todas":
    df = df[df["category"] == cat_sel]
if señal_sel != "Todas":
    df = df[df["señal"] == señal_sel]
df = df[df["prob_yes_gb_pct"] >= min_prob]

# KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("Mercados mostrados", len(df))
col2.metric("Señales YES (GB)", int(df["pred_gb"].sum()))
col3.metric("Prob. media GB", f"{df['prob_yes_gb_pct'].mean():.1f}%")
col4.metric("Prob. máx GB", f"{df['prob_yes_gb_pct'].max():.1f}%")

st.divider()

# Tabla
display_cols = {
    "señal":           "Señal",
    "prob_yes_gb_pct": "Prob GB (%)",
    "prob_yes_lr_pct": "Prob LR (%)",
    "category":        "Categoría",
    "start_date":      "Apertura",
    "question":        "Pregunta",
}

df_show = (
    df[list(display_cols.keys())]
    .rename(columns=display_cols)
    .sort_values("Prob GB (%)", ascending=False)
    .reset_index(drop=True)
)

selection = st.dataframe(
    df_show,
    use_container_width=True,
    hide_index=True,
    on_select="rerun",
    selection_mode="single-row",
    column_config={
        "Prob GB (%)": st.column_config.ProgressColumn(
            "Prob GB (%)", min_value=0, max_value=100, format="%.1f%%"
        ),
        "Prob LR (%)": st.column_config.ProgressColumn(
            "Prob LR (%)", min_value=0, max_value=100, format="%.1f%%"
        ),
    },
)

# ── Detalle del mercado seleccionado ─────────────────────────────────────────
selected_rows = selection.selection.rows if selection.selection else []

if selected_rows:
    idx = df_show.index[selected_rows[0]]
    row = df.iloc[idx]

    st.divider()
    st.subheader(row["question"])

    info_cols = st.columns(4)
    info_cols[0].metric("Señal GB", row["señal"])
    info_cols[1].metric("Prob GB", f"{row['prob_yes_gb_pct']:.1f}%")
    info_cols[2].metric("Prob LR", f"{row['prob_yes_lr_pct']:.1f}%")
    info_cols[3].metric("Categoría", row["category"])

    # Precio histórico
    with st.spinner("Cargando historial de precios..."):
        hist = fetch_price_history(row["condition_id"])

    if not hist.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hist["fecha"], y=hist["precio_yes"],
            mode="lines+markers",
            name="Precio YES",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=5),
        ))

        # Línea de threshold
        fig.add_hline(
            y=THRESHOLD, line_dash="dash", line_color="orange",
            annotation_text=f"Umbral {THRESHOLD}",
            annotation_position="bottom right"
        )

        # Ventana de observación (primeros 7 días)
        if not hist.empty:
            t0 = hist["fecha"].min()
            t7 = t0 + pd.Timedelta(days=OBS_DAYS)
            fig.add_vrect(
                x0=t0, x1=t7,
                fillcolor="green", opacity=0.07,
                annotation_text="Ventana 7d", annotation_position="top left"
            )

        fig.update_layout(
            title="Historial de precio token YES",
            yaxis_title="Precio (implica prob. de resolución YES)",
            yaxis=dict(range=[0, 1]),
            xaxis_title="Fecha",
            height=380,
            margin=dict(t=50, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No hay datos de precio disponibles para este mercado.")

    st.caption(f"condition_id: `{row['condition_id']}`")
