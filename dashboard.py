"""
Dashboard de scoring — Polymarket Prediction Tool
Correr con: streamlit run dashboard.py
"""

import html
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

CATEGORY_COLORS = {
    "Sports":         "#06b6d4",
    "Politics":       "#a855f7",
    "Crypto":         "#f97316",
    "Entertainment":  "#ec4899",
}
CATEGORY_COLOR_DEFAULT = "#71717a"

MESES_ES = {
    1: "ene", 2: "feb", 3: "mar", 4: "abr", 5: "may", 6: "jun",
    7: "jul", 8: "ago", 9: "sep", 10: "oct", 11: "nov", 12: "dic",
}

SEÑAL_FILTRO_MAP = {"Todas": None, "YES": "✅ YES", "NO": "NO"}

st.set_page_config(
    page_title="Polymarket Scorer",
    page_icon="📊",
    layout="wide",
)

# ── Iconos (Lucide, inline SVG) ─────────────────────────────────────────────────
def lucide(paths: str, size: int = 18) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round">{paths}</svg>'
    )

ICON_BAR_CHART = lucide('<path d="M3 3v18h18"/><path d="M18 17V9"/><path d="M13 17V5"/><path d="M8 17v-3"/>')
ICON_CHECK     = lucide('<path d="M21.801 10A10 10 0 1 1 17 3.335"/><path d="m9 11 3 3L22 4"/>')
ICON_TRENDING  = lucide('<polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/>')
ICON_TARGET    = lucide('<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>')
ICON_CLOCK     = lucide('<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>', size=14)

# ── Estilos base ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --bg: #0a0a0b;
    --surface: #141416;
    --border: #27272a;
    --text: #fafafa;
    --text-muted: #a1a1a6;
    --accent: #3b82f6;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.app-title {
    font-size: 1.9rem;
    font-weight: 700;
    letter-spacing: -0.01em;
    margin-bottom: 2px;
}
.app-subtitle {
    font-size: 0.9rem;
    color: var(--text-muted);
    margin-bottom: 1.4rem;
}

/* KPI cards */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
    margin-bottom: 1.2rem;
}
.kpi-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 22px 24px;
}
.kpi-card .kpi-icon {
    color: var(--text-muted);
    margin-bottom: 10px;
}
.kpi-card .kpi-label {
    font-size: 0.8rem;
    font-weight: 500;
    color: var(--text-muted);
    margin-bottom: 6px;
}
.kpi-card .kpi-value {
    font-size: 1.9rem;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
    color: var(--text);
}

/* Tabla */
.mkt-table-wrap {
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: auto;
    max-height: 560px;
}
.mkt-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
}
.mkt-table thead th {
    position: sticky;
    top: 0;
    background: #0f0f11;
    text-align: left;
    padding: 10px 14px;
    font-weight: 600;
    color: var(--text-muted);
    border-bottom: 1px solid var(--border);
    z-index: 1;
}
.mkt-table tbody td {
    padding: 10px 14px;
    border-bottom: 1px solid var(--border);
    vertical-align: middle;
}
.mkt-table tbody tr:nth-child(even) {
    background: rgba(255, 255, 255, 0.015);
}
.mkt-table tbody tr:hover {
    background: rgba(255, 255, 255, 0.03);
    transition: background 150ms;
}

.badge {
    display: inline-block;
    min-width: 56px;
    text-align: center;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
}
.badge-yes { color: #22c55e; background: #22c55e26; }
.badge-no  { color: #a1a1a6; background: #71717a26; }

.prob-bar-wrap {
    display: flex;
    align-items: center;
    gap: 8px;
}
.prob-bar-track {
    flex: 1;
    height: 7px;
    border-radius: 999px;
    background: var(--border);
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 999px;
}
.prob-bar-value {
    font-variant-numeric: tabular-nums;
    font-size: 0.85rem;
    min-width: 42px;
    text-align: right;
}

/* Sidebar */
.sidebar-section-title {
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--text-muted);
    border-top: 1px solid var(--border);
    padding-top: 14px;
    margin-top: 14px;
    margin-bottom: 10px;
}
.sidebar-section-title.first {
    border-top: none;
    padding-top: 0;
    margin-top: 0;
}
.sidebar-timestamp {
    display: flex;
    align-items: center;
    gap: 6px;
    color: var(--text-muted);
    font-size: 0.8rem;
    margin-top: 6px;
}

/* Footer */
.app-footer {
    display: flex;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 6px;
    border-top: 1px solid var(--border);
    padding-top: 20px;
    margin-top: 2.5rem;
    color: var(--text-muted);
    font-size: 0.8rem;
}
</style>
""", unsafe_allow_html=True)

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


def format_fecha_es(valor) -> str:
    try:
        dt = pd.to_datetime(valor)
        return f"{dt.day} {MESES_ES[dt.month]} {dt.year}"
    except Exception:
        return str(valor)


def badge_signal(pred_gb) -> str:
    if int(pred_gb) == 1:
        return '<span class="badge badge-yes">YES</span>'
    return '<span class="badge badge-no">NO</span>'


def badge_category(categoria) -> str:
    color = CATEGORY_COLORS.get(categoria, CATEGORY_COLOR_DEFAULT)
    label = html.escape(str(categoria))
    return f'<span class="badge" style="color:{color};background:{color}26;">{label}</span>'


def prob_bar(valor) -> str:
    valor = float(valor)
    if valor < 33:
        color = "#64748b"
    elif valor < 66:
        color = "#f59e0b"
    else:
        color = "#22c55e"
    return (
        '<div class="prob-bar-wrap">'
        f'<div class="prob-bar-track"><div class="prob-bar-fill" style="width:{valor}%;background:{color};"></div></div>'
        f'<span class="prob-bar-value">{valor:.1f}%</span>'
        '</div>'
    )


# ── Layout ────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="app-title">📊 Polymarket Scorer</div>'
    '<div class="app-subtitle">Ranking de mercados por señal de modelo — Gradient Boosting &amp; Logistic Regression</div>',
    unsafe_allow_html=True,
)

if "scorer_error" in st.session_state:
    st.error(f"Error al ejecutar scorer:\n```\n{st.session_state['scorer_error']}\n```")

# Sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-section-title first">Acciones</div>', unsafe_allow_html=True)

    if st.button("🔄 Actualizar scores", type="primary", use_container_width=True):
        st.cache_data.clear()
        run_scorer(top=200)

    if SCORING_CSV.exists():
        mtime = datetime.fromtimestamp(SCORING_CSV.stat().st_mtime)
        st.markdown(
            f'<div class="sidebar-timestamp">{ICON_CLOCK} {mtime.strftime("%d/%m/%Y %H:%M")}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("No hay datos. Presioná Actualizar para descargar mercados.")

    st.markdown('<div class="sidebar-section-title">Filtros</div>', unsafe_allow_html=True)

    df_full = load_csv()
    cat_sel = "Todas"
    señal_sel = None
    min_prob = 0
    if not df_full.empty:
        categorias = ["Todas"] + sorted(df_full["category"].dropna().unique().tolist())
        cat_sel = st.selectbox("Categoría", categorias)

        señal_filtro_sel = st.radio("Señal GB", ["Todas", "YES", "NO"], horizontal=True)
        señal_sel = SEÑAL_FILTRO_MAP[señal_filtro_sel]

        min_prob = st.slider(
            f"Prob. mínima GB ({st.session_state.get('min_prob_slider', 0)}%)",
            0, 100, 0, step=5, key="min_prob_slider",
        )

# ── Tabla principal ───────────────────────────────────────────────────────────
df_full = load_csv()

if df_full.empty:
    st.warning("Sin datos. Presioná **Actualizar scores** en el panel izquierdo.")
    st.stop()

# Aplicar filtros
df = df_full.copy()
if cat_sel != "Todas":
    df = df[df["category"] == cat_sel]
if señal_sel is not None:
    df = df[df["señal"] == señal_sel]
df = df[df["prob_yes_gb_pct"] >= min_prob]

# KPIs
kpis = [
    (ICON_BAR_CHART, "Mercados mostrados", f"{len(df)}"),
    (ICON_CHECK, "Señales YES (GB)", f"{int(df['pred_gb'].sum())}"),
    (ICON_TRENDING, "Prob. media GB", f"{df['prob_yes_gb_pct'].mean():.1f}%"),
    (ICON_TARGET, "Prob. máx GB", f"{df['prob_yes_gb_pct'].max():.1f}%"),
]
kpi_cards = "".join(
    f'<div class="kpi-card">'
    f'<div class="kpi-icon">{icon}</div>'
    f'<div class="kpi-label">{label}</div>'
    f'<div class="kpi-value">{value}</div>'
    f'</div>'
    for icon, label, value in kpis
)
st.markdown(f'<div class="kpi-grid">{kpi_cards}</div>', unsafe_allow_html=True)

st.divider()

# Tabla
df_sorted = df.sort_values("prob_yes_gb_pct", ascending=False)

rows_html = "".join(
    "<tr>"
    f"<td>{badge_signal(r['pred_gb'])}</td>"
    f"<td>{prob_bar(r['prob_yes_gb_pct'])}</td>"
    f"<td>{prob_bar(r['prob_yes_lr_pct'])}</td>"
    f"<td>{badge_category(r['category'])}</td>"
    f"<td>{format_fecha_es(r['start_date'])}</td>"
    f"<td>{html.escape(str(r['question']))}</td>"
    "</tr>"
    for _, r in df_sorted.iterrows()
)

table_html = f"""
<div class="mkt-table-wrap">
<table class="mkt-table">
<thead>
<tr>
<th>Señal</th>
<th>Prob GB (%)</th>
<th>Prob LR (%)</th>
<th>Categoría</th>
<th>Apertura</th>
<th>Pregunta</th>
</tr>
</thead>
<tbody>
{rows_html}
</tbody>
</table>
</div>
"""
st.markdown(table_html, unsafe_allow_html=True)

sel_idx = st.selectbox(
    "Ver detalle de mercado:",
    options=[-1] + df_sorted.index.tolist(),
    format_func=lambda i: "— seleccioná un mercado —" if i == -1 else str(df_sorted.loc[i, "question"])[:90],
)

# ── Detalle del mercado seleccionado ─────────────────────────────────────────
if sel_idx != -1:
    row = df.loc[sel_idx]

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

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    f'<div class="app-footer">'
    f'<div>Umbral del modelo: {THRESHOLD} · Modelos: GB (champion) · LR-C</div>'
    f'<div>Polymarket Scorer · v1.0</div>'
    f'</div>',
    unsafe_allow_html=True,
)
