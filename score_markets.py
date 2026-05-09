"""
Scoring de mercados abiertos en Polymarket.

Descarga mercados activos que llevan >= 7 días abiertos,
computa las mismas features del TFI y score con GB (champion) y LR-C.

Uso:
    python score_markets.py               # top 50 mercados por prob_yes
    python score_markets.py --top 20      # top 20
    python score_markets.py --all         # todos sin filtrar
    python score_markets.py --out mi.csv  # archivo de salida distinto
"""

import argparse
import json
import logging
import pickle
import re
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests

# ── Config ────────────────────────────────────────────────────────────────────
GAMMA_API  = "https://gamma-api.polymarket.com"
CLOB_API   = "https://clob.polymarket.com"
THRESHOLD  = 0.25          # umbral óptimo del TFI
OBS_DAYS   = 7             # ventana de observación
MIN_PTS    = 3             # mínimo de puntos de precio
CLOB_PAUSE = 0.15          # pausa entre requests CLOB

MODELS_DIR    = Path("models")
PROCESSED_DIR = Path("data/processed")
OUT_DEFAULT   = Path("scoring_output.csv")

CAT_DUMMIES = ["Crypto", "Entertainment", "Finance", "Politics", "Sports", "Tech"]
PRICE_DAY_COLS = [f"precio_dia_{d}" for d in range(1, 8)]
PRICE_AGG_COLS = ["precio_inicio", "precio_fin", "precio_media", "precio_mediana",
                  "precio_std", "precio_rango", "precio_tendencia", "volatilidad_retornos"]
ACTIVITY_COLS  = ["n_puntos_precio"]
CAT_COLS       = [f"cat_{c}" for c in CAT_DUMMIES]
ALL_FEATURES   = PRICE_DAY_COLS + PRICE_AGG_COLS + ACTIVITY_COLS + CAT_COLS
NUM_COLS       = PRICE_DAY_COLS + PRICE_AGG_COLS + ACTIVITY_COLS

logging.basicConfig(format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S", level=logging.INFO, stream=sys.stdout)
log = logging.getLogger(__name__)


# ── Categorización (mismas reglas v2 del TFI) ─────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.features.categorization import infer_category_coarse


# ── Feature engineering (idéntico a make_dataset.py) ─────────────────────────
def parse_dt(s) -> datetime | None:
    if not s:
        return None
    s = str(s).strip().replace(" ", "T")
    s = re.sub(r"([+-]\d{2})$", r"\g<1>:00", s)
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def build_features(history: list, start_ts: int) -> dict | None:
    history = sorted(history, key=lambda x: x["t"])
    n_pts = len(history)
    if n_pts < MIN_PTS:
        return None
    prices = [h["p"] for h in history]

    day_prices: dict[int, float] = {}
    for h in history:
        d = min(int((h["t"] - start_ts) / 86400) + 1, OBS_DAYS)
        day_prices[d] = h["p"]

    raw_vec = [day_prices.get(d, np.nan) for d in range(1, 8)]
    filled = raw_vec[:]
    for i in range(1, 7):
        if np.isnan(filled[i]) and not np.isnan(filled[i - 1]):
            filled[i] = filled[i - 1]
    for i in range(5, -1, -1):
        if np.isnan(filled[i]) and not np.isnan(filled[i + 1]):
            filled[i] = filled[i + 1]

    row = {f"precio_dia_{d}": filled[d - 1] for d in range(1, 8)}
    row["precio_inicio"]  = prices[0]
    row["precio_fin"]     = prices[-1]
    row["precio_media"]   = float(np.mean(prices))
    row["precio_mediana"] = float(np.median(prices))
    row["precio_std"]     = float(np.std(prices, ddof=1)) if n_pts > 1 else 0.0
    row["precio_rango"]   = float(max(prices) - min(prices))
    t_norm = np.arange(n_pts, dtype=float)
    row["precio_tendencia"] = float(np.polyfit(t_norm, prices, 1)[0]) if n_pts >= 2 else 0.0

    if n_pts >= 3:
        safe = [(prices[i], prices[i - 1]) for i in range(1, n_pts)
                if prices[i] > 0 and prices[i - 1] > 0]
        log_rets = [np.log(a / b) for a, b in safe]
        row["volatilidad_retornos"] = float(np.std(log_rets, ddof=1)) if len(log_rets) >= 2 else 0.0
    else:
        row["volatilidad_retornos"] = 0.0

    row["n_puntos_precio"] = n_pts
    return row


# ── API helpers ───────────────────────────────────────────────────────────────
def fetch_open_markets(min_days_open: int = OBS_DAYS) -> list[dict]:
    """Devuelve mercados binarios abiertos que llevan >= min_days_open días."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=min_days_open)
    markets, cursor = [], None
    log.info("Descargando mercados abiertos de Gamma API...")

    while True:
        params = {"active": "true", "closed": "false", "limit": 100}
        if cursor:
            params["next_cursor"] = cursor
        r = requests.get(f"{GAMMA_API}/markets", params=params, timeout=30)
        r.raise_for_status()
        data = r.json()

        page = data if isinstance(data, list) else data.get("data", [])
        if not page:
            break

        for m in page:
            # Solo mercados binarios (un token YES/NO)
            outcomes_raw = m.get("tokens") or m.get("outcomes", [])
            if isinstance(outcomes_raw, str):
                try:
                    outcomes_raw = json.loads(outcomes_raw)
                except Exception:
                    outcomes_raw = []
            if len(outcomes_raw) != 2:
                continue

            # Extraer YES token ID (clobTokenIds[0])
            clob_ids = m.get("clobTokenIds", [])
            if isinstance(clob_ids, str):
                try:
                    clob_ids = json.loads(clob_ids)
                except Exception:
                    clob_ids = []
            if not clob_ids:
                continue
            m["_yes_token_id"] = clob_ids[0]

            sd = parse_dt(m.get("startDate") or m.get("createdAt"))
            if sd and sd.replace(tzinfo=timezone.utc) <= cutoff:
                markets.append(m)

        cursor = data.get("next_cursor") if isinstance(data, dict) else None
        if not cursor:
            break
        time.sleep(0.5)

    log.info(f"  {len(markets)} mercados candidatos con >= {min_days_open} días abiertos")
    return markets


def fetch_prices(token_id: str, start_ts: int) -> list[dict]:
    """Descarga precios del token YES para los primeros OBS_DAYS días."""
    end_ts = start_ts + OBS_DAYS * 86400
    try:
        r = requests.get(
            f"{CLOB_API}/prices-history",
            params={"market": token_id, "startTs": start_ts, "endTs": end_ts, "fidelity": 1440},
            timeout=20,
        )
        r.raise_for_status()
        return r.json().get("history", [])
    except Exception as e:
        log.debug(f"    Error precios {token_id}: {e}")
        return []


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=50, help="Mostrar top N por prob_yes (default 50)")
    parser.add_argument("--all", action="store_true", help="Mostrar todos sin filtrar por top")
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT, help="Archivo CSV de salida")
    args = parser.parse_args()

    # Cargar modelos y scaler
    log.info("Cargando modelos...")
    gb     = joblib.load(MODELS_DIR / "best_tree_model.pkl")
    lr     = joblib.load(MODELS_DIR / "best_lr.pkl")
    scaler = pickle.load(open(PROCESSED_DIR / "scaler.pkl", "rb"))

    # Descargar mercados abiertos
    markets = fetch_open_markets()
    if not markets:
        log.warning("No se encontraron mercados candidatos.")
        return

    # Construir features
    rows = []
    log.info(f"Extrayendo features para {len(markets)} mercados...")
    for i, m in enumerate(markets):
        cid = m.get("conditionId", "")
        if not cid:
            continue

        sd = parse_dt(m.get("startDate") or m.get("createdAt"))
        if sd is None:
            continue
        start_ts = int(sd.replace(tzinfo=timezone.utc).timestamp())

        token_id = m.get("_yes_token_id", cid)
        history = fetch_prices(token_id, start_ts)
        feats = build_features(history, start_ts)
        if feats is None:
            continue

        cat = infer_category_coarse(m.get("question", ""))
        for c in CAT_DUMMIES:
            feats[f"cat_{c}"] = int(cat == c)

        feats["condition_id"] = cid
        feats["question"]     = m.get("question", "")[:120]
        feats["category"]     = cat
        feats["start_date"]   = sd.strftime("%Y-%m-%d")
        rows.append(feats)

        time.sleep(CLOB_PAUSE)
        if (i + 1) % 20 == 0:
            log.info(f"  {i + 1}/{len(markets)} procesados, {len(rows)} con features válidas")

    if not rows:
        log.warning("Ningún mercado tuvo suficientes datos de precio.")
        return

    df = pd.DataFrame(rows)
    log.info(f"Mercados con features válidas: {len(df)}")

    # Escalar features numéricas (mismo scaler del entrenamiento)
    df[NUM_COLS] = scaler.transform(df[NUM_COLS])

    # Scoring
    X = df[ALL_FEATURES].values
    df["prob_yes_gb"] = gb.predict_proba(X)[:, 1]
    df["prob_yes_lr"] = lr.predict_proba(X)[:, 1]
    df["pred_gb"]     = (df["prob_yes_gb"] >= THRESHOLD).astype(int)
    df["pred_lr"]     = (df["prob_yes_lr"] >= THRESHOLD).astype(int)

    # Ordenar por prob_yes_gb descendente
    df = df.sort_values("prob_yes_gb", ascending=False).reset_index(drop=True)

    # Seleccionar columnas de salida
    out_cols = ["question", "category", "start_date", "prob_yes_gb", "prob_yes_lr",
                "pred_gb", "pred_lr", "condition_id"]
    result = df[out_cols] if not args.all else df[out_cols]
    if not args.all:
        result = result.head(args.top)

    # Guardar CSV
    result.to_csv(args.out, index=False)
    log.info(f"Resultado guardado en {args.out}")

    # Mostrar en consola
    print("\n" + "=" * 90)
    print(f"TOP {len(result)} MERCADOS POR PROBABILIDAD DE CIERRE EN YES (GB)")
    print("=" * 90)
    for _, row in result.iterrows():
        pred_str = "YES" if row["pred_gb"] else "NO "
        print(f"  [{pred_str}] {row['prob_yes_gb']:.2f}  {row['category']:<14}  {row['question']}")
    print("=" * 90)
    print(f"\nUmbral aplicado: {THRESHOLD}  |  Mercados predichos YES: {result['pred_gb'].sum()}")


if __name__ == "__main__":
    main()
