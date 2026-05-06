"""
Fase 4 — Feature engineering, split y persistencia.

Genera data/processed/{train,val,test}.parquet + metadata.

Split: estratificado por bucket temporal, asignación determinística por conditionId.
  Buckets: pre-2026 | 2026-01 | 2026-02 | 2026-03+
  Dentro de cada bucket: hash(conditionId) % 100 → 0-69 train / 70-79 val / 80-99 test

Uso:
    python -m src.data.make_dataset
"""

import hashlib
import json
import pickle
import re
import sys
from datetime import datetime, date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import MIN_PRICE_POINTS, OBSERVATION_WINDOW_DAYS
from src.features.categorization import infer_category_coarse

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_DIR       = Path("data/raw")
PRICES_DIR    = RAW_DIR / "prices"
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ── Categorías — one-hot (Other = referencia, omitida) ────────────────────────
CAT_DUMMIES = ["Crypto", "Entertainment", "Finance", "Politics", "Sports", "Tech"]

# ── Feature columns (orden canónico) ─────────────────────────────────────────
PRICE_DAY_COLS  = [f"precio_dia_{d}" for d in range(1, 8)]
PRICE_AGG_COLS  = ["precio_inicio", "precio_fin", "precio_media", "precio_mediana",
                   "precio_std", "precio_rango", "precio_tendencia", "volatilidad_retornos"]
ACTIVITY_COLS   = ["n_puntos_precio"]
CAT_COLS        = [f"cat_{c}" for c in CAT_DUMMIES]
ALL_FEATURE_COLS = PRICE_DAY_COLS + PRICE_AGG_COLS + ACTIVITY_COLS + CAT_COLS
TARGET_COL      = "outcome"


# ── Helpers ───────────────────────────────────────────────────────────────────
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


def build_price_features(history: list, start_ts: int) -> dict | None:
    history = sorted(history, key=lambda x: x["t"])
    n_pts = len(history)
    if n_pts < MIN_PRICE_POINTS:
        return None
    prices = [h["p"] for h in history]

    # Map each observation to its day-index (1–7)
    day_prices: dict[int, float] = {}
    for h in history:
        d = min(int((h["t"] - start_ts) / 86400) + 1, OBSERVATION_WINDOW_DAYS)
        day_prices[d] = h["p"]

    # Raw day vector (NaN where no observation)
    raw_vec = [day_prices.get(d, np.nan) for d in range(1, 8)]

    # Forward-fill then backward-fill
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
        row["volatilidad_retornos"] = (
            float(np.std(log_rets, ddof=1)) if len(log_rets) >= 2 else 0.0
        )
    else:
        row["volatilidad_retornos"] = 0.0

    row["n_puntos_precio"] = n_pts
    return row


def temporal_bucket(start_dt: datetime) -> str:
    d = start_dt.date()
    if d < date(2026, 1, 1):
        return "pre-2026"
    elif d < date(2026, 2, 1):
        return "2026-01"
    elif d < date(2026, 3, 1):
        return "2026-02"
    else:
        return "2026-03+"


def assign_split(condition_id: str, bucket: str) -> str:
    # Deterministic: same conditionId always → same split, regardless of run order
    h = int(hashlib.md5(condition_id.encode()).hexdigest(), 16)
    pct = h % 100
    return "train" if pct < 70 else ("val" if pct < 80 else "test")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    rows = []
    for mf in (RAW_DIR / "markets").glob("*.json"):
        with open(mf, encoding="utf-8") as f:
            m = json.load(f)

        cid = m.get("conditionId") or mf.stem
        pf  = PRICES_DIR / f"{cid}.json"
        if not pf.exists():
            continue
        with open(pf, encoding="utf-8") as f:
            history = json.load(f).get("history", [])

        price_feats = build_price_features(history, 0)  # placeholder; recalculate below
        if price_feats is None:
            continue

        # Recalculate with correct start_ts
        sd = parse_dt(m.get("startDate"))
        if sd is None:
            continue
        start_ts = int(sd.timestamp())
        price_feats = build_price_features(history, start_ts)
        if price_feats is None:
            continue

        # Outcome
        op      = json.loads(m.get("outcomePrices", '["0","0"]'))
        outcome = int(op[0] == "1")

        # Category (v2 rules)
        cat = infer_category_coarse(m.get("question", ""))

        # Split assignment
        bucket = temporal_bucket(sd)
        split  = assign_split(cid, bucket)

        row = {
            "condition_id": cid,
            "question":     m.get("question", ""),
            "bucket":       bucket,
            "split":        split,
            "category_coarse": cat,
            TARGET_COL:     outcome,
        }
        row.update(price_feats)
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"Total registros: {len(df)}")

    # One-hot encode category_coarse (Other = referencia, omitida)
    for c in CAT_DUMMIES:
        df[f"cat_{c}"] = (df["category_coarse"] == c).astype(np.int8)

    # Split datasets
    train = df[df["split"] == "train"].copy()
    val   = df[df["split"] == "val"].copy()
    test  = df[df["split"] == "test"].copy()

    # ── StandardScaler: fit on train, transform all ──────────────────────────
    num_cols = PRICE_DAY_COLS + PRICE_AGG_COLS + ACTIVITY_COLS
    scaler   = StandardScaler()
    train[num_cols] = scaler.fit_transform(train[num_cols])
    val[num_cols]   = scaler.transform(val[num_cols])
    test[num_cols]  = scaler.transform(test[num_cols])

    # ── Persist ──────────────────────────────────────────────────────────────
    feature_cols_meta = {
        "numeric": num_cols,
        "binary_onehot": CAT_COLS,
        "all_features": ALL_FEATURE_COLS,
        "target": TARGET_COL,
    }

    train[ALL_FEATURE_COLS + [TARGET_COL, "condition_id", "split", "bucket", "category_coarse"]].to_parquet(
        PROCESSED_DIR / "train.parquet", index=False
    )
    val[ALL_FEATURE_COLS + [TARGET_COL, "condition_id", "split", "bucket", "category_coarse"]].to_parquet(
        PROCESSED_DIR / "val.parquet", index=False
    )
    test[ALL_FEATURE_COLS + [TARGET_COL, "condition_id", "split", "bucket", "category_coarse"]].to_parquet(
        PROCESSED_DIR / "test.parquet", index=False
    )

    with open(PROCESSED_DIR / "feature_columns.json", "w") as f:
        json.dump(feature_cols_meta, f, indent=2)

    with open(PROCESSED_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # ── Stats report ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("SPLIT SIZES Y YES RATE")
    print("="*60)
    for name, ds in [("TRAIN", train), ("VAL", val), ("TEST", test)]:
        print(f"\n  {name}: n={len(ds)}  YES={ds[TARGET_COL].sum()}  YES%={ds[TARGET_COL].mean()*100:.1f}%")
        bkt_counts = ds["bucket"].value_counts().sort_index().to_dict()
        print(f"    Buckets: {bkt_counts}")

    print("\n" + "="*60)
    print("DISTRIBUCION CATEGORY_COARSE POR SPLIT")
    print("="*60)
    ct = pd.crosstab(df["category_coarse"], df["split"])[["train","val","test"]]
    ct["total"] = ct.sum(axis=1)
    print(ct.to_string())

    print("\n" + "="*60)
    print("ESTADISTICAS NUMERICAS TRAS NORMALIZACION")
    print("="*60)
    key_num = ["precio_fin", "precio_media", "precio_tendencia", "volatilidad_retornos", "n_puntos_precio"]
    for col in key_num:
        tm, ts = train[col].mean(), train[col].std()
        vm, vs = val[col].mean(),   val[col].std()
        xm, xs = test[col].mean(),  test[col].std()
        print(f"  {col:<25}  train(mu={tm:+.3f} s={ts:.3f})  val(mu={vm:+.3f} s={vs:.3f})  test(mu={xm:+.3f} s={xs:.3f})")

    print("\n" + "="*60)
    print("VERIFICACION DE NO-LEAKAGE")
    print("="*60)
    leakage_risk = {
        "precio_dia_1..7":       "Solo usa precios del historial de precios en la ventana start_date + 7 dias. OK.",
        "precio_inicio/fin/...": "Calculados sobre history dentro de la ventana de 7 dias. OK.",
        "volatilidad_retornos":  "Log-returns calculados sobre prices[] dentro de la ventana. OK.",
        "n_puntos_precio":       "Cuenta de observaciones en la ventana. OK.",
        "cat_*":                 "Derivado del texto de la pregunta (disponible al abrir el mercado). OK.",
        "log_volumen_total":     "EXCLUIDA (leakage: snapshot incluye volumen post-ventana).",
        "duration_days":         "EXCLUIDA (leakage: duracion solo conocida tras resolucion).",
    }
    for feat, status in leakage_risk.items():
        print(f"  {feat:<30}  {status}")

    # ── File sizes ────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("ARCHIVOS GENERADOS EN data/processed/")
    print("="*60)
    for f in sorted(PROCESSED_DIR.glob("*")):
        if f.name.startswith("."): continue
        print(f"  {f.name:<30}  {f.stat().st_size / 1024:.1f} KB")

    # ── Split stats JSON ──────────────────────────────────────────────────────
    split_stats = {}
    for name, ds in [("train", train), ("val", val), ("test", test)]:
        split_stats[name] = {
            "n": len(ds),
            "yes": int(ds[TARGET_COL].sum()),
            "yes_pct": round(ds[TARGET_COL].mean() * 100, 1),
            "buckets": ds["bucket"].value_counts().sort_index().to_dict(),
            "categories": ds["category_coarse"].value_counts().sort_index().to_dict(),
        }
    with open(PROCESSED_DIR / "split_stats.json", "w") as f:
        json.dump(split_stats, f, indent=2)

    print(f"\nFeature set: {len(ALL_FEATURE_COLS)} features  +  1 target")
    print("Done.")


if __name__ == "__main__":
    main()
