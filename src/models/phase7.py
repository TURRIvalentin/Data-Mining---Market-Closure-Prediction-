"""
Fase 7 — Threshold optimization, calibración, errores por categoría,
feature importance comparado.

Uso:
    python -m src.models.phase7
"""

import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    accuracy_score, roc_auc_score, confusion_matrix,
)

PROCESSED   = Path("data/processed")
MODELS_DIR  = Path("models")
FIGURES_DIR = Path("reports/figures"); FIGURES_DIR.mkdir(exist_ok=True)
REPORTS_DIR = Path("reports"); REPORTS_DIR.mkdir(exist_ok=True)

# ── Cargar datos ──────────────────────────────────────────────────────────────
with open(PROCESSED / "feature_columns.json") as f:
    fc = json.load(f)

ALL_FEATURES = fc["all_features"]
NUM_COLS     = fc["numeric"]
TARGET       = fc["target"]

train    = pd.read_parquet(PROCESSED / "train.parquet")
val      = pd.read_parquet(PROCESSED / "val.parquet")
test     = pd.read_parquet(PROCESSED / "test.parquet")
trainval = pd.concat([train, val], ignore_index=True)

X_test  = test[ALL_FEATURES].values
y_test  = test[TARGET].values

# ── Cargar modelos ────────────────────────────────────────────────────────────
with open(MODELS_DIR / "best_tree_model.pkl", "rb") as f:
    gb_model = pickle.load(f)

with open(MODELS_DIR / "best_lr.pkl", "rb") as f:
    lr_c_model = pickle.load(f)

with open(PROCESSED / "scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Probabilidades en test
gb_prob   = gb_model.predict_proba(X_test)[:, 1]
lr_c_prob = lr_c_model.predict_proba(X_test)[:, 1]

pf_idx  = NUM_COLS.index("precio_fin")
b3_prob = np.clip(
    test["precio_fin"].values * scaler.scale_[pf_idx] + scaler.mean_[pf_idx],
    0.0, 1.0
)

print("Modelos y datos cargados.")
print(f"  GB:   mean={gb_prob.mean():.3f}  std={gb_prob.std():.3f}")
print(f"  LR-C: mean={lr_c_prob.mean():.3f}  std={lr_c_prob.std():.3f}")
print(f"  B3:   mean={b3_prob.mean():.3f}  std={b3_prob.std():.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# SECCION 1: THRESHOLD OPTIMIZATION (GB)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("SECCION 1: THRESHOLD OPTIMIZATION — GB")
print("="*65)

thresholds = np.round(np.arange(0.05, 0.55, 0.05), 2)
rows = []
for t in thresholds:
    y_pred  = (gb_prob >= t).astype(int)
    prec    = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    rec     = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1y     = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1mac   = f1_score(y_test, y_pred, average="macro", zero_division=0)
    acc     = accuracy_score(y_test, y_pred)
    n_yes   = int(y_pred.sum())
    rows.append(dict(threshold=t, precision=round(prec,4), recall=round(rec,4),
                     f1_yes=round(f1y,4), f1_macro=round(f1mac,4),
                     accuracy=round(acc,4), n_pred_yes=n_yes))

thresh_df = pd.DataFrame(rows)
print(thresh_df.to_string(index=False))

best_f1yes  = thresh_df.loc[thresh_df["f1_yes"].idxmax()]
best_f1mac  = thresh_df.loc[thresh_df["f1_macro"].idxmax()]
opt_threshold = float(best_f1mac["threshold"])

print(f"\n  >>> Max F1(YES):   t={best_f1yes['threshold']:.2f}  "
      f"F1={best_f1yes['f1_yes']:.4f}  prec={best_f1yes['precision']:.4f}  rec={best_f1yes['recall']:.4f}  "
      f"n_pred_YES={int(best_f1yes['n_pred_yes'])}")
print(f"  >>> Max F1(macro): t={best_f1mac['threshold']:.2f}  "
      f"F1={best_f1mac['f1_yes']:.4f}  prec={best_f1mac['precision']:.4f}  rec={best_f1mac['recall']:.4f}  "
      f"n_pred_YES={int(best_f1mac['n_pred_yes'])}")
print(f"\n  Threshold elegido: {opt_threshold:.2f} (max F1 macro)")
print(f"  F1(YES) con threshold default 0.50: "
      f"{thresh_df[thresh_df['threshold']==0.50]['f1_yes'].values[0]:.4f}")
print(f"  F1(YES) con threshold optimo:       {best_f1mac['f1_yes']:.4f}")

# Figura 1: Threshold analysis
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(thresh_df["threshold"], thresh_df["precision"],  "o-", color="#2196F3", lw=2, label="Precision")
ax.plot(thresh_df["threshold"], thresh_df["recall"],     "s-", color="#E91E63", lw=2, label="Recall")
ax.plot(thresh_df["threshold"], thresh_df["f1_yes"],     "^-", color="#FF9800", lw=2, label="F1(YES)")
ax.plot(thresh_df["threshold"], thresh_df["f1_macro"],   "D-", color="#4CAF50", lw=2, label="F1(macro)")
ax.axvline(opt_threshold, color="grey", ls="--", lw=1.5,
           label=f"Threshold opt. = {opt_threshold}")
ax.axvline(0.5, color="black", ls=":", lw=1, alpha=0.5, label="Default = 0.50")
ax.set_xlabel("Threshold de clasificacion"); ax.set_ylabel("Metrica")
ax.set_title("GB — Precision / Recall / F1 vs Threshold")
ax.legend(fontsize=9); ax.set_xlim(0.03, 0.52); ax.set_ylim(0, 1.02)
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "threshold_analysis_gb.png", dpi=150)
plt.close(fig)
print(f"\n  Figura: reports/figures/threshold_analysis_gb.png")


# ─────────────────────────────────────────────────────────────────────────────
# SECCION 2: CALIBRACION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("SECCION 2: CALIBRACION")
print("="*65)

N_BINS = 10

def calibration_errors(y_true, y_prob, n_bins=N_BINS):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = mce = 0.0
    n = len(y_true)
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i+1])
        if mask.sum() == 0:
            continue
        frac_pos  = float(y_true[mask].mean())
        mean_prob = float(y_prob[mask].mean())
        err = abs(frac_pos - mean_prob)
        ece += (mask.sum() / n) * err
        mce = max(mce, err)
    return round(ece, 4), round(mce, 4)

models_cal = [("GB", gb_prob, "#E91E63"), ("LR-C", lr_c_prob, "#2196F3"), ("B3", b3_prob, "#4CAF50")]
cal_summary = {}
for name, prob, _ in models_cal:
    ece, mce = calibration_errors(y_test, prob)
    cal_summary[name] = {"ECE": ece, "MCE": mce}
    print(f"  {name:<10}  ECE={ece:.4f}  MCE={mce:.4f}")

# Figura 2: Reliability diagram 3-panel
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
for ax, (name, prob, color) in zip(axes, models_cal):
    prob_true, prob_pred = calibration_curve(y_test, prob, n_bins=N_BINS)
    ax.plot(prob_pred, prob_true, "o-", color=color, lw=2, ms=6, label="Calibracion real")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfecta")
    ax.fill_between([0, 1], [0, 1], alpha=0.04, color="grey")
    ece, mce = cal_summary[name]["ECE"], cal_summary[name]["MCE"]
    ax.set_title(f"{name}\nECE={ece:.4f}  MCE={mce:.4f}", fontsize=10)
    ax.set_xlabel("Probabilidad predicha"); ax.set_ylabel("Fraccion positivos")
    ax.legend(fontsize=8); ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.2)
plt.suptitle("Reliability Diagrams — Comparacion GB vs LR-C vs B3", fontsize=12)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "calibracion_comparacion.png", dpi=150)
plt.close(fig)
print(f"\n  Figura: reports/figures/calibracion_comparacion.png")


# ─────────────────────────────────────────────────────────────────────────────
# SECCION 3: ANALISIS DE ERRORES POR CATEGORIA
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("SECCION 3: ERRORES POR CATEGORIA (GB, threshold opt.)")
print("="*65)

opt_pred = (gb_prob >= opt_threshold).astype(int)
meta = test[["category_coarse", "bucket"]].copy()
meta["y_true"] = y_test
meta["y_pred"] = opt_pred
meta["y_prob"] = gb_prob

cat_rows = []
for cat in sorted(meta["category_coarse"].unique()):
    m = meta["category_coarse"] == cat
    yt = y_test[m.values]; yp = opt_pred[m.values]; yprob = gb_prob[m.values]
    n = m.sum(); n_yes = int(yt.sum())
    if n < 5:
        cat_rows.append({"category": cat, "n": n, "n_yes": n_yes,
                         "yes_pct": round(n_yes/n*100,1), "note": "n<5, omitido"})
        continue
    auc   = roc_auc_score(yt, yprob) if len(np.unique(yt)) > 1 else float("nan")
    prec  = precision_score(yt, yp, pos_label=1, zero_division=0)
    rec   = recall_score(yt, yp, pos_label=1, zero_division=0)
    f1y   = f1_score(yt, yp, pos_label=1, zero_division=0)
    acc   = accuracy_score(yt, yp)
    cm    = confusion_matrix(yt, yp, labels=[0, 1])
    cat_rows.append({"category": cat, "n": n, "n_yes": n_yes,
                     "yes_pct": round(n_yes/n*100, 1),
                     "auc": round(float(auc), 3),
                     "precision": round(prec, 3), "recall": round(rec, 3),
                     "f1_yes": round(f1y, 3), "accuracy": round(acc, 3),
                     "tn": int(cm[0,0]), "fp": int(cm[0,1]),
                     "fn": int(cm[1,0]), "tp": int(cm[1,1])})

cat_df = pd.DataFrame([r for r in cat_rows if "note" not in r])
print(cat_df[["category","n","n_yes","yes_pct","auc","precision",
              "recall","f1_yes","accuracy"]].to_string(index=False))
print("\n  Confusion matrix por categoria:")
print(cat_df[["category","tn","fp","fn","tp"]].to_string(index=False))

# Error types per category
meta["error_type"] = "correct"
meta.loc[(meta["y_true"]==1) & (meta["y_pred"]==0), "error_type"] = "FN"
meta.loc[(meta["y_true"]==0) & (meta["y_pred"]==1), "error_type"] = "FP"
print("\n  Errores por categoria y tipo:")
print(pd.crosstab(meta["category_coarse"], meta["error_type"]).to_string())

# Figura 3: por categoría
cats_plot = cat_df[cat_df["n"] >= 5].copy()
x  = np.arange(len(cats_plot))
w  = 0.35

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax1, ax2 = axes

# AUC por categoría
colors_auc = ["#E91E63" if v < 0.75 else "#4CAF50" for v in cats_plot["auc"]]
bars1 = ax1.bar(x, cats_plot["auc"], color=colors_auc, width=0.6, alpha=0.85)
ax1.axhline(0.8933, color="#E91E63", ls="--", lw=1.5, label=f"AUC global={0.8933}")
ax1.set_xticks(x); ax1.set_xticklabels(cats_plot["category"], rotation=30, ha="right")
ax1.set_ylim(0.4, 1.02); ax1.set_ylabel("AUC")
ax1.set_title("AUC por categoria (GB)")
ax1.legend(fontsize=8); ax1.grid(axis="y", alpha=0.3)
for bar, v in zip(bars1, cats_plot["auc"]):
    if not np.isnan(v):
        ax1.text(bar.get_x()+bar.get_width()/2, v+0.01, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=8)

# Precision/Recall/F1 stacked
b1 = ax2.bar(x - w/2, cats_plot["precision"], w, label="Precision",  color="#2196F3", alpha=0.85)
b2 = ax2.bar(x + w/2, cats_plot["recall"],    w, label="Recall",     color="#E91E63", alpha=0.85)
ax2.plot(x, cats_plot["f1_yes"], "D--", color="#FF9800", ms=7, lw=2, label="F1(YES)")
ax2.set_xticks(x); ax2.set_xticklabels(cats_plot["category"], rotation=30, ha="right")
ax2.set_ylim(0, 1.05); ax2.set_ylabel("Metrica")
ax2.set_title("Precision / Recall / F1(YES) por categoria (GB)")
ax2.legend(fontsize=8); ax2.grid(axis="y", alpha=0.3)

# Add n_yes labels
for xi, row in zip(x, cats_plot.itertuples()):
    ax2.text(xi, 1.02, f"n={row.n_yes}YES/{row.n}",
             ha="center", va="bottom", fontsize=7, color="grey")

plt.suptitle(f"GB con threshold={opt_threshold:.2f} — Analisis por categoria", fontsize=11)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "errores_por_categoria_gb.png", dpi=150)
plt.close(fig)
print(f"\n  Figura: reports/figures/errores_por_categoria_gb.png")


# ─────────────────────────────────────────────────────────────────────────────
# SECCION 4: FEATURE IMPORTANCE — GB vs LR-C
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("SECCION 4: FEATURE IMPORTANCE COMPARADO")
print("="*65)

gb_mdi    = gb_model.feature_importances_
lrc_coefs = lr_c_model.coef_[0]

# Rankings
mdi_s = pd.Series(gb_mdi,     index=ALL_FEATURES).sort_values(ascending=False)
lrc_s = pd.Series(np.abs(lrc_coefs), index=ALL_FEATURES).sort_values(ascending=False)

print("\n  Top 10 — GB (MDI)       vs LR-C (|coef|):")
print(f"  {'GB feature':<30}  {'MDI':>6}  |  {'LR-C feature':<30}  {'|coef|':>7}")
for i in range(10):
    gf, gv = mdi_s.index[i], mdi_s.values[i]
    lf, lv = lrc_s.index[i], lrc_s.values[i]
    print(f"  {gf:<30}  {gv:.4f}  |  {lf:<30}  {lv:.4f}")

# Permutation importance
print("\n  Calculando permutation importance en test (30 repeats, AUC)...")
perm_res = permutation_importance(
    gb_model, X_test, y_test,
    n_repeats=30, random_state=42, scoring="roc_auc", n_jobs=1
)
perm_s = pd.Series(perm_res.importances_mean, index=ALL_FEATURES).sort_values(ascending=False)
perm_std = pd.Series(perm_res.importances_std, index=ALL_FEATURES)

print("\n  Top 10 — GB (Permutation Importance, AUC drop):")
for fname in perm_s.head(10).index:
    print(f"    {fname:<30}  {perm_s[fname]:+.4f} +/- {perm_std[fname]:.4f}")

print("\n  Features con importancia negativa (permutacion mejora AUC):")
neg = perm_s[perm_s < -0.005]
if len(neg):
    for f, v in neg.items():
        print(f"    {f:<30}  {v:+.4f}")
else:
    print("    Ninguna.")

# Rank convergencia MDI vs PI
print(f"\n  Convergencia MDI vs PI (rango de las mismas features):")
top10_mdi = set(mdi_s.head(10).index)
top10_pi  = set(perm_s.head(10).index)
print(f"  Top 10 MDI:  {list(mdi_s.head(10).index)}")
print(f"  Top 10 PI:   {list(perm_s.head(10).index)}")
print(f"  Overlap top10: {len(top10_mdi & top10_pi)}/10  features en comun: {sorted(top10_mdi & top10_pi)}")

# Figura 4: Feature importance comparado (3 columnas: MDI, PI, LR-C |coef|)
top_n = 15
fig, axes = plt.subplots(1, 3, figsize=(16, 6))

for ax, (s, title, color) in zip(axes, [
    (mdi_s.head(top_n), "GB: MDI Importance", "#E91E63"),
    (perm_s.head(top_n), "GB: Permutation Importance (AUC drop)", "#9C27B0"),
    (lrc_s.head(top_n), "LR-C: |Coeficiente| (L1)", "#2196F3"),
]):
    ax.barh(s.index[::-1], s.values[::-1], color=color, alpha=0.8, height=0.65)
    ax.set_xlabel("Importancia"); ax.set_title(title, fontsize=10)
    ax.grid(axis="x", alpha=0.3)

plt.suptitle("Feature Importance: GB (MDI y Permutacion) vs LR-C", fontsize=12)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "feature_importance_comparado.png", dpi=150)
plt.close(fig)
print(f"\n  Figura: reports/figures/feature_importance_comparado.png")


# ─────────────────────────────────────────────────────────────────────────────
# SECCION 5 (opcional): Análisis por bucket temporal
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("SECCION 5: ANALISIS POR BUCKET TEMPORAL (GB)")
print("="*65)

for bucket in sorted(meta["bucket"].unique()):
    m = meta["bucket"] == bucket
    yt = y_test[m.values]; yp = opt_pred[m.values]; yprob = gb_prob[m.values]
    n = m.sum()
    if n < 5 or len(np.unique(yt)) < 2:
        print(f"  {bucket}: n={n}, omitido (n<5 o sin YES)")
        continue
    auc = roc_auc_score(yt, yprob)
    f1y = f1_score(yt, yp, pos_label=1, zero_division=0)
    cm  = confusion_matrix(yt, yp, labels=[0,1])
    print(f"  {bucket:<12}  n={n:3d}  YES={int(yt.sum()):2d}  AUC={auc:.3f}  "
          f"F1(YES)={f1y:.3f}  CM: TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")


# ─────────────────────────────────────────────────────────────────────────────
# Guardar métricas JSON
# ─────────────────────────────────────────────────────────────────────────────
summary = {
    "threshold_optimization": {
        "optimal_threshold": opt_threshold,
        "by_threshold": thresh_df.to_dict(orient="records"),
        "best_f1_yes": {"threshold": float(best_f1yes["threshold"]),
                        "f1_yes": float(best_f1yes["f1_yes"]),
                        "precision": float(best_f1yes["precision"]),
                        "recall": float(best_f1yes["recall"])},
        "best_f1_macro": {"threshold": float(best_f1mac["threshold"]),
                          "f1_yes": float(best_f1mac["f1_yes"]),
                          "precision": float(best_f1mac["precision"]),
                          "recall": float(best_f1mac["recall"])},
    },
    "calibration": cal_summary,
    "per_category": cat_df.to_dict(orient="records"),
    "feature_importance_top10_mdi":  dict(zip(mdi_s.head(10).index, [round(v,4) for v in mdi_s.head(10)])),
    "feature_importance_top10_perm": dict(zip(perm_s.head(10).index, [round(v,4) for v in perm_s.head(10)])),
    "feature_importance_top10_lrc":  dict(zip(lrc_s.head(10).index, [round(v,4) for v in lrc_s.head(10)])),
}

with open(REPORTS_DIR / "fase7_analysis.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*65)
print("FASE 7 COMPLETA")
print("="*65)
print(f"  Threshold optimo (max F1 macro): {opt_threshold}")
print(f"  F1(YES) con t=0.50: "
      f"{thresh_df[thresh_df['threshold']==0.50]['f1_yes'].values[0]:.4f}")
print(f"  F1(YES) con t={opt_threshold}: {best_f1mac['f1_yes']:.4f}")
print(f"  Calibracion GB:   ECE={cal_summary['GB']['ECE']:.4f}  MCE={cal_summary['GB']['MCE']:.4f}")
print(f"  Calibracion LR-C: ECE={cal_summary['LR-C']['ECE']:.4f}  MCE={cal_summary['LR-C']['MCE']:.4f}")
print(f"  Calibracion B3:   ECE={cal_summary['B3']['ECE']:.4f}  MCE={cal_summary['B3']['MCE']:.4f}")
print(f"  reports/fase7_analysis.json")
print(f"  reports/figures/{{threshold_analysis_gb, calibracion_comparacion,")
print(f"                    errores_por_categoria_gb, feature_importance_comparado}}.png")
