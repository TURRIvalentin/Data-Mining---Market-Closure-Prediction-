"""
Fase 6 — Random Forest y Gradient Boosting.

Uso:
    python -m src.models.phase6
"""

import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, log_loss,
    brier_score_loss, accuracy_score, f1_score,
    confusion_matrix, roc_curve,
)

PROCESSED   = Path("data/processed")
MODELS_DIR  = Path("models");  MODELS_DIR.mkdir(exist_ok=True)
FIGURES_DIR = Path("reports/figures"); FIGURES_DIR.mkdir(exist_ok=True)
REPORTS_DIR = Path("reports"); REPORTS_DIR.mkdir(exist_ok=True)
RANDOM_SEED = 42
N_ITER      = 25
CV_FOLDS    = 5

# ── Load data ─────────────────────────────────────────────────────────────────
with open(PROCESSED / "feature_columns.json") as f:
    fc = json.load(f)

ALL_FEATURES = fc["all_features"]
TARGET       = fc["target"]

train    = pd.read_parquet(PROCESSED / "train.parquet")
val      = pd.read_parquet(PROCESSED / "val.parquet")
test     = pd.read_parquet(PROCESSED / "test.parquet")
trainval = pd.concat([train, val], ignore_index=True)

X_train    = train[ALL_FEATURES].values
y_train    = train[TARGET].values
X_test     = test[ALL_FEATURES].values
y_test     = test[TARGET].values
X_trainval = trainval[ALL_FEATURES].values
y_trainval = trainval[TARGET].values


# ── Metrics ───────────────────────────────────────────────────────────────────
def evaluate(name: str, y_true, y_prob, y_pred=None, threshold=0.5) -> dict:
    if y_pred is None:
        y_pred = (y_prob >= threshold).astype(int)
    yp = np.clip(y_prob, 1e-7, 1 - 1e-7)
    return {
        "name":    name,
        "auc":     round(roc_auc_score(y_true, yp), 4),
        "pr_auc":  round(average_precision_score(y_true, yp), 4),
        "logloss": round(log_loss(y_true, yp), 4),
        "brier":   round(brier_score_loss(y_true, yp), 4),
        "acc":     round(accuracy_score(y_true, y_pred), 4),
        "f1_mac":  round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "f1_yes":  round(f1_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
        "f1_no":   round(f1_score(y_true, y_pred, pos_label=0, zero_division=0), 4),
        "cm":      confusion_matrix(y_true, y_pred).tolist(),
        "y_prob":  yp,
    }


def print_metrics(r: dict):
    print(f"  {r['name']:<35}  AUC={r['auc']:.4f}  PR-AUC={r['pr_auc']:.4f}  "
          f"LogLoss={r['logloss']:.4f}  Brier={r['brier']:.4f}  "
          f"Acc={r['acc']:.4f}  F1(YES)={r['f1_yes']:.4f}  F1(NO)={r['f1_no']:.4f}")


def print_cm(r: dict):
    cm = np.array(r["cm"])
    print(f"  Confusion matrix  (rows=true, cols=pred):")
    print(f"          pred_NO  pred_YES")
    print(f"  true_NO  {cm[0,0]:6d}   {cm[0,1]:6d}")
    print(f"  true_YES {cm[1,0]:6d}   {cm[1,1]:6d}")


def print_feature_importance(r: dict, top_n: int = 15):
    imp   = r["feature_importances"]
    names = ALL_FEATURES
    pairs = sorted(zip(names, imp), key=lambda x: x[1], reverse=True)[:top_n]
    print(f"  Feature importance — top {top_n} ({r['name']}):")
    for fname, fi in pairs:
        bar = "#" * int(fi * 300)
        print(f"    {fname:<30} {fi:.4f}  {bar}")


# ── Random Forest ─────────────────────────────────────────────────────────────
def run_rf() -> dict:
    param_dist = {
        "n_estimators":     [100, 200, 300, 400, 500],
        "max_depth":        [None, 5, 8, 12, 15, 20],
        "min_samples_split": [2, 5, 10, 15, 20],
        "min_samples_leaf":  [1, 2, 4, 6, 8],
        "max_features":     ["sqrt", "log2", 0.5],
        "class_weight":     [None, "balanced"],
    }
    base = RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1)
    search = RandomizedSearchCV(
        base, param_dist,
        n_iter=N_ITER, cv=CV_FOLDS, scoring="roc_auc",
        random_state=RANDOM_SEED, n_jobs=1, refit=False,
    )
    search.fit(X_train, y_train)
    best_p = search.best_params_
    cv_auc = search.best_score_

    print(f"    RF best params: {best_p}")
    print(f"    RF CV AUC (train, {CV_FOLDS}-fold): {cv_auc:.4f}")

    # Refit on trainval with best params
    rf_final = RandomForestClassifier(**best_p, random_state=RANDOM_SEED, n_jobs=-1)
    rf_final.fit(X_trainval, y_trainval)

    prob = rf_final.predict_proba(X_test)[:, 1]
    pred = rf_final.predict(X_test)
    r = evaluate("RF", y_test, prob, pred)
    r["model"]               = rf_final
    r["best_params"]         = best_p
    r["cv_auc"]              = round(cv_auc, 4)
    r["feature_importances"] = rf_final.feature_importances_
    return r


# ── Gradient Boosting ─────────────────────────────────────────────────────────
def run_gb() -> dict:
    param_dist = {
        "n_estimators":     [100, 150, 200, 250, 300],
        "max_depth":        [2, 3, 4, 5, 6],
        "learning_rate":    [0.01, 0.03, 0.05, 0.1, 0.15, 0.2],
        "min_samples_split": [2, 5, 10, 15],
        "min_samples_leaf":  [1, 2, 4],
        "subsample":        [0.6, 0.7, 0.8, 0.9, 1.0],
        "max_features":     ["sqrt", "log2", None],
    }
    base = GradientBoostingClassifier(random_state=RANDOM_SEED)
    search = RandomizedSearchCV(
        base, param_dist,
        n_iter=N_ITER, cv=CV_FOLDS, scoring="roc_auc",
        random_state=RANDOM_SEED, n_jobs=1, refit=False,
    )
    search.fit(X_train, y_train)
    best_p = search.best_params_
    cv_auc = search.best_score_

    print(f"    GB best params: {best_p}")
    print(f"    GB CV AUC (train, {CV_FOLDS}-fold): {cv_auc:.4f}")

    gb_final = GradientBoostingClassifier(**best_p, random_state=RANDOM_SEED)
    gb_final.fit(X_trainval, y_trainval)

    prob = gb_final.predict_proba(X_test)[:, 1]
    pred = gb_final.predict(X_test)
    r = evaluate("GB (sklearn)", y_test, prob, pred)
    r["model"]               = gb_final
    r["best_params"]         = best_p
    r["cv_auc"]              = round(cv_auc, 4)
    r["feature_importances"] = gb_final.feature_importances_
    return r


# ── Figures ───────────────────────────────────────────────────────────────────
def plot_roc_all(all_results: list[dict], fname: str):
    colors  = ["#999", "#AAA", "#2196F3", "#FF5722", "#E91E63",
               "#9C27B0", "#FF9800", "#4CAF50", "#00BCD4", "#795548"]
    styles  = ["--", ":", "-.", "-", "-", "-", "-", "-", "-", "-"]
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, r in enumerate(all_results):
        fpr, tpr, _ = roc_curve(y_test, r["y_prob"])
        ax.plot(fpr, tpr, color=colors[i % len(colors)],
                ls=styles[i % len(styles)], lw=2,
                label=f"{r['name']}  (AUC={r['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Test Set (todas las fases)")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / fname, dpi=150)
    plt.close(fig)


def plot_feature_importance(r: dict, fname: str, top_n: int = 15):
    imp   = r["feature_importances"]
    names = ALL_FEATURES
    pairs = sorted(zip(names, imp), key=lambda x: x[1], reverse=True)[:top_n]
    names_sorted = [p[0] for p in pairs]
    imp_sorted   = [p[1] for p in pairs]

    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.4)))
    bars = ax.barh(names_sorted[::-1], imp_sorted[::-1], color="#2196F3", height=0.65)
    ax.set_xlabel("Importancia (Gini MDI)")
    ax.set_title(f"Feature Importance — {r['name']}", fontsize=11)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / fname, dpi=150)
    plt.close(fig)


def plot_comparison_bar(all_results: list[dict], fname: str):
    names = [r["name"] for r in all_results]
    aucs  = [r["auc"]  for r in all_results]
    cols  = ["#CCCCCC" if n.startswith("B") else
             ("#2196F3" if n.startswith("LR") else "#E91E63")
             for n in names]
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(len(names)), aucs, color=cols, width=0.6)
    ax.axhline(0.5, color="grey", lw=1, ls="--", alpha=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("AUC (Test Set)")
    ax.set_title("Comparacion AUC — todos los modelos Fases 5-6")
    ax.set_ylim(0.45, 0.95)
    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, auc + 0.003,
                f"{auc:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / fname, dpi=150)
    plt.close(fig)


# ── Save / report ─────────────────────────────────────────────────────────────
def _save_metrics(results: list[dict], tag: str):
    out = []
    for r in results:
        row = {k: v for k, v in r.items()
               if k not in ("y_prob", "model", "feature_importances")}
        if isinstance(row.get("best_params"), dict):
            row["best_params"] = {str(k): str(v) for k, v in row["best_params"].items()}
        out.append(row)
    with open(REPORTS_DIR / f"fase6_metrics_{tag}.json", "w") as f:
        json.dump(out, f, indent=2)


def _write_report(p5_results, p6_results, champion):
    all_r = p5_results + p6_results
    lines = [
        "# Fase 6 — Random Forest y Gradient Boosting\n",
        "## Metricas en Test Set (todas las fases)\n",
        "| Modelo | AUC | PR-AUC | Log-Loss | Brier | Acc | F1(YES) | F1(NO) |",
        "|--------|-----|--------|----------|-------|-----|---------|--------|",
    ]
    for r in all_r:
        lines.append(
            f"| {r['name']} | {r['auc']:.4f} | {r['pr_auc']:.4f} | "
            f"{r['logloss']:.4f} | {r['brier']:.4f} | {r['acc']:.4f} | "
            f"{r['f1_yes']:.4f} | {r['f1_no']:.4f} |"
        )
    lines += [
        f"\n**Campeon general:** {champion['name']}  AUC={champion['auc']:.4f}\n",
        "\n## Figuras\n",
        "- `reports/figures/roc_fases5_6.png`",
        "- `reports/figures/fi_rf.png`",
        "- `reports/figures/fi_gb.png`",
        "- `reports/figures/auc_comparacion.png`",
    ]
    with open(REPORTS_DIR / "fase6_rf_gb.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Load Phase 5 results (for comparison table)
    p5_path = REPORTS_DIR / "fase5_metrics_full.json"
    if p5_path.exists():
        with open(p5_path) as f:
            p5_raw = json.load(f)
        # Recover y_prob from test parquet — not possible without re-running
        # We use p5_raw for the table only (no ROC for them in combined plot)
        p5_results = p5_raw
    else:
        print("  AVISO: fase5_metrics_full.json no encontrado. Comparacion parcial.")
        p5_results = []

    print("\n" + "="*70)
    print("FASE 6 — RANDOM FOREST")
    print("="*70)
    print(f"  n_iter={N_ITER}, cv={CV_FOLDS}, scoring=roc_auc, seed={RANDOM_SEED}")
    print("  Buscando hiperparametros en train (684), refitting en trainval (760)...")
    rf_r = run_rf()
    print_metrics(rf_r)
    print_cm(rf_r)
    print_feature_importance(rf_r)

    print("\n" + "="*70)
    print("FASE 6 — GRADIENT BOOSTING")
    print("="*70)
    print(f"  n_iter={N_ITER}, cv={CV_FOLDS}, scoring=roc_auc, seed={RANDOM_SEED}")
    print("  Buscando hiperparametros en train (684), refitting en trainval (760)...")
    gb_r = run_gb()
    print_metrics(gb_r)
    print_cm(gb_r)
    print_feature_importance(gb_r)

    p6_results = [rf_r, gb_r]

    # ── Resumen comparativo ───────────────────────────────────────────────────
    b3_auc = next((r["auc"] for r in p5_results if "precio_fin" in r["name"]), None)

    print("\n" + "="*70)
    print("RESUMEN COMPARATIVO — TODAS LAS FASES")
    print("="*70)
    print(f"\n  {'Modelo':<40}  {'AUC':>6}  {'PR-AUC':>7}  {'LogLoss':>8}  {'Brier':>6}  {'F1-YES':>7}  {'F1-NO':>6}")
    print("  " + "-"*93)
    for r in p5_results:
        mark = " <-- B3" if "precio_fin" in r["name"] else ""
        print(f"  {r['name']:<40}  {r['auc']:>6.4f}  {r['pr_auc']:>7.4f}  "
              f"{r['logloss']:>8.4f}  {r['brier']:>6.4f}  {r['f1_yes']:>7.4f}  "
              f"{r['f1_no']:>6.4f}{mark}")
    print("  " + "-"*93)
    for r in p6_results:
        beat_b3 = " *** SUPERA B3" if b3_auc and r["auc"] > b3_auc else ""
        print(f"  {r['name']:<40}  {r['auc']:>6.4f}  {r['pr_auc']:>7.4f}  "
              f"{r['logloss']:>8.4f}  {r['brier']:>6.4f}  {r['f1_yes']:>7.4f}  "
              f"{r['f1_no']:>6.4f}{beat_b3}")

    # Champion across all
    all_numeric = [
        {"name": r["name"], "auc": r["auc"], "brier": r["brier"], "logloss": r["logloss"]}
        for r in p5_results + p6_results
    ]
    champion_auc   = max(all_numeric, key=lambda x: x["auc"])
    champion_brier = min(all_numeric, key=lambda x: x["brier"])
    champion_ll    = min(all_numeric, key=lambda x: x["logloss"])

    print(f"\n  Por AUC:     {champion_auc['name']}  ({champion_auc['auc']:.4f})")
    print(f"  Por Brier:   {champion_brier['name']}  ({champion_brier['brier']:.4f})")
    print(f"  Por LogLoss: {champion_ll['name']}  ({champion_ll['logloss']:.4f})")

    if b3_auc:
        for r in p6_results:
            if r["auc"] > b3_auc:
                print(f"\n  *** HALLAZGO DESTACADO: {r['name']} supera B3 (precio_fin)!")
                print(f"      {r['name']} AUC={r['auc']:.4f} vs B3 AUC={b3_auc:.4f}  "
                      f"(delta={r['auc']-b3_auc:+.4f})")

    # ── Figuras ───────────────────────────────────────────────────────────────
    print("\n  Generando figuras...")
    plot_feature_importance(rf_r, "fi_rf.png")
    plot_feature_importance(gb_r, "fi_gb.png")

    # Para ROC combinado necesitamos y_prob de P5 — re-computar rápido
    try:
        from src.models.phase5 import (
            run_baselines, run_lr_A, run_lr_MIN, run_lr_B, run_lr_C,
            X_train, y_train, X_val, y_val, X_test as X_test_p5, y_test as y_test_p5
        )
        p5_live = run_baselines()
        p5_live += [run_lr_A(), run_lr_MIN(), run_lr_B(), run_lr_C()]
        all_for_roc = p5_live + p6_results
        plot_roc_all(all_for_roc, "roc_fases5_6.png")
        print("  reports/figures/roc_fases5_6.png generado (con P5 re-computado)")
    except Exception as e:
        print(f"  AVISO: no se pudo re-computar ROC de P5 ({e}). Solo graficando P6.")
        plot_roc_all(p6_results, "roc_fases5_6.png")

    plot_comparison_bar(
        [r for r in p5_results + [rf_r, gb_r]],
        "auc_comparacion.png"
    )

    # ── Persistencia ─────────────────────────────────────────────────────────
    champion = max(p6_results, key=lambda r: r["auc"])
    with open(MODELS_DIR / "best_tree_model.pkl", "wb") as f:
        pickle.dump(champion["model"], f)

    _save_metrics(p6_results, "full")
    _write_report(p5_results, p6_results, champion)
    print("\nFase 6 completa. Modelos en models/, reportes en reports/")


if __name__ == "__main__":
    main()
