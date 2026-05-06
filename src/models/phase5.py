"""
Fase 5 — Baselines + Regresión Logística.

Uso:
    python -m src.models.phase5 --checkpoint   # baselines + LR-A
    python -m src.models.phase5 --full         # todas las variantes (A, MIN, B, C, D)
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score, log_loss, brier_score_loss,
    f1_score, confusion_matrix, roc_curve, average_precision_score,
)

PROCESSED   = Path("data/processed")
MODELS_DIR  = Path("models");  MODELS_DIR.mkdir(exist_ok=True)
FIGURES_DIR = Path("reports/figures"); FIGURES_DIR.mkdir(exist_ok=True)
REPORTS_DIR = Path("reports"); REPORTS_DIR.mkdir(exist_ok=True)
RANDOM_SEED = 42

# ── Load data ─────────────────────────────────────────────────────────────────
with open(PROCESSED / "feature_columns.json") as f:
    fc = json.load(f)

ALL_FEATURES = fc["all_features"]      # 22 features (scaled)
NUM_COLS     = fc["numeric"]           # 15 numeric (scaled)
TARGET       = fc["target"]

train = pd.read_parquet(PROCESSED / "train.parquet")
val   = pd.read_parquet(PROCESSED / "val.parquet")
test  = pd.read_parquet(PROCESSED / "test.parquet")

trainval = pd.concat([train, val], ignore_index=True)

X_train    = train[ALL_FEATURES].values
y_train    = train[TARGET].values
X_val      = val[ALL_FEATURES].values
y_val      = val[TARGET].values
X_test     = test[ALL_FEATURES].values
y_test     = test[TARGET].values
X_trainval = trainval[ALL_FEATURES].values
y_trainval = trainval[TARGET].values

# ── LR-MIN: 4-feature minimal set ────────────────────────────────────────────
MIN_FEATURES = ["precio_fin", "precio_media", "precio_tendencia", "volatilidad_retornos"]
_min_idx     = [ALL_FEATURES.index(f) for f in MIN_FEATURES]

X_train_min    = X_train[:, _min_idx]
X_val_min      = X_val[:, _min_idx]
X_test_min     = X_test[:, _min_idx]
X_trainval_min = X_trainval[:, _min_idx]

# ── Recover original precio_fin (for Baseline 3) ─────────────────────────────
with open(PROCESSED / "scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

pf_idx = NUM_COLS.index("precio_fin")
def recover_precio_fin(df: pd.DataFrame) -> np.ndarray:
    scaled   = df["precio_fin"].values
    original = scaled * scaler.scale_[pf_idx] + scaler.mean_[pf_idx]
    return np.clip(original, 0.0, 1.0)

pf_train = recover_precio_fin(train)
pf_val   = recover_precio_fin(val)
pf_test  = recover_precio_fin(test)


# ── Metrics helper ────────────────────────────────────────────────────────────
def evaluate(name: str, y_true: np.ndarray,
             y_prob: np.ndarray | None,
             y_pred: np.ndarray | None = None,
             threshold: float = 0.5) -> dict:
    if y_prob is None:
        y_prob = (y_pred == 1).astype(float)
    if y_pred is None:
        y_pred = (y_prob >= threshold).astype(int)
    y_prob_c = np.clip(y_prob, 1e-7, 1 - 1e-7)

    return {
        "name":    name,
        "auc":     round(roc_auc_score(y_true, y_prob_c), 4),
        "pr_auc":  round(average_precision_score(y_true, y_prob_c), 4),
        "logloss": round(log_loss(y_true, y_prob_c), 4),
        "brier":   round(brier_score_loss(y_true, y_prob_c), 4),
        "acc":     round(accuracy_score(y_true, y_pred), 4),
        "f1_mac":  round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "f1_yes":  round(f1_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
        "f1_no":   round(f1_score(y_true, y_pred, pos_label=0, zero_division=0), 4),
        "cm":      confusion_matrix(y_true, y_pred).tolist(),
        "y_prob":  y_prob_c,
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


def print_coefficients(r: dict, feature_names: list[str]):
    model  = r["model"]
    coefs  = model.coef_[0]
    pairs  = sorted(zip(feature_names, coefs), key=lambda x: abs(x[1]), reverse=True)
    print(f"  Coeficientes ({r['name']}) — ordenados por magnitud:")
    for fname, c in pairs:
        bar = "#" * int(abs(c) * 30)
        sign = "+" if c >= 0 else "-"
        print(f"    {fname:<30} {sign}{abs(c):6.4f}  {bar}")
    if hasattr(model, "coef_"):
        n_nonzero = int((coefs != 0).sum())
        if n_nonzero < len(feature_names):
            print(f"  Features activas: {n_nonzero}/{len(feature_names)}")


# ── Baselines ─────────────────────────────────────────────────────────────────
def run_baselines() -> list[dict]:
    results = []

    dummy_maj = DummyClassifier(strategy="most_frequent", random_state=RANDOM_SEED)
    dummy_maj.fit(X_train, y_train)
    results.append(evaluate("B1: Mayoria (siempre NO)", y_test,
                             dummy_maj.predict_proba(X_test)[:, 1],
                             dummy_maj.predict(X_test)))

    dummy_prior = DummyClassifier(strategy="prior", random_state=RANDOM_SEED)
    dummy_prior.fit(X_train, y_train)
    results.append(evaluate("B2: Prior (12.4% YES)", y_test,
                             dummy_prior.predict_proba(X_test)[:, 1],
                             dummy_prior.predict(X_test)))

    prob_pf = pf_test
    results.append(evaluate("B3: precio_fin directo", y_test,
                             prob_pf, (prob_pf >= 0.5).astype(int)))
    return results


# ── Logistic Regression ───────────────────────────────────────────────────────
def run_lr_A() -> dict:
    lr = LogisticRegression(C=1e4, max_iter=2000, random_state=RANDOM_SEED,
                            solver="lbfgs", class_weight=None)
    lr.fit(X_train, y_train)
    r = evaluate("LR-A: 22f sin regularizacion", y_test,
                 lr.predict_proba(X_test)[:, 1], lr.predict(X_test))
    r["model"] = lr
    return r


def run_lr_MIN() -> dict:
    """Logística mínima: solo 4 features de precio."""
    lr = LogisticRegression(C=1e4, max_iter=2000, random_state=RANDOM_SEED,
                            solver="lbfgs", class_weight=None)
    lr.fit(X_train_min, y_train)
    r = evaluate("LR-MIN: 4f (fin,media,tend,vol)", y_test,
                 lr.predict_proba(X_test_min)[:, 1], lr.predict(X_test_min))
    r["model"]    = lr
    r["features"] = MIN_FEATURES
    return r


def run_lr_B() -> dict:
    """L2, C buscado por val AUC, refit en trainval."""
    Cs = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]
    best_C, best_auc = Cs[0], -1.0
    for C in Cs:
        m = LogisticRegression(C=C, max_iter=2000, random_state=RANDOM_SEED,
                               solver="lbfgs")
        m.fit(X_train, y_train)
        auc = roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])
        if auc > best_auc:
            best_auc, best_C = auc, C

    lr = LogisticRegression(C=best_C, max_iter=2000, random_state=RANDOM_SEED,
                            solver="lbfgs")
    lr.fit(X_trainval, y_trainval)
    r = evaluate(f"LR-B: L2 C={best_C}", y_test,
                 lr.predict_proba(X_test)[:, 1], lr.predict(X_test))
    r["model"]  = lr
    r["best_C"] = best_C
    print(f"    LR-B: best C={best_C}  val AUC={best_auc:.4f}")
    return r


def run_lr_C() -> dict:
    """L1 (saga), C buscado por val AUC, refit en trainval."""
    Cs = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
    best_C, best_auc = Cs[0], -1.0
    for C in Cs:
        m = LogisticRegression(C=C, max_iter=5000, random_state=RANDOM_SEED,
                               l1_ratio=1.0, solver="saga")
        m.fit(X_train, y_train)
        auc = roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])
        if auc > best_auc:
            best_auc, best_C = auc, C

    lr = LogisticRegression(C=best_C, max_iter=5000, random_state=RANDOM_SEED,
                            l1_ratio=1.0, solver="saga")
    lr.fit(X_trainval, y_trainval)
    n_nonzero = int((lr.coef_[0] != 0).sum())
    r = evaluate(f"LR-C: L1 C={best_C}", y_test,
                 lr.predict_proba(X_test)[:, 1], lr.predict(X_test))
    r["model"]      = lr
    r["best_C"]     = best_C
    r["n_nonzero"]  = n_nonzero
    print(f"    LR-C: best C={best_C}  val AUC={best_auc:.4f}  features activas={n_nonzero}/{len(ALL_FEATURES)}")
    return r


def run_lr_D(base_lr: dict) -> dict:
    """Mejor LR (B o C) con class_weight='balanced'."""
    best_C   = base_lr.get("best_C", 1.0)
    params   = base_lr["model"].get_params()
    l1_ratio = params.get("l1_ratio", 0.0)
    is_l1    = l1_ratio is not None and l1_ratio >= 0.5
    solver   = "saga" if is_l1 else "lbfgs"
    max_it   = 5000 if is_l1 else 2000

    lr = LogisticRegression(C=best_C, max_iter=max_it, random_state=RANDOM_SEED,
                            l1_ratio=l1_ratio, solver=solver,
                            class_weight="balanced")
    lr.fit(X_trainval, y_trainval)
    reg_label = "L1" if is_l1 else "L2"
    r = evaluate(f"LR-D: {reg_label} C={best_C} balanced", y_test,
                 lr.predict_proba(X_test)[:, 1], lr.predict(X_test))
    r["model"]  = lr
    r["best_C"] = best_C
    return r


# ── Figures ───────────────────────────────────────────────────────────────────
def plot_roc_curves(results: list[dict], fname: str):
    colors  = ["#999999", "#AAAAAA", "#2196F3", "#FF5722", "#E91E63",
               "#9C27B0", "#FF9800", "#4CAF50"]
    styles  = ["--", ":", "-.", "-", "-", "-", "-", "-"]
    fig, ax = plt.subplots(figsize=(9, 7))
    for i, r in enumerate(results):
        fpr, tpr, _ = roc_curve(y_test, r["y_prob"])
        ax.plot(fpr, tpr, color=colors[i % len(colors)],
                ls=styles[i % len(styles)], lw=2,
                label=f"{r['name']}  (AUC={r['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Test Set")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / fname, dpi=150)
    plt.close(fig)


def plot_confusion_matrices(results: list[dict], fname: str):
    n   = len(results)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]
    for ax, r in zip(axes_flat, results):
        cm = np.array(r["cm"])
        ax.imshow(cm, cmap="Blues")
        for row in range(2):
            for col in range(2):
                ax.text(col, row, str(cm[row, col]), ha="center", va="center",
                        fontsize=14, fontweight="bold",
                        color="white" if cm[row, col] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["NO", "YES"]); ax.set_yticklabels(["NO", "YES"])
        ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
        ax.set_title(f"{r['name'].split(':')[0]}\nAUC={r['auc']:.3f}", fontsize=9)
    for ax in axes_flat[n:]:
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / fname, dpi=150)
    plt.close(fig)


def plot_reliability_champion_vs_b3(champion: dict, b3: dict, fname: str):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, r in zip(axes, [b3, champion]):
        prob_true, prob_pred = calibration_curve(y_test, r["y_prob"], n_bins=8)
        ax.plot(prob_pred, prob_true, "o-", lw=2, label="Calibracion real")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfecta")
        ax.fill_between([0, 1], [0, 1], alpha=0.05, color="grey")
        ax.set_xlabel("Prob. predicha"); ax.set_ylabel("Fraccion positivos")
        ax.set_title(f"{r['name']}\nBrier={r['brier']:.4f}", fontsize=10)
        ax.legend(fontsize=8); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.suptitle("Reliability Diagram: Campeon vs B3", fontsize=12)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / fname, dpi=150)
    plt.close(fig)


def plot_coefficients(lr_result: dict, fname: str):
    model    = lr_result["model"]
    feat_names = lr_result.get("features", ALL_FEATURES)
    coefs    = model.coef_[0]
    df_coef  = pd.DataFrame({"feature": feat_names, "coef": coefs})
    df_coef  = df_coef.sort_values("coef")

    fig, ax = plt.subplots(figsize=(10, max(5, len(feat_names) * 0.45)))
    colors = ["#E91E63" if c > 0 else "#2196F3" for c in df_coef["coef"]]
    ax.barh(df_coef["feature"], df_coef["coef"], color=colors, height=0.6)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Coeficiente (espacio normalizado)")
    ax.set_title(f"Coeficientes — {lr_result['name']}", fontsize=11)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / fname, dpi=150)
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="store_true")
    parser.add_argument("--full",       action="store_true")
    args = parser.parse_args()
    if not args.checkpoint and not args.full:
        args.checkpoint = True

    all_results = []

    # ── Baselines ────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("BASELINES")
    print("="*70)
    baselines = run_baselines()
    for r in baselines:
        print_metrics(r)
        print_cm(r)
        print()
    all_results.extend(baselines)

    # ── LR-A ─────────────────────────────────────────────────────────────────
    print("="*70)
    print("LOGISTIC REGRESSION")
    print("="*70)
    print("\n  Variante A (22 features, sin regularizacion)...")
    lr_a = run_lr_A()
    print_metrics(lr_a)
    print_cm(lr_a)
    print_coefficients(lr_a, ALL_FEATURES)
    all_results.append(lr_a)

    b3_auc = baselines[2]["auc"]
    print(f"\n  DELTA LR-A vs B3: AUC {lr_a['auc'] - b3_auc:+.4f}")

    if args.checkpoint:
        plot_roc_curves(all_results, "roc_checkpoint.png")
        plot_confusion_matrices(all_results, "confusion_checkpoint.png")
        print(f"\n  Figuras: reports/figures/roc_checkpoint.png")
        print("\n[CHECKPOINT] Baselines + LR-A completados. Esperando aprobacion.")
        _save_metrics(all_results, "checkpoint")
        return all_results

    # ── LR-MIN ───────────────────────────────────────────────────────────────
    print("\n" + "-"*60)
    print("  Variante MIN (4 features, sin regularizacion)...")
    lr_min = run_lr_MIN()
    print_metrics(lr_min)
    print_cm(lr_min)
    print_coefficients(lr_min, MIN_FEATURES)
    print(f"  DELTA LR-MIN vs B3: AUC {lr_min['auc'] - b3_auc:+.4f}")
    print(f"  DELTA LR-MIN vs LR-A: AUC {lr_min['auc'] - lr_a['auc']:+.4f}")
    all_results.append(lr_min)

    # ── LR-B ─────────────────────────────────────────────────────────────────
    print("\n" + "-"*60)
    print("  Variante B (22 features, L2 con CV)...")
    lr_b = run_lr_B()
    print_metrics(lr_b)
    print_cm(lr_b)
    print_coefficients(lr_b, ALL_FEATURES)
    print(f"  DELTA LR-B vs B3: AUC {lr_b['auc'] - b3_auc:+.4f}")
    all_results.append(lr_b)

    # ── LR-C ─────────────────────────────────────────────────────────────────
    print("\n" + "-"*60)
    print("  Variante C (22 features, L1 con CV)...")
    lr_c = run_lr_C()
    print_metrics(lr_c)
    print_cm(lr_c)
    print_coefficients(lr_c, ALL_FEATURES)
    print(f"  DELTA LR-C vs B3: AUC {lr_c['auc'] - b3_auc:+.4f}")
    all_results.append(lr_c)

    # ── LR-D ─────────────────────────────────────────────────────────────────
    print("\n" + "-"*60)
    best_bc = lr_b if lr_b["auc"] >= lr_c["auc"] else lr_c
    print(f"  Variante D (balanced, base={best_bc['name']})...")
    lr_d = run_lr_D(best_bc)
    print_metrics(lr_d)
    print_cm(lr_d)
    print_coefficients(lr_d, ALL_FEATURES)

    # Mostrar cambio de confusion matrix vs mismo modelo sin balanceo
    cm_base = np.array(best_bc["cm"])
    cm_bal  = np.array(lr_d["cm"])
    print(f"\n  Cambio CM vs {best_bc['name']} sin balanceo:")
    print(f"    NO->NO:  {cm_base[0,0]:3d} -> {cm_bal[0,0]:3d}  ({cm_bal[0,0]-cm_base[0,0]:+d})")
    print(f"    NO->YES: {cm_base[0,1]:3d} -> {cm_bal[0,1]:3d}  ({cm_bal[0,1]-cm_base[0,1]:+d})")
    print(f"    YES->NO: {cm_base[1,0]:3d} -> {cm_bal[1,0]:3d}  ({cm_bal[1,0]-cm_base[1,0]:+d})")
    print(f"    YES->YES:{cm_base[1,1]:3d} -> {cm_bal[1,1]:3d}  ({cm_bal[1,1]-cm_base[1,1]:+d})")
    all_results.append(lr_d)

    # ── Resumen final ─────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("RESUMEN COMPARATIVO COMPLETO")
    print("="*70)
    print(f"\n  {'Modelo':<40}  {'AUC':>6}  {'PR-AUC':>7}  {'LogLoss':>8}  {'Brier':>6}  {'F1-YES':>7}  {'F1-NO':>6}")
    print("  " + "-"*90)
    for r in all_results:
        marker = " <-- campeon?" if r["auc"] == max(x["auc"] for x in all_results[2:]) else ""
        print(f"  {r['name']:<40}  {r['auc']:>6.4f}  {r['pr_auc']:>7.4f}  "
              f"{r['logloss']:>8.4f}  {r['brier']:>6.4f}  {r['f1_yes']:>7.4f}  "
              f"{r['f1_no']:>6.4f}{marker}")

    # ── Champion selection ────────────────────────────────────────────────────
    lr_variants = [lr_a, lr_min, lr_b, lr_c, lr_d]
    champion    = max(lr_variants, key=lambda r: r["auc"])
    print(f"\n  Campeon por AUC: {champion['name']}  (AUC={champion['auc']:.4f})")
    print(f"  Delta vs B3 (precio_fin): {champion['auc'] - b3_auc:+.4f}")

    # ── Hallazgos ─────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("HALLAZGOS PARA EL INFORME")
    print("="*70)
    lr_a_auc = lr_a["auc"];  lr_min_auc = lr_min["auc"]
    lr_b_auc = lr_b["auc"];  lr_c_auc   = lr_c["auc"];  lr_d_auc = lr_d["auc"]

    findings = [
        f"1. baseline competitivo: precio_fin directo (B3) AUC={b3_auc:.4f}, "
        f"supera a LR sin regularizacion ({lr_a_auc:.4f}, delta={lr_a_auc-b3_auc:+.4f})",
        f"2. hipotesis minima: LR-MIN 4 features AUC={lr_min_auc:.4f} "
        f"(delta vs B3={lr_min_auc-b3_auc:+.4f}, delta vs LR-A={lr_min_auc-lr_a_auc:+.4f})",
        f"3. regularizacion L2: LR-B AUC={lr_b_auc:.4f} (delta vs B3={lr_b_auc-b3_auc:+.4f})",
        f"4. regularizacion L1: LR-C AUC={lr_c_auc:.4f}, {lr_c.get('n_nonzero','?')}/{len(ALL_FEATURES)} features activas",
        f"5. clase balanceada: LR-D AUC={lr_d_auc:.4f}, F1-YES={lr_d['f1_yes']:.4f} vs "
        f"base F1-YES={best_bc['f1_yes']:.4f}",
        f"6. campeon: {champion['name']}  AUC={champion['auc']:.4f}",
    ]
    for f in findings:
        print(f"  {f}")

    # ── Figuras ───────────────────────────────────────────────────────────────
    print("\n  Generando figuras...")
    plot_roc_curves(all_results, "roc_fase5.png")
    plot_confusion_matrices(all_results, "confusion_fase5.png")
    plot_reliability_champion_vs_b3(champion, baselines[2], "reliability_campeon_vs_b3.png")
    plot_coefficients(champion, "coeficientes_campeon.png")
    plot_coefficients(lr_min,   "coeficientes_lr_min.png")
    print("  reports/figures/{roc_fase5, confusion_fase5, reliability_campeon_vs_b3, coeficientes_campeon, coeficientes_lr_min}.png")

    # ── Save ──────────────────────────────────────────────────────────────────
    with open(MODELS_DIR / "best_lr.pkl", "wb") as f:
        pickle.dump(champion["model"], f)

    _save_metrics(all_results, "full")
    _write_report(all_results, champion, lr_c, best_bc, lr_d)
    _error_analysis(champion)

    print("\nFase 5 completa.")
    return all_results


def _save_metrics(results: list[dict], tag: str):
    out = []
    for r in results:
        row = {k: v for k, v in r.items() if k not in ("y_prob", "model")}
        out.append(row)
    with open(REPORTS_DIR / f"fase5_metrics_{tag}.json", "w") as f:
        json.dump(out, f, indent=2)


def _error_analysis(lr_result: dict):
    feat = lr_result.get("features", ALL_FEATURES)
    if feat is ALL_FEATURES:
        y_prob = lr_result["y_prob"]
    else:
        y_prob = lr_result["y_prob"]

    y_pred = (y_prob >= 0.5).astype(int)
    meta   = test[["condition_id", "category_coarse", "bucket"]].copy()
    meta["y_true"]     = y_test
    meta["y_pred"]     = y_pred
    meta["y_prob"]     = y_prob
    meta["error_type"] = "correct"
    meta.loc[(meta["y_true"]==1) & (meta["y_pred"]==0), "error_type"] = "FN"
    meta.loc[(meta["y_true"]==0) & (meta["y_pred"]==1), "error_type"] = "FP"

    print("\n" + "="*70)
    print(f"ANALISIS DE ERRORES — {lr_result['name']}")
    print("="*70)
    print("\n  Distribucion:")
    print(meta["error_type"].value_counts().to_string())
    print("\n  Errores por categoria:")
    errors = meta[meta["error_type"] != "correct"]
    if len(errors):
        print(pd.crosstab(errors["category_coarse"], errors["error_type"]).to_string())
    print("\n  Peores FN (YES pred como NO):")
    for _, row in meta[meta["error_type"]=="FN"].nsmallest(5, "y_prob").iterrows():
        print(f"    p={row['y_prob']:.3f}  cat={row['category_coarse']}")
    print("\n  Peores FP (NO pred como YES):")
    for _, row in meta[meta["error_type"]=="FP"].nlargest(5, "y_prob").iterrows():
        print(f"    p={row['y_prob']:.3f}  cat={row['category_coarse']}")


def _write_report(results, champion, lr_c, best_bc, lr_d):
    lines = [
        "# Fase 5 — Baselines y Regresion Logistica\n",
        "## Metricas en Test Set\n",
        "| Modelo | AUC | PR-AUC | Log-Loss | Brier | Acc | F1(YES) | F1(NO) |",
        "|--------|-----|--------|----------|-------|-----|---------|--------|",
    ]
    for r in results:
        lines.append(
            f"| {r['name']} | {r['auc']:.4f} | {r['pr_auc']:.4f} | "
            f"{r['logloss']:.4f} | {r['brier']:.4f} | {r['acc']:.4f} | "
            f"{r['f1_yes']:.4f} | {r['f1_no']:.4f} |"
        )
    lines += [
        f"\n**Campeon:** {champion['name']}  AUC={champion['auc']:.4f}\n",
        "\n## Figuras\n",
        "- `reports/figures/roc_fase5.png`",
        "- `reports/figures/confusion_fase5.png`",
        "- `reports/figures/reliability_campeon_vs_b3.png`",
        "- `reports/figures/coeficientes_campeon.png`",
        "- `reports/figures/coeficientes_lr_min.png`",
    ]
    with open(REPORTS_DIR / "fase5_baselines_logistica.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
