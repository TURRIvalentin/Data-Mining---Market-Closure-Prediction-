# Fase 6 — Random Forest y Gradient Boosting

## Metricas en Test Set (todas las fases)

| Modelo | AUC | PR-AUC | Log-Loss | Brier | Acc | F1(YES) | F1(NO) |
|--------|-----|--------|----------|-------|-----|---------|--------|
| B1: Mayoria (siempre NO) | 0.5000 | 0.1122 | 1.8084 | 0.1122 | 0.8878 | 0.0000 | 0.9406 |
| B2: Prior (12.4% YES) | 0.5000 | 0.1122 | 0.3518 | 0.0998 | 0.8878 | 0.0000 | 0.9406 |
| B3: precio_fin directo | 0.8471 | 0.6031 | 0.3697 | 0.1181 | 0.8293 | 0.4615 | 0.8986 |
| LR-A: 22f sin regularizacion | 0.8162 | 0.5451 | 0.2630 | 0.0743 | 0.9024 | 0.4737 | 0.9462 |
| LR-MIN: 4f (fin,media,tend,vol) | 0.8245 | 0.5680 | 0.2613 | 0.0731 | 0.9024 | 0.4444 | 0.9465 |
| LR-B: L2 C=50.0 | 0.8069 | 0.5555 | 0.2639 | 0.0730 | 0.9073 | 0.4242 | 0.9496 |
| LR-C: L1 C=0.5 | 0.8339 | 0.5824 | 0.2545 | 0.0715 | 0.9220 | 0.5000 | 0.9577 |
| LR-D: L1 C=0.5 balanced | 0.8073 | 0.5390 | 0.4874 | 0.1604 | 0.7756 | 0.3947 | 0.8623 |
| RF | 0.8876 | 0.6338 | 0.3414 | 0.1000 | 0.9024 | 0.5833 | 0.9448 |
| GB (sklearn) | 0.8933 | 0.6348 | 0.2436 | 0.0678 | 0.9024 | 0.3333 | 0.9474 |

**Campeon general:** GB (sklearn)  AUC=0.8933


## Figuras

- `reports/figures/roc_fases5_6.png`
- `reports/figures/fi_rf.png`
- `reports/figures/fi_gb.png`
- `reports/figures/auc_comparacion.png`