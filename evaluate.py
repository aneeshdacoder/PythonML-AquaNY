"""Stage 4 - evaluate the holdout predictions (never the refit deployment models).

Prints per-target skill vs. persistence, and for parameters with a screening
threshold, how well forecasts catch exceedances (precision / recall / F1).
"""
import argparse
import json
import logging

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

import config
import utils

log = logging.getLogger("evaluate")


def run():
    with open(config.REPORT_DIR / "metrics.json") as f:
        metrics = json.load(f)
    table = pd.DataFrame(metrics).T
    cols = [c for c in ("status", "best_model", "n_train", "n_test", "mae", "rmse", "r2",
                        "persistence_mae", "skill_vs_persistence") if c in table]
    print("\n=== Regression metrics (holdout) ===")
    print(table[cols].round(3).to_string())

    path = config.REPORT_DIR / "holdout_predictions.csv"
    if not path.exists():
        return
    hp = pd.read_csv(path, dtype={"site_no": str}, parse_dates=["date"])

    rows = []
    for target, g in hp.groupby("target"):
        spec = config.THRESHOLDS.get(target)
        if not spec:
            continue
        truth = utils.exceeds(g["y_true"], spec).to_numpy()
        pred = utils.exceeds(g["y_pred"], spec).to_numpy()
        rows.append({
            "target": target, "n": len(g), "actual_exceedances": int(truth.sum()),
            "precision": precision_score(truth, pred, zero_division=0),
            "recall": recall_score(truth, pred, zero_division=0),
            "f1": f1_score(truth, pred, zero_division=0)})
    if rows:
        ex = pd.DataFrame(rows)
        ex.to_csv(config.REPORT_DIR / "exceedance_report.csv", index=False)
        print("\n=== Threshold exceedance detection (holdout) ===")
        print(ex.round(3).to_string(index=False))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    targets = sorted(hp["target"].unique())
    ncols = 3
    nrows = -(-len(targets) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows), squeeze=False)
    for ax, t in zip(axes.ravel(), targets):
        g = hp[hp["target"] == t]
        ax.scatter(g["y_true"], g["y_pred"], s=8, alpha=0.5)
        lim = [min(g["y_true"].min(), g["y_pred"].min()), max(g["y_true"].max(), g["y_pred"].max())]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_title(t)
        ax.set_xlabel("observed")
        ax.set_ylabel("predicted")
    for ax in axes.ravel()[len(targets):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(config.REPORT_DIR / "holdout_scatter.png", dpi=120)
    log.info("Saved scatter plots")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    argparse.ArgumentParser(description=__doc__).parse_args()
    run()
