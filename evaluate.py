"""Stage 4 - report holdout performance (from train.py's saved holdout predictions).

Prints, per parameter and horizon:
  known stations vs NEW (never-seen) stations; skill vs persistence and vs climatology
  (positive = the model adds value over that naive baseline); interval coverage vs nominal;
  exceedance-classifier AUC. Also writes per-station errors and scatter plots.
"""
import argparse
import json
import logging

import numpy as np
import pandas as pd

import config

log = logging.getLogger("evaluate")


def run():
    with open(config.REPORT_DIR / "metrics.json") as f:
        metrics = json.load(f)
    rows = []
    for key, r in metrics.items():
        if r.get("status") != "trained":
            rows.append({"model": key, "status": r.get("status")})
            continue
        row = {"model": key, "algo": r["best_model"], "n_train": r["n_train"]}
        for split in ("known", "new"):
            s = r.get(split, {})
            row.update({f"{split}_n": s.get("n_test"), f"{split}_fallback": s.get("fallback_share"),
                        f"{split}_mae": s.get("mae"),
                        f"{split}_skill_pers": s.get("skill_vs_persistence"),
                        f"{split}_skill_clim": s.get("skill_vs_climatology"),
                        f"{split}_cover": s.get("interval_coverage"),
                        f"{split}_auc": r.get(f"{split}_classifier", {}).get("auc")})
        rows.append(row)
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(f"\n=== Holdout results (nominal interval coverage = {config.INTERVAL_COVERAGE}) ===")
    print(table.round(3).to_string(index=False))
    table.to_csv(config.REPORT_DIR / "evaluation_table.csv", index=False)

    path = config.REPORT_DIR / "holdout_predictions.csv"
    if not path.exists():
        return
    hp = pd.read_csv(path, dtype={"station_id": str}, parse_dates=["date"])
    if "fallback" in hp:
        hp = hp[~hp["fallback"].astype(bool)]        # score real forecasts only
    err = (hp.assign(abs_err=(hp["y_true"] - hp["y_pred"]).abs())
             .groupby(["station_id", "parameter", "horizon", "split"])["abs_err"]
             .agg(["count", "mean"]).rename(columns={"mean": "mae"}).reset_index())
    err.to_csv(config.REPORT_DIR / "station_errors.csv", index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    h = sorted(hp["horizon"].unique())[len(hp["horizon"].unique()) // 2]
    sub = hp[hp["horizon"] == h]
    params = sorted(sub["parameter"].unique())
    ncols = 3
    fig, axes = plt.subplots(-(-len(params) // ncols), ncols, figsize=(4.5 * ncols, 4 * -(-len(params) // ncols)),
                             squeeze=False)
    for ax, p in zip(axes.ravel(), params):
        for split, color in (("known", "tab:blue"), ("new", "tab:orange")):
            g = sub[(sub["parameter"] == p) & (sub["split"] == split)]
            ax.scatter(g["y_true"], g["y_pred"], s=7, alpha=0.4, c=color, label=split)
        g = sub[sub["parameter"] == p]
        lim = [np.nanmin([g.y_true.min(), g.y_pred.min()]), np.nanmax([g.y_true.max(), g.y_pred.max()])]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_title(f"{p} (h={h}d)")
        ax.set_xlabel("observed")
        ax.set_ylabel("predicted")
        ax.legend(fontsize=7)
    for ax in axes.ravel()[len(params):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(config.REPORT_DIR / "holdout_scatter.png", dpi=120)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    argparse.ArgumentParser(description=__doc__).parse_args()
    run()
