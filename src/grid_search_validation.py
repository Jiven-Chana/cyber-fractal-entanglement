#!/usr/bin/env python3
"""
GridSearch Validation

Runs CSeriesValidator.v2 over:
  - All single assets: SPX, GOLD, BTC
  - All pairs: SPX+GOLD, SPX+BTC, GOLD+BTC
  - All three: ANY

Across a grid of thresholds and horizons, collecting:
  - Best (min) Granger p-value
  - Logit coefficient p-value
  - Logit CV AUC

Outputs:
  - reports/grid/summary_table.csv
  - reports/grid/significant_runs.csv
  - reports/grid/<asset>_th<…>_h<…>/{granger_pvalues.csv,logit_summary.csv,roc_curve.png}
"""

import itertools
import logging
from pathlib import Path
import pandas as pd

# import validator class
from src.validate_ct_v2 import CSeriesValidator

# Config
ASSETS = {
    "SPX":     {"prices": "data/processed/prices.csv",            "asset": "SPX"},
    "GOLD":    {"prices": "data/processed/prices.csv",            "asset": "GOLD"},
    "BTC":     {"prices": "data/processed/prices.csv",            "asset": "BTC"},
    "ANY3":    {"prices": "data/processed/prices.csv",            "asset": "ANY"},
    "SPX_GOLD":{"prices": "data/processed/prices_SPX_GOLD.csv",  "asset": "ANY"},
    "SPX_BTC": {"prices": "data/processed/prices_SPX_BTC.csv",   "asset": "ANY"},
    "GOLD_BTC":{"prices": "data/processed/prices_GOLD_BTC.csv", "asset": "ANY"},
}

THRESHOLDS = [-0.01, -0.02, -0.03]
HORIZONS  = [3, 5, 10]
MAX_LAG   = 5
C_SERIES  = "data/processed/C_series_diff.csv"

OUT_ROOT = Path("reports/grid")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger("grid_validate")


def run_grid_search():
    records = []

    for label, cfg in ASSETS.items():
        for thr, hor in itertools.product(THRESHOLDS, HORIZONS):
            out_dir = OUT_ROOT / f"{label}_th{abs(int(thr*100))}_h{hor}"
            validator = CSeriesValidator(
                price_csv=cfg["prices"],
                c_csv=C_SERIES,
                output_dir=str(out_dir),
                asset=cfg["asset"],
                drawdown_horizon=hor,
                drawdown_threshold=thr,
                max_granger_lag=MAX_LAG,
            )

            logger.info(f"→ Running {label} | thr={thr} | hor={hor}")
            validator.run()

            # the validator writes into out_dir/<ASSET>/
            report_dir = out_dir / cfg["asset"].upper()

            # Collect summary stats
            gp = pd.read_csv(report_dir / "granger_pvalues.csv", index_col=0)
            best_gr = gp["pvalue"].min()

            lg = pd.read_csv(report_dir / "logit_summary.csv")
            logit_p = lg["coef_pvalue"].iloc[0]
            cv_auc   = lg["cv_auc"].iloc[0]

            records.append({
                "Basket":         label,
                "Threshold":      thr,
                "Horizon":        hor,
                "Best Granger p": best_gr,
                "Logit p-value":  logit_p,
                "Logit CV AUC":   cv_auc,
                "Report Dir":     str(report_dir)
            })

    # Build master table
    df = pd.DataFrame(records)
    df.sort_values(["Basket", "Threshold", "Horizon"], inplace=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_ROOT / "summary_table.csv"
    df.to_csv(summary_path, index=False)
    logger.info(f"Saved master summary → {summary_path}")

    # Extract statistically significant runs
    sig = df[(df["Best Granger p"] < 0.05) & (df["Logit p-value"] < 0.05)]
    sig_path = OUT_ROOT / "significant_runs.csv"
    sig.to_csv(sig_path, index=False)
    logger.info(f"Saved significant runs → {sig_path}")

    print("\n=== GRID SEARCH COMPLETE ===")
    print(f"Total runs: {len(df)}")
    print(f"Significant runs: {len(sig)}")
    print(f"See:\n  • {summary_path}\n  • {sig_path}")


if __name__ == "__main__":
    run_grid_search()