# src/validate_ct.py

"""
CSeriesValidator v2: validate that systemic-entanglement score C(t)
predicts future market drawdowns across single or multiple assets.

Validation steps:
  1. Load price series and C(t) series.
  2. Compute binary drawdown labels for one asset or cross-asset ("ANY"):
     event if drop ≥ threshold within horizon days.
  3. Align C(t) and labels on common dates.
  4. Run Granger causality tests (lags 1..max_lag):
     → reports/validation/<asset>/granger_pvalues.csv
  5. Compute ROC curve and AUC:
     → reports/validation/<asset>/roc_curve.png
  6. Fit logistic regression:
     → reports/validation/<asset>/logit_summary.csv
"""

import logging
from pathlib import Path
from typing import Dict, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.metrics import roc_curve, auc
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


class CSeriesValidator:
    def __init__(
        self,
        price_csv: str = "data/processed/prices.csv",
        c_csv: str = "data/processed/C_series.csv",
        output_dir: str = "reports/validation",
        asset: str = "SPX",
        drawdown_horizon: int = 5,
        drawdown_threshold: float = -0.05,
        max_granger_lag: int = 5,
    ) -> None:
        self.price_csv = Path(price_csv)
        self.c_csv = Path(c_csv)
        self.asset = asset.upper()
        self.output_dir = Path(output_dir) / self.asset
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.horizon = drawdown_horizon
        self.threshold = drawdown_threshold
        self.max_lag = max_granger_lag

        # Logger setup
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        if not self.logger.handlers:
            self.logger.addHandler(handler)

    def load_series(self) -> None:
        """Load price(s) and C(t) series."""
        if not self.price_csv.exists():
            raise FileNotFoundError(f"Prices file not found: {self.price_csv}")
        if not self.c_csv.exists():
            raise FileNotFoundError(f"C_series file not found: {self.c_csv}")

        df_price = pd.read_csv(self.price_csv, index_col=0, parse_dates=True)

        # select single-asset or full DF
        if self.asset == "ANY":
            self.prices = df_price.copy()
        else:
            if self.asset not in df_price.columns:
                raise KeyError(f"Asset '{self.asset}' not in prices CSV")
            self.prices = df_price[[self.asset]].copy()

        df_c = pd.read_csv(self.c_csv, index_col=0, parse_dates=True)
        if "C" not in df_c.columns:
            raise KeyError("Column 'C' not found in C_series")
        self.C = df_c["C"].copy()

        self.logger.info(f"Loaded prices {self.prices.shape} and C-series ({len(self.C)})")

    def compute_drawdown_labels(self) -> None:
        """
        Compute binary labels:
          - For single-asset: event if that asset drops ≥ threshold.
          - For ANY: event if ANY asset drops ≥ threshold.
        """
        price = self.prices
        peak = price.cummax()
        labels = pd.Series(False, index=price.index)

        for i in range(len(price) - self.horizon):
            curr_peak = peak.iloc[i]
            future = price.iloc[i + 1 : i + 1 + self.horizon]
            dd = (future - curr_peak) / curr_peak

            if isinstance(dd, pd.DataFrame):
                triggered = dd.le(self.threshold).any(axis=1).any()
            else:
                triggered = dd.le(self.threshold).any()

            labels.iloc[i] = bool(triggered)

        self.labels = labels
        self.logger.info(f"Computed drawdown labels for asset={self.asset}")

    def align_data(self) -> None:
        """Align C(t) and label series on common index and drop NaNs."""
        df = pd.DataFrame({"C": self.C, "label": self.labels.astype(int)})
        df = df.dropna()
        self.df = df
        self.logger.info(f"Aligned data: {len(df)} observations")

    def run_granger(self) -> Dict[int, float]:
        """Run Granger causality and save p-values."""
        data = self.df[["label", "C"]]
        pvals: Dict[int, float] = {}
        for lag in range(1, self.max_lag + 1):
            res = grangercausalitytests(data, maxlag=lag, verbose=False)
            pval = res[lag][0]["ssr_ftest"][1]
            pvals[lag] = pval
            self.logger.info(f"Lag={lag} p-value={pval:.4f}")
        pd.Series(pvals).to_csv(self.output_dir / "granger_pvalues.csv", header=["pvalue"])
        return pvals

    def roc_analysis(self) -> float:
        """Compute ROC AUC and save curve."""
        y = self.df["label"]
        scores = self.df["C"]
        fpr, tpr, _ = roc_curve(y, scores)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
        plt.plot([0,1],[0,1],"--",color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC: C(t) vs drawdown [{self.asset}]")
        plt.legend(loc="lower right")
        plt.savefig(self.output_dir / "roc_curve.png", dpi=300)
        plt.close()

        self.logger.info(f"Saved ROC curve (AUC={roc_auc:.3f})")
        return roc_auc

    def logistic_regression(self) -> Dict[str, float]:
        """Fit logistic model; save p-value & CV AUC."""
        X = sm.add_constant(self.df["C"])
        logit = sm.Logit(self.df["label"], X).fit(disp=False)
        pvalue = logit.pvalues["C"]

        model = LogisticRegression()
        cv_scores = cross_val_score(model, self.df[["C"]], self.df["label"],
                                    cv=5, scoring="roc_auc")
        cv_auc = float(cv_scores.mean())

        pd.DataFrame({
            "coef_pvalue": [pvalue],
            "cv_auc":      [cv_auc]
        }).to_csv(self.output_dir / "logit_summary.csv", index=False)

        self.logger.info(f"Logit p-value={pvalue:.4f}, CV AUC={cv_auc:.3f}")
        return {"pvalue": pvalue, "cv_auc": cv_auc}

    def run(self) -> None:
        """Execute full pipeline."""
        self.load_series()
        self.compute_drawdown_labels()
        self.align_data()
        self.run_granger()
        self.roc_analysis()
        self.logistic_regression()
        self.logger.info(f"Done asset={self.asset}. Reports in {self.output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate C(t) vs drawdown (v2)")
    parser.add_argument("--prices",   default="data/processed/prices.csv",
                        help="Prices CSV with columns SPX,GOLD,BTC")
    parser.add_argument("--c_series", default="data/processed/C_series_diff.csv",
                        help="Differenced C(t) series CSV")
    parser.add_argument("--output_dir",default="reports/validation",
                        help="Base directory for validation outputs")
    parser.add_argument("--asset",    choices=["SPX","GOLD","BTC","ANY"], default="SPX",
                        help="Single asset or ANY for cross-asset")
    parser.add_argument("--horizon",  type=int, default=5,
                        help="Lookahead horizon (days)")
    parser.add_argument("--threshold",type=float, default=-0.05,
                        help="Drawdown threshold (e.g., -0.05 for 5%)")
    parser.add_argument("--max_lag",  type=int, default=5,
                        help="Max lag for Granger causality")

    args = parser.parse_args()

    validator = CSeriesValidator(
        price_csv=args.prices,
        c_csv=args.c_series,
        output_dir=args.output_dir,
        asset=args.asset,
        drawdown_horizon=args.horizon,
        drawdown_threshold=args.threshold,
        max_granger_lag=args.max_lag
    )
    validator.run()