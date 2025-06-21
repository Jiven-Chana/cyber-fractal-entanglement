# src/validate_ct.py

"""
CSeriesValidator: validate that systemic-entanglement score C(t)
predicts future market drawdowns.

Validation steps:
  1. Load price series and C(t) series.
  2. Compute binary drawdown labels: event if drawdown ≥ threshold within horizon days.
  3. Align C(t) and labels on common dates.
  4. Run Granger causality tests (lags 1..max_lag) → reports/validation/granger_pvalues.csv
  5. Compute ROC curve and AUC → reports/validation/roc_curve.png
  6. Fit logistic regression → reports/validation/logit_summary.csv
"""

import logging
from pathlib import Path
from typing import Dict

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
        drawdown_horizon: int = 5,
        drawdown_threshold: float = -0.05,
        max_granger_lag: int = 5,
    ) -> None:
        self.price_csv = Path(price_csv)
        self.c_csv = Path(c_csv)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.horizon = drawdown_horizon
        self.threshold = drawdown_threshold
        self.max_lag = max_granger_lag

        # Logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        )
        if not self.logger.handlers:
            self.logger.addHandler(handler)

    def load_series(self) -> None:
        """Load price and C(t) series."""
        if not self.price_csv.exists() or not self.c_csv.exists():
            raise FileNotFoundError("Price or C_series CSV not found.")
        df_price = pd.read_csv(self.price_csv, index_col=0, parse_dates=True)
        self.prices = df_price["SPX"].copy()

        df_C = pd.read_csv(self.c_csv, index_col=0, parse_dates=True)
        self.C = df_C["C"].copy()

        self.logger.info(f"Loaded prices ({len(self.prices)}) and C-series ({len(self.C)})")

    def compute_drawdown_labels(self) -> None:
        """Compute binary labels: True if within next horizon days a drawdown ≥ threshold occurs."""
        price = self.prices
        peak = price.cummax()
        labels = pd.Series(False, index=price.index)

        for i in range(len(price) - self.horizon):
            current_peak = peak.iloc[i]
            future = price.iloc[i + 1 : i + 1 + self.horizon]
            if ((future - current_peak) / current_peak <= self.threshold).any():
                labels.iloc[i] = True

        self.labels = labels
        self.logger.info("Computed drawdown labels")

    def align_data(self) -> None:
        """Align C and labels on common dates."""
        df = pd.DataFrame({"C": self.C, "label": self.labels.astype(int)})
        df = df.dropna()
        self.df = df
        self.logger.info(f"Aligned data: {len(df)} observations")

    def run_granger(self) -> Dict[int, float]:
        """Run Granger causality tests and write p-values by lag."""
        data = self.df[["label", "C"]]
        pvals = {}
        for lag in range(1, self.max_lag + 1):
            res = grangercausalitytests(data, maxlag=lag, verbose=False)
            pval = res[lag][0]["ssr_ftest"][1]
            pvals[lag] = pval
            self.logger.info(f"Granger lag={lag} p-value={pval:.4f}")
        pd.Series(pvals).to_csv(self.output_dir / "granger_pvalues.csv", header=["pvalue"])
        return pvals

    def roc_analysis(self) -> float:
        """Compute ROC AUC and save ROC plot."""
        y = self.df["label"]
        scores = self.df["C"]
        fpr, tpr, _ = roc_curve(y, scores)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC: C(t) → Drawdown Event")
        plt.legend(loc="lower right")
        plt.savefig(self.output_dir / "roc_curve.png", dpi=300)
        plt.close()
        self.logger.info(f"Saved ROC curve (AUC={roc_auc:.3f})")
        return roc_auc

    def logistic_regression(self) -> Dict[str, float]:
        """Fit logistic regression; save p-value & CV AUC."""
        X = sm.add_constant(self.df["C"])
        logit = sm.Logit(self.df["label"], X).fit(disp=False)
        pvalue = logit.pvalues["C"]

        model = LogisticRegression()
        scores = cross_val_score(model, self.df[["C"]], self.df["label"],
                                 cv=5, scoring="roc_auc")
        cv_auc = float(scores.mean())

        pd.DataFrame({
            "coef_pvalue": [pvalue],
            "cv_auc":      [cv_auc]
        }).to_csv(self.output_dir / "logit_summary.csv", index=False)

        self.logger.info(f"Logistic regression p-value={pvalue:.4f}, CV AUC={cv_auc:.3f}")
        return {"pvalue": pvalue, "cv_auc": cv_auc}

    def run(self) -> None:
        """Execute the full validation pipeline."""
        self.load_series()
        self.compute_drawdown_labels()
        self.align_data()
        self.run_granger()
        self.roc_analysis()
        self.logistic_regression()
        self.logger.info("Validation complete. See reports in %s", self.output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate C(t) vs drawdown hypothesis")
    parser.add_argument("--prices",    default="data/processed/prices.csv",
                        help="Processed prices CSV")
    parser.add_argument("--c_series",  default="data/processed/C_series.csv",
                        help="Systemic-entanglement series CSV")
    parser.add_argument("--output_dir",default="reports/validation",
                        help="Directory for validation outputs")
    parser.add_argument("--horizon",   type=int,   default=5,
                        help="Lookahead horizon for drawdown (days)")
    parser.add_argument("--threshold", type=float, default=-0.05,
                        help="Drawdown threshold (e.g., -0.05 for 5%)")
    parser.add_argument("--max_lag",   type=int,   default=5,
                        help="Max lag for Granger-causality")

    args = parser.parse_args()

    validator = CSeriesValidator(
        price_csv=args.prices,
        c_csv=args.c_series,
        output_dir=args.output_dir,
        drawdown_horizon=args.horizon,
        drawdown_threshold=args.threshold,
        max_granger_lag=args.max_lag
    )
    validator.run()