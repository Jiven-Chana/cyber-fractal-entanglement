# src/pre_fractal.py

import logging
from pathlib import Path
from typing import Union, Dict, Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller


class PreFractalChecker:
    """
    Pre‐Fractal Sanity Checker

    Performs the following checks on a price time series matrix:
      1. Minimum length
      2. Non‐zero variance per asset
      3. Extreme return outliers (z‐score)
      4. (Optional) Stationarity via ADF test

    Usage:
        checker = PreFractalChecker("data/processed/prices.csv", window=30)
        report_df = checker.run_all()
        checker.save_report("reports/pre_fractal_report.csv")
    """

    def __init__(self,
                 filepath: Union[str, Path],
                 window: int = 30,
                 outlier_z_threshold: float = 10.0,
                 adf_p_threshold: float = 0.10):
        """
        :param filepath: path to prices CSV (Date indexed, assets in columns)
        :param window: minimum required number of observations
        :param outlier_z_threshold: z‐score above which returns are flagged
        :param adf_p_threshold: p‐value cutoff for unit‐root rejection
        """
        # Logging setup
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        self.logger.addHandler(ch)

        # Resolve paths
        self.input_path = Path(filepath)
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")
        self.df = pd.read_csv(self.input_path,
                              index_col=0,
                              parse_dates=True).sort_index()

        # Parameters
        self.window = window
        self.zthr = outlier_z_threshold
        self.adf_pthr = adf_p_threshold

        self.logger.info(f"Loaded {len(self.df)} rows × {self.df.shape[1]} assets "
                         f"from {self.input_path.name}")

    def check_length(self) -> Dict[str, Any]:
        """Ensure DataFrame has at least `window` rows."""
        n = len(self.df)
        ok = n >= self.window
        msg = f"{n} rows {'OK' if ok else 'LESS THAN'} required {self.window}"
        self.logger.info("Length check: " + msg)
        return {"check": "min_length", "asset": "ALL",
                "result": ok, "value": n}

    def check_variance(self) -> Dict[str, Any]:
        """Check no asset has zero variance over the sample."""
        variances = self.df.var()
        rows = []
        for asset, v in variances.items():
            ok = v > 0
            rows.append({"check": "zero_variance", "asset": asset,
                         "result": ok, "value": v})
            status = "OK" if ok else "ZERO VAR"
            self.logger.info(f"Variance check {asset}: {v:.6f} ({status})")
        return rows

    def check_outliers(self) -> Dict[str, Any]:
        """
        Flag extreme returns beyond ±zthr.
        Returns count of outliers per asset.
        """
        rets = self.df.pct_change().dropna()
        z = np.abs(stats.zscore(rets, nan_policy="omit"))
        rows = []
        for i, asset in enumerate(rets.columns):
            count = int((z[:, i] > self.zthr).sum())
            ok = count == 0
            rows.append({"check": "extreme_outliers", "asset": asset,
                         "result": ok, "value": count})
            self.logger.info(f"Outlier check {asset}: {count} >±{self.zthr}")
        return rows

    def check_stationarity(self) -> Dict[str, Any]:
        """
        Run Augmented Dickey‐Fuller test per asset.
        Warn if p‐value > adf_p_threshold (unit-root not rejected).
        """
        rows = []
        for asset in self.df.columns:
            series = self.df[asset].dropna()
            try:
                pvalue = adfuller(series, autolag="AIC")[1]
            except Exception as e:
                pvalue = np.nan
                self.logger.warning(f"ADF error on {asset}: {e}")
            ok = pvalue < self.adf_pthr if not np.isnan(pvalue) else False
            rows.append({"check": "adf_stationarity", "asset": asset,
                         "result": ok, "value": pvalue})
            self.logger.info(f"ADF check {asset}: p={pvalue:.4f} "
                             f"({'stationary' if ok else 'non‐stationary'})")
        return rows

    def run_all(self) -> pd.DataFrame:
        """
        Execute all pre-fractal checks and return a DataFrame report.
        Columns: [check, asset, result, value]
        """
        report = []
        # Global length check
        report.append(self.check_length())
        # Per‐asset checks
        report.extend(self.check_variance())
        report.extend(self.check_outliers())
        report.extend(self.check_stationarity())

        return pd.DataFrame(report)

    def save_report(self, report_path: Union[str, Path]) -> None:
        """
        Save the report DataFrame to CSV.
        """
        rp = Path(report_path)
        rp.parent.mkdir(parents=True, exist_ok=True)
        df = self.run_all()
        df.to_csv(rp, index=False)
        self.logger.info(f"Pre-fractal report saved to {rp}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Pre-Fractal Data Sanity Checks")
    parser.add_argument("--input", "-i",
                        default="data/processed/prices.csv",
                        help="Cleaned prices CSV")
    parser.add_argument("--output", "-o",
                        default="reports/pre_fractal_report.csv",
                        help="Destination for report CSV")
    parser.add_argument("--window", "-w", type=int, default=30,
                        help="Minimum required data length")
    parser.add_argument("--zthreshold", "-z", type=float, default=10.0,
                        help="Z-score threshold for outliers")
    parser.add_argument("--adf_p", "-p", type=float, default=0.10,
                        help="P-value threshold for ADF stationarity")
    args = parser.parse_args()

    checker = PreFractalChecker(
        filepath=args.input,
        window=args.window,
        outlier_z_threshold=args.zthreshold,
        adf_p_threshold=args.adf_p
    )
    report_df = checker.run_all()
    checker.save_report(args.output)