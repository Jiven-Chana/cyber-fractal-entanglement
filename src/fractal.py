# src/fractal.py

"""
FractalExtractor: compute rolling‐window fractal dimensions (Higuchi or Hurst)
for multiple assets from a cleaned prices CSV.

Author: Jiven Chana
Date: 2025-06-21
"""

import logging
from pathlib import Path
from typing import Union
import numpy as np
import pandas as pd

# Try nolds.higuchi_fd, otherwise fallback to antropy.fractal.higuchi_fd
try:
    from nolds import higuchi_fd
except (ImportError, AttributeError):
    from antropy.fractal import higuchi_fd

class FractalExtractor:
    """
    Computes rolling‐window fractal dimensions (Higuchi or Hurst)
    from a cleaned prices CSV.

    Attributes:
        input_path: Path to cleaned prices CSV (Date index, asset columns).
        window: Window length (in rows) for rolling FD.
        method: 'higuchi' or 'hurst'.
        output_path: Path to save the resulting fractal-dimension CSV.
        df_prices: Loaded price DataFrame.
        df_fd: Computed fractal-dimension DataFrame.
    """

    def __init__(self,
                 input_path: Union[str, Path] = "data/processed/prices.csv",
                 window: int = 30,
                 method: str = "higuchi",
                 output_path: Union[str, Path] = "data/processed/fractals.csv"):
        # Resolve and store paths
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.window = window
        self.method = method.lower()
        # Prepare logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        if not self.logger.handlers:
            self.logger.addHandler(handler)
        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        # Placeholders
        self.df_prices: pd.DataFrame = None
        self.df_fd: pd.DataFrame = None

    def load_data(self) -> None:
        """
        Load cleaned prices CSV into DataFrame with Date index.
        Raises FileNotFoundError if the input file is missing.
        """
        if not self.input_path.exists():
            self.logger.error(f"Input file not found: {self.input_path}")
            raise FileNotFoundError(f"{self.input_path} does not exist")
        df = pd.read_csv(self.input_path, index_col=0, parse_dates=True)
        self.df_prices = df.sort_index()
        self.logger.info(f"Loaded prices: {df.shape[0]} rows × {df.shape[1]} assets")

    def _higuchi_fd(self, arr: np.ndarray) -> float:
        """
        Compute Higuchi fractal dimension for a 1D numpy array.
        Uses nolds.higuchi_fd if available, otherwise antropy.fractal.higuchi_fd.
        """

        data = np.asarray(arr, dtype=float)
        return higuchi_fd(data)

    # def _higuchi_fd(self, arr: np.ndarray) -> float:
    #     """
    #     Compute Higuchi fractal dimension for a 1D numpy array.
    #     """
    #     return nolds.higuchi_fd(arr)

    def _hurst_exp(self, arr: np.ndarray) -> float:
        """
        Compute Hurst exponent via rescaled range as a fallback.
        """
        series = pd.Series(arr)
        lags = range(2, min(len(series)//2, 20))
        tau = [ (series.diff(lag).std()) for lag in lags ]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        hurst = max(poly[0] * 2.0, 0.0)
        return hurst

    def compute_fd(self) -> None:
        """
        Compute rolling-window fractal dimension for each asset.
        Stores result in self.df_fd, aligned to full-window indices.
        """
        if self.df_prices is None:
            self.load_data()

        results = {}
        for asset in self.df_prices.columns:
            self.logger.info(f"Computing {self.method} FD for {asset}, window={self.window}")
            series = self.df_prices[asset]
            if self.method == "higuchi":
                fd_series = series.rolling(self.window) \
                                  .apply(self._higuchi_fd, raw=True)
            else:
                fd_series = series.rolling(self.window) \
                                  .apply(self._hurst_exp, raw=True)
            results[asset] = fd_series

        df_fd = pd.DataFrame(results).dropna()
        self.df_fd = df_fd
        self.logger.info(f"Computed fractal-dimension matrix: {df_fd.shape[0]} rows × {df_fd.shape[1]} assets")

    def save(self) -> None:
        """
        Save the computed fractal-dimension DataFrame to CSV.
        """
        if self.df_fd is None:
            self.compute_fd()
        self.df_fd.to_csv(self.output_path)
        self.logger.info(f"Saved fractal dimensions to {self.output_path}")

    def run(self) -> None:
        """
        Execute full pipeline: load data, compute FD, and save results.
        """
        self.load_data()
        self.compute_fd()
        self.save()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute rolling-window fractal dimensions")
    parser.add_argument("-i", "--input",
                        default="data/processed/prices.csv",
                        help="Cleaned prices CSV")
    parser.add_argument("-w", "--window", type=int, default=30,
                        help="Window length for fractal dimension")
    parser.add_argument("-m", "--method", choices=["higuchi", "hurst"],
                        default="higuchi", help="Fractal method")
    parser.add_argument("-o", "--output",
                        default="data/processed/fractals.csv",
                        help="Where to save fractal dimensions")
    args = parser.parse_args()

    fe = FractalExtractor(input_path=args.input,
                          window=args.window,
                          method=args.method,
                          output_path=args.output)
    fe.run()