# src/density.py

"""
DensityBuilder: build rolling-window covariance (Σ_t) and
trace-normalized density matrices (ρ_t) from fractal-dimension series.

Window parameter counts rows in the fractal-dimension matrix (not calendar days).
"""

import logging
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd


class DensityBuilder:
    """
    Build and validate density matrices from fractal dimensions.

    Attributes:
        fractal_path (Path): Path to data/processed/fractals.csv
        window (int): Number of rows in each covariance window
        regularisation_eps (float): small diagonal added to Σ for PSD guarantee
        output_dir (Path): directory to save .npz files per date
    """

    def __init__(
        self,
        fractal_path: Union[str, Path] = "data/processed/fractals.csv",
        window: int = 100,
        regularization_eps: float = 1e-6,
        output_dir: Union[str, Path] = "data/processed/density/"
    ) -> None:
        self.fractal_path = Path(fractal_path)
        self.window = window
        self.regularization_eps = regularization_eps
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Logger setup
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        )
        if not self.logger.handlers:
            self.logger.addHandler(handler)

        self.df_fd: pd.DataFrame = None

    def load_fractals(self) -> None:
        """
        Load fractal-dimension CSV into DataFrame with Date index.
        Raises FileNotFoundError if missing.
        """
        if not self.fractal_path.exists():
            raise FileNotFoundError(f"{self.fractal_path} not found")
        self.df_fd = (
            pd.read_csv(self.fractal_path, index_col=0, parse_dates=True)
              .sort_index()
        )
        self.logger.info(
            f"Loaded FD: {self.df_fd.shape[0]} rows × {self.df_fd.shape[1]} assets"
        )

    def compute_density_series(self) -> None:
        """
        For each index t >= window:
          1. Extract last `window` rows of FD matrix.
          2. Compute Σ_t = cov(block) + regularization_eps * I.
          3. Validate Σ_t PSD via eigenvalues.
          4. Compute ρ_t = Σ_t / Tr(Σ_t) and validate Tr(ρ_t)=1.
          5. Save Σ_t, ρ_t, and asset list as .npz(date).
        """
        if self.df_fd is None:
            self.load_fractals()

        assets = list(self.df_fd.columns)
        N = len(assets)

        for idx in range(self.window, len(self.df_fd) + 1):
            block = self.df_fd.iloc[idx - self.window : idx].values  # shape (window, N)
            Sigma = np.cov(block, rowvar=False)                      # (N×N)
            # regularize for PSD
            Sigma += np.eye(N) * self.regularization_eps

            # PSD validation
            eigvals = np.linalg.eigvalsh(Sigma)
            if np.any(eigvals < -1e-8):
                self.logger.warning(
                    f"Non-PSD Σ at {self.df_fd.index[idx-1]}: min eigenvalue {eigvals.min():.2e}"
                )

            # trace-normalise
            trace_val = np.trace(Sigma)
            if trace_val <= 0:
                self.logger.error(
                    f"Non-positive trace at {self.df_fd.index[idx-1]}: {trace_val:.2e}"
                )
            rho = Sigma / trace_val

            # trace(ρ) validation
            tr_rho = np.trace(rho)
            if not np.isclose(tr_rho, 1.0, atol=1e-8):
                self.logger.error(
                    f"Trace(ρ)≠1 at {self.df_fd.index[idx-1]}: {tr_rho:.6f}"
                )

            date_str = self.df_fd.index[idx - 1].strftime("%Y-%m-%d")
            out_path = self.output_dir / f"{date_str}.npz"
            np.savez(out_path, Sigma=Sigma, rho=rho, assets=assets)
            self.logger.debug(f"Saved density for {date_str}")

        self.logger.info(f"All densities saved to {self.output_dir}")

    def run(self) -> None:
        """
        Execute full pipeline: load fractals and compute densities.
        """
        self.load_fractals()
        self.compute_density_series()


if __name__ == "__main__":

    """
    Run: 
    python src/density.py \
  --input  data/processed/fractals.csv \
  --window 100 \
  --eps    1e-6 \
  --output data/processed/density/
    """

    import argparse

    parser = argparse.ArgumentParser(description="Build Σ → ρ from fractals")
    parser.add_argument(
        "-i", "--input", default="data/processed/fractals.csv",
        help="Path to fractals.csv"
    )
    parser.add_argument(
        "-w", "--window", type=int, default=100,
        help="Number of rows per covariance window"
    )
    parser.add_argument(
        "-e", "--eps", type=float, default=1e-6,
        help="Regularization epsilon added to Σ diagonal"
    )
    parser.add_argument(
        "-o", "--output", default="data/processed/density/",
        help="Directory to save density .npz files"
    )
    args = parser.parse_args()

    db = DensityBuilder(
        fractal_path=args.input,
        window=args.window,
        regularization_eps=args.eps,
        output_dir=args.output
    )
    db.run()