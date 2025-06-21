# src/qmi.py

"""
QMIComputer: compute von Neumann entropy and pairwise
quantum mutual information (QMI) time series from density matrices.

Usage:
    python src/qmi.py \
      --density_dir data/processed/density/ \
      --output_csv data/processed/qmi.csv
"""

import logging
from pathlib import Path
from typing import Union, List, Dict

import numpy as np
import pandas as pd


def von_neumann_entropy(rho: np.ndarray) -> float:
    """
    Compute the von Neumann entropy S(ρ) = -Tr(ρ log ρ).
    Only positive eigenvalues contribute.
    """
    vals = np.linalg.eigvalsh(rho)
    vals = vals[vals > 0]
    return -np.sum(vals * np.log(vals))


class QMIComputer:
    """
    Compute pairwise quantum mutual information (QMI) time series.

    Attributes:
        density_dir: Path to folder of .npz density files (ρ matrices).
        output_csv: Path to save resulting QMI DataFrame.
        logger: logging.Logger instance.
    """

    def __init__(
        self,
        density_dir: Union[str, Path] = "data/processed/density/",
        output_csv: Union[str, Path] = "data/processed/qmi.csv"
    ) -> None:
        self.density_dir = Path(density_dir)
        self.output_csv = Path(output_csv)

        # Logger setup
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        )
        if not self.logger.handlers:
            self.logger.addHandler(handler)

    def run(self) -> None:
        """
        End-to-end:
          1. Load each .npz (Σ, ρ, assets) from density_dir.
          2. Compute S_i = S([ρ_ii]) for each asset.
          3. Compute pairwise QMI I_Q^{i,j} = S_i + S_j - S_{ij}.
          4. Save a CSV with Date index and one column per asset-pair.
        """
        files = sorted(self.density_dir.glob("*.npz"))
        if not files:
            raise FileNotFoundError(f"No density files (*.npz) in {self.density_dir}")

        dates: List[pd.Timestamp] = []
        records: List[Dict[str, float]] = []

        for f in files:
            data = np.load(f, allow_pickle=True)
            rho = data["rho"]
            assets = list(data["assets"])
            N = len(assets)

            # Compute single-asset entropies
            S: List[float] = []
            for i in range(N):
                marginal = np.array([[rho[i, i]]])
                S.append(von_neumann_entropy(marginal))

            # Compute pairwise QMI
            rec: Dict[str, float] = {}
            for i in range(N):
                for j in range(i + 1, N):
                    sub_rho = rho[np.ix_([i, j], [i, j])]
                    Sij = von_neumann_entropy(sub_rho)
                    Iq = S[i] + S[j] - Sij
                    col_name = f"QMI_{assets[i]}_{assets[j]}"
                    # Clip negative artifacts
                    rec[col_name] = max(Iq, 0.0)
                    if Iq < 0:
                        self.logger.warning(
                            f"Negative QMI {col_name}={Iq:.2e} on {f.stem}, clipped to 0"
                        )

            dates.append(pd.to_datetime(f.stem))
            records.append(rec)
            self.logger.debug(f"Computed QMI for {f.stem}")

        # Construct DataFrame
        df_qmi = pd.DataFrame(records, index=dates).sort_index()
        df_qmi.to_csv(self.output_csv)
        self.logger.info(f"Saved QMI series to {self.output_csv}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute QMI time series from density matrices")
    parser.add_argument(
        "-d", "--density_dir",
        default="data/processed/density/",
        help="Directory containing density .npz files"
    )
    parser.add_argument(
        "-o", "--output_csv",
        default="data/processed/qmi.csv",
        help="Path to save QMI CSV"
    )
    args = parser.parse_args()

    qmi = QMIComputer(
        density_dir=args.density_dir,
        output_csv=args.output_csv
    )
    qmi.run()