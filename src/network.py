# src/network.py

"""
NetworkBuilder: aggregate pairwise QMI edge-weights into
a daily systemic-entanglement score via eigenvector centrality.
"""

import logging
from pathlib import Path
from typing import Union

import networkx as nx
import pandas as pd


class NetworkBuilder:
    """
    Attributes:
        qmi_csv: Path to data/processed/qmi.csv
        output_csv: Path to save systemic-entanglement series C(t)
    """

    def __init__(
        self,
        qmi_csv: Union[str, Path] = "data/processed/qmi.csv",
        output_csv: Union[str, Path] = "data/processed/C_series.csv"
    ) -> None:
        self.qmi_csv = Path(qmi_csv)
        self.output_csv = Path(output_csv)

        # Logger
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
        1. Read QMI CSV into DataFrame (index=Date).
        2. For each date:
            a. Create weighted Graph with edges from QMI columns.
            b. Compute eigenvector centrality.
            c. Record the leading eigenvector score as C(t).
        3. Save C(t) series to output CSV.
        """
        df_qmi = pd.read_csv(self.qmi_csv, index_col=0, parse_dates=True)
        dates = []
        scores = []

        # Extract asset list from column names
        # e.g. columns like 'QMI_SPX_GOLD'
        asset_pairs = [col.replace("QMI_", "").split("_") for col in df_qmi.columns]
        assets = sorted({a for pair in asset_pairs for a in pair})

        for date, row in df_qmi.iterrows():
            G = nx.Graph()
            G.add_nodes_from(assets)

            # Add weighted edges
            for col, weight in row.items():
                a, b = col.replace("QMI_", "").split("_")
                G.add_edge(a, b, weight=float(weight))

            # Compute eigenvector centrality (uses weights)
            try:
                centrality = nx.eigenvector_centrality_numpy(G, weight="weight")
                # Systemic-entanglement score is sum of centralities, or the principal value
                C_t = sum(centrality.values())
            except Exception as e:
                self.logger.error(f"Centrality failure on {date.date()}: {e}")
                C_t = float("nan")

            dates.append(date)
            scores.append(C_t)

        df_C = pd.DataFrame({"C": scores}, index=dates).sort_index()
        df_C.to_csv(self.output_csv)
        self.logger.info(f"Saved systemic-entanglement C(t) to {self.output_csv}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build systemic-entanglement score C(t)")
    parser.add_argument(
        "-i", "--input",
        default="data/processed/qmi.csv",
        help="Path to QMI CSV"
    )
    parser.add_argument(
        "-o", "--output",
        default="data/processed/C_series.csv",
        help="Path to save C(t) series"
    )
    args = parser.parse_args()

    nb = NetworkBuilder(qmi_csv=args.input, output_csv=args.output)
    nb.run()