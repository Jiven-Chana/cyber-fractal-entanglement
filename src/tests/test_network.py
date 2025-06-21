# tests/test_network.py

import pandas as pd
import numpy as np
import pytest
from pathlib import Path

from src.network import NetworkBuilder

@pytest.fixture
def synthetic_qmi(tmp_path):
    """
    Create a synthetic QMI CSV for 3 assets A, B, C with uniform edge weights.
    """
    df = pd.DataFrame({
        "QMI_A_B": [1.0, 2.0],
        "QMI_A_C": [1.0, 2.0],
        "QMI_B_C": [1.0, 2.0],
    }, index=pd.to_datetime(["2025-01-01", "2025-01-02"]))
    csv_path = tmp_path / "qmi.csv"
    df.to_csv(csv_path)
    return csv_path

def test_network_on_synthetic(synthetic_qmi, tmp_path):
    """
    For a triangle graph with equal weights:
    NetworkX normalizes the centrality vector to L2‐norm = 1,
    so sum of centralities across 3 nodes = sqrt(3).
    """
    output_csv = tmp_path / "C_series.csv"
    nb = NetworkBuilder(qmi_csv=synthetic_qmi, output_csv=output_csv)
    nb.run()

    df_C = pd.read_csv(output_csv, index_col=0, parse_dates=True)

    # Check shape & missing
    assert df_C.shape == (2, 1)
    assert not df_C["C"].isna().any()

    # Expect sum = sqrt(3)
    expected = np.sqrt(3)
    assert pytest.approx(expected, rel=1e-6) == df_C["C"].iloc[0]
    assert pytest.approx(expected, rel=1e-6) == df_C["C"].iloc[1]

def test_real_C_series_properties():
    """
    Validate your real C_series.csv:
      - Row count matches qmi.csv
      - No NaNs or negatives
      - No duplicate dates
      - Business-day frequency
      - Positive mean
    """
    root = Path(__file__).resolve().parents[2]
    qmi_path = root / "data/processed/qmi.csv"
    c_path   = root / "data/processed/C_series.csv"

    df_qmi = pd.read_csv(qmi_path, index_col=0, parse_dates=True)
    df_C   = pd.read_csv(c_path, index_col=0, parse_dates=True)

    # 1) Same number of rows
    assert len(df_C) == len(df_qmi)

    # 2) No missing or negative values
    assert not df_C["C"].isna().any()
    assert (df_C["C"] >= 0).all()

    # 3) No duplicate dates
    assert df_C.index.duplicated().sum() == 0

    # 4) Business-day frequency
    assert df_C.index.inferred_freq == "B"

    # 5) Mean > 0
    assert df_C["C"].mean() > 0