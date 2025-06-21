# tests/test_qmi.py

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.qmi import QMIComputer, von_neumann_entropy

@pytest.fixture  # default function scope
def synthetic_density(tmp_path):
    """
    Creates two .npz files with trivial density matrices (2 assets):
      - Uniform density (I/2 → maximum entropy)
      - Off-diagonal coupling example
    """
    d = tmp_path / "density"
    d.mkdir()
    assets = ["A", "B"]

    # 1) Uniform density: rho = I/2
    rho1 = np.eye(2) * 0.5
    np.savez(d / "2025-01-01.npz", rho=rho1, assets=assets)

    # 2) Coupled density: off-diagonal terms
    rho2 = np.array([[0.6, 0.4],
                     [0.4, 0.4]])
    rho2 /= np.trace(rho2)
    np.savez(d / "2025-01-02.npz", rho=rho2, assets=assets)

    return d

def test_von_neumann_entropy():
    # 1x1 density [1.0] → entropy 0
    assert von_neumann_entropy(np.array([[1.0]])) == pytest.approx(0.0)

    # 2x2 uniform diag [0.5,0.5] → -2*0.5*log(0.5) = ln(2)
    expected = -2 * 0.5 * np.log(0.5)
    rho = np.diag([0.5, 0.5])
    assert von_neumann_entropy(rho) == pytest.approx(expected)

def test_qmi_non_negative_and_shape(synthetic_density, tmp_path):
    """
    Run QMIComputer on the synthetic densities and verify:
      - Output CSV has exactly 2 rows and 1 column 'QMI_A_B'.
      - All QMI values are >= 0.
    """
    out_csv = tmp_path / "qmi.csv"
    qmi = QMIComputer(density_dir=synthetic_density, output_csv=out_csv)
    qmi.run()

    df = pd.read_csv(out_csv, index_col=0, parse_dates=True)

    # Should produce two dates
    assert df.shape[0] == 2

    # Should have exactly one column
    assert list(df.columns) == ["QMI_A_B"]

    # All values non-negative
    assert (df["QMI_A_B"] >= 0).all()

@pytest.mark.skipif(
    not Path("data/processed/qmi.csv").exists(),
    reason="data/processed/qmi.csv not found, skipping real-data tests"
)
def test_real_qmi_properties():
    """
    Load your actual QMI CSV and assert:
      - No missing values
      - Non-negative everywhere
      - Number of columns matches N choose 2 for N assets
    """
    df = pd.read_csv("data/processed/qmi.csv", index_col=0, parse_dates=True)

    # 1) No missing
    assert not df.isna().any().any()

    # 2) Non-negative
    assert (df.values >= 0).all()

    # 3) Columns count matches N(N-1)/2 for some integer N
    M = df.shape[1]
    N = (1 + np.sqrt(1 + 8*M)) / 2
    assert N.is_integer(), f"Expected M={M} to satisfy N(N-1)/2, got N={N}"