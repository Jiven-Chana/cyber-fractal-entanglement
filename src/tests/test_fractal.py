# tests/test_fractal.py

import numpy as np
import pytest

from src.fractal import FractalExtractor

@pytest.fixture(scope="module")
def fe_higuchi():
    """FractalExtractor configured to use Higuchi method."""
    return FractalExtractor(method="higuchi")

@pytest.fixture(scope="module")
def fe_hurst():
    """FractalExtractor configured to use Hurst method."""
    return FractalExtractor(method="hurst")

def test_higuchi_on_sinusoid(fe_higuchi):
    """
    A pure sinusoid is a smooth 1D curve → fractal dimension ≈ 1.0.
    We allow a small tolerance around 1.0 due to numerical error.
    """
    t = np.linspace(0, 2 * np.pi, 1000)
    ts = np.sin(t)
    fd = fe_higuchi._higuchi_fd(ts)
    assert fd == pytest.approx(1.0, rel=0.05), \
        f"Expected FD≈1.0 for sinusoid, got {fd}"

def test_higuchi_on_constant(fe_higuchi):
    """
    A constant series should have zero 'roughness' → FD = 0.
    """
    ts = np.ones(500)
    fd = fe_higuchi._higuchi_fd(ts)
    assert fd == pytest.approx(0.0, abs=1e-6), \
        f"Expected FD=0 for constant series, got {fd}"

def test_hurst_on_sinusoid(fe_hurst):
    """
    Sinusoid for Hurst exponent fallback should also ≈ 1 (no persistence).
    """
    t = np.linspace(0, 2 * np.pi, 1000)
    ts = np.sin(t)
    fd = fe_hurst._hurst_exp(ts)
    assert fd == pytest.approx(1.0, rel=0.1), \
        f"Expected Hurst ≈1.0 for sinusoid, got {fd}"

def test_hurst_on_constant(fe_hurst):
    """
    Constant series for Hurst fallback should return 0.
    """
    ts = np.ones(500)
    fd = fe_hurst._hurst_exp(ts)
    assert fd == pytest.approx(0.0, abs=1e-6), \
        f"Expected Hurst=0 for constant series, got {fd}"