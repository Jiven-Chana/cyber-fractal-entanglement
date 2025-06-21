# src/validate_density.py

"""
ValidateDensity: sanity-check .npz density matrix files.

For each .npz in the specified directory, verifies:
  1. Keys: 'Sigma', 'rho', 'assets' present.
  2. Shapes: Sigma and rho are (N,N); assets list length = N.
  3. No NaN or infinite values in Sigma or rho.
  4. Symmetry: Sigma ≈ Sigma.T, rho ≈ rho.T.
  5. Positive semidefiniteness of Sigma (all eigenvalues ≥ -tol_psd).
  6. Trace normalization: trace(rho) ≈ 1 (±tol_trace).

Usage:
    python src/validate_density.py data/processed/density
"""

import sys
from pathlib import Path
import numpy as np


def validate_file(path: Path, tol_psd: float = 1e-8, tol_trace: float = 1e-8) -> bool:
    data = np.load(path, allow_pickle=True)
    required = {'Sigma', 'rho', 'assets'}
    missing = required - set(data.keys())
    if missing:
        print(f"{path.name}: missing keys {missing}")
        return False

    Sigma = data['Sigma']
    rho   = data['rho']
    assets = data['assets']
    N = len(assets)

    ok = True

    # shape checks
    if Sigma.shape != (N, N):
        print(f"{path.name}: Sigma shape {Sigma.shape} != ({N},{N})")
        ok = False
    if rho.shape != (N, N):
        print(f"{path.name}: rho shape {rho.shape} != ({N},{N})")
        ok = False

    # finiteness checks
    if not np.isfinite(Sigma).all():
        print(f"{path.name}: Sigma contains NaN or Inf")
        ok = False
    if not np.isfinite(rho).all():
        print(f"{path.name}: rho contains NaN or Inf")
        ok = False

    # symmetry checks
    if not np.allclose(Sigma, Sigma.T, atol=1e-8):
        print(f"{path.name}: Sigma not symmetric")
        ok = False
    if not np.allclose(rho, rho.T, atol=1e-8):
        print(f"{path.name}: rho not symmetric")
        ok = False

    # PSD check
    eigs = np.linalg.eigvalsh(Sigma)
    if np.any(eigs < -tol_psd):
        print(f"{path.name}: Sigma min eigenvalue {eigs.min():.2e} < -{tol_psd:.1e}")
        ok = False

    # trace check
    tr = np.trace(rho)
    if not np.isclose(tr, 1.0, atol=tol_trace):
        print(f"{path.name}: trace(rho) = {tr:.6f} != 1")
        ok = False

    return ok


def main(directory: str = "data/processed/density") -> None:
    folder = Path(directory)
    if not folder.exists():
        print(f"Directory not found: {directory}")
        sys.exit(1)

    files = sorted(folder.glob("*.npz"))
    if not files:
        print(f"No .npz files found in {directory}")
        sys.exit(1)

    failures = []
    for f in files:
        if not validate_file(f):
            failures.append(f.name)

    total = len(files)
    passed = total - len(failures)
    print(f"Validation complete: {passed}/{total} files passed.")
    if failures:
        print("Failed files:")
        for name in failures:
            print("  -", name)
        sys.exit(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate density .npz files")
    parser.add_argument(
        "directory",
        nargs="?",
        default="data/processed/density",
        help="Directory containing density .npz files"
    )
    args = parser.parse_args()
    main(args.directory)