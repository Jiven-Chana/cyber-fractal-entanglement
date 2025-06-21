#!/usr/bin/env python3
"""
animate_qmi.py

Generate a 3D surface animation of quantum mutual information (QMI)
evolution across asset pairs over time.

Usage:
    python src/animate_qmi.py \
      --input_csv data/processed/qmi.csv \
      --output_mp4 reports/figures/qmi_evolution.mp4 \
      --fps 10 --dpi 150
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s"
    )


class QMIAnimator:
    """
    Loads a QMI time-series CSV and produces a 3D surface animation showing
    the evolution of QMI values over time across asset pairs.
    """
    def __init__(
        self,
        input_csv: Path,
        output_mp4: Path,
        fps: int = 10,
        dpi: int = 150,
    ):
        self.input_csv = Path(input_csv)
        self.output_mp4 = Path(output_mp4)
        self.fps = fps
        self.dpi = dpi
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_qmi(self) -> None:
        if not self.input_csv.exists():
            self.logger.error(f"QMI CSV not found: {self.input_csv}")
            raise FileNotFoundError(f"{self.input_csv} not found")
        df = pd.read_csv(self.input_csv, index_col=0, parse_dates=True)
        self.times = df.index.to_pydatetime()
        self.pair_labels = df.columns.tolist()
        self.Z = df.values  # shape (T, P)
        self.T, self.P = self.Z.shape
        self.logger.info(f"Loaded QMI matrix: {self.T} time points × {self.P} pairs")

    def animate(self) -> None:
        # Prepare grids
        t_vals = np.arange(self.T)
        p_vals = np.arange(self.P)
        T_grid, P_grid = np.meshgrid(t_vals, p_vals, indexing='ij')

        # Create figure
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('Time Index')
        ax.set_ylabel('Pair Index')
        ax.set_zlabel('QMI')

        # Animation update
        def update(frame: int):
            ax.clear()
            ax.set_title(f"3D QMI Evolution: {self.times[frame].date()}")
            ax.set_xlabel('Time Index')
            ax.set_ylabel('Pair Index')
            ax.set_zlabel('QMI')
            Z_frame = np.zeros_like(self.Z)
            Z_frame[:frame + 1, :] = self.Z[:frame + 1, :]
            surf = ax.plot_surface(
                T_grid, P_grid, Z_frame,
                rstride=1, cstride=1,
                cmap='viridis',
                edgecolor='none',
                antialiased=True
            )
            return surf,

        anim = animation.FuncAnimation(
            fig, update, frames=self.T,
            interval=1000 / self.fps, blit=False
        )

        # Ensure output directory exists
        self.output_mp4.parent.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Saving animation to {self.output_mp4}")
        anim.save(
            str(self.output_mp4),
            writer='ffmpeg',
            fps=self.fps,
            dpi=self.dpi
        )
        plt.close(fig)
        self.logger.info("Animation saved successfully.")

    def run(self) -> None:
        self.load_qmi()
        self.animate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Animate QMI evolution in 3D.")
    parser.add_argument(
        '--input_csv',
        default='data/processed/qmi.csv',
        help='Path to QMI CSV'
    )
    parser.add_argument(
        '--output_mp4',
        default='reports/figures/qmi_evolution.mp4',
        help='Output MP4 path'
    )
    parser.add_argument(
        '--fps', type=int, default=10,
        help='Frames per second'
    )
    parser.add_argument(
        '--dpi', type=int, default=150,
        help='Figure DPI'
    )
    args = parser.parse_args()
    setup_logging()
    animator = QMIAnimator(
        input_csv=Path(args.input_csv),
        output_mp4=Path(args.output_mp4),
        fps=args.fps,
        dpi=args.dpi
    )
    animator.run()