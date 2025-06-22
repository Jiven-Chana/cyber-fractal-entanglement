Below is a fully fleshed-out, polished README template you can drop into the root of your repo (README.md). It dives into each module, shows how to run everything, explains key findings/insights, and presents next steps clearly—making it compelling both as documentation and as a project narrative.

⸻


# Cyber-Fractal Entanglement 🧠⚛️📉

**Detecting Systemic Coupling and Market Drawdowns Using Fractal Geometry & Quantum Mutual Information**

---

## 🚀 Overview

This repository implements a **proof-of-concept pipeline** to detect early-warning indicators of market drawdowns by merging:
- **Fractal geometry** (Higuchi fractal dimension)
- **Quantum information theory** (density matrices & quantum mutual information)
- **Dynamic network analysis** (eigenvector centrality)
- **Statistical validation** (Granger causality, logistic regression, ROC AUC)

In essence, we convert market price series into a *time-evolving network entanglement measure*, then verify whether spikes in systemic coupling (**C(t)**) precede actual market downturns (drawdowns).

---

## 📁 Project Structure

```

cyber-fractal-entanglement/
├── data/
│   ├── raw/                  # raw downloaded price data (via yfinance)
│   └── processed/            # cleaned prices, FD, QMI, C(t)
├── src/
│   ├── data_ingest.py        # fetch & align prices
│   ├── data_quality.py       # QC checks, return stats
│   ├── fractal.py            # rolling Higuchi FD extraction
│   ├── density.py            # covariance → density matrix
│   ├── qmi.py                # compute quantum mutual information
│   ├── network.py            # eigenvector centrality → C(t)
│   ├── validate_ct.py        # single-asset drawdown tests
│   ├── validate_ct_v2.py     # multi-asset, grid-search validation
│   ├── animate_qmi.py        # 3D animations of QMI over time
│   └── grid_search_validation.py  # orchestrates full grid and summary CSV outputs
├── notebooks/                # exploratory analysis notebooks
├── reports/
│   ├── figures/              # final assets: plots + animations
│   └── grid/                 # validation results + summary CSVs
├── tests/                    # unit tests for each module
├── LICENSE
└── README.md

```


---

## 🎯 Installation & Setup

1. **Clone and set up your environment**  
   ```bash
   git clone https://github.com/Jiven-Chana/cyber-fractal-entanglement.git
   cd cyber-fractal-entanglement
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2.	Download and preprocess price data

  ```bash

  python src/data_ingest.py
  python src/data_quality.py

  ```

Outputs are stored in data/processed/prices.csv.

3.	Compute fractal dimensions & build density matrices

  ```bash

  python src/fractal.py
  python src/density.py

  ```

Results: FD time series & rolling covariance → data/processed/density/.

4.	Compute pairwise QMI & systemic coupling C(t)

  ```bash

  python src/qmi.py
  python src/network.py

  ```

Output files: QMI time series and centrality signal in data/processed/qmi.csv and data/processed/centrality.csv.

5.	Run predictive validation
	•	Single-asset proof-of-concept:

  ```bash

  python src/validate_ct.py --asset SPX --threshold -0.01 --horizon 5

  ```

•	Multi-asset grid search (all baskets, thresholds, horizons):

  ```bash

  python src/validate_ct_v2.py

  ```

👉 Results are saved to reports/grid/summary_table.csv and significant_runs.csv.

6.	Visualize and animate QMI evolution

  ```bash

  python src/animate_qmi.py --input_csv data/processed/qmi.csv --output_mp4 reports/figures/qmi_evolution.mp4

  ```


⸻

📊 Key Insights
	1.	Cross-asset QMI entanglement (ANY3) achieved the highest AUC (~0.68) in predicting ≥3% drawdowns over a 5-day horizon—surpassing SPX-alone models.
	2.	SPX-alone signal achieved strong statistical significance (Granger p‑values ≈0.001–0.005) with AUC ≈0.61–0.63 for thresholds of −1% over 5–10 days.
	3.	Longer horizons (5–10d) demonstrated better signal quality than short-term ones (3d), suggesting anticipatory structural coupling.
	4.	Sensitivity to FD input anomalies was identified: fractal estimates >2.0 can lead to artificial QMI spikes. We recommend clipping to [1, 2], and optionally using robust covariance estimators or winsorization for a production version.

⸻

📎 Outputs & Artefacts
	•	reports/figures/qmi_evolution.mp4: 3D animation visualizing QMI evolution across asset pairs over time.
	•	reports/grid/*.csv: comprehensive grid-output, summary, and filterable significant results.
	•	reports/figures/roc_curves/: validation plots for each grid run.

⸻

✅ Validation
	•	Univariate: SPX-alone validation shows that coupling spikes lead drawdowns with strong statistical confidence.
	•	Multivariate: Combining SPX–GOLD–BTC delivers improved classification performance, supporting the thesis that cross-asset entanglement is a better systemic risk signal.

⸻

🔭 Future Directions
	•	Expand asset universe: include interest rates, VIX, credit spreads, commodities.
	•	Apply KARMA-style temporal decomposition: extract seasonal, cyclical, and high-frequency signal layers.
	•	Back-test over historic crises: 2008, 2015–16, 2020 COVID.
	•	Production-ready models: apply winsorization, robust covariance, real-time pipelines, and integrate signal into RL-based portfolio strategies.

⸻

📞 Contact & Codebase
	•	Full code, test suite, notebooks, and visual assets are available here.
	•	For questions or collaboration, feel free to open an issue or reach me at jiven.chana@icloud.com

⸻

📚 References
	•	Higuchi, T. (1988). Fractal Time Series Analysis. Physica D
	•	Nielsen, M. A., & Chuang, I. L. (2010). Quantum Computation and Quantum Information
	•	Mantegna, R. N., & Stanley, H. E. (1999). Introduction to Econophysics
	•	Barabási, A.-L. (2016). Network Science — Eigenvector centrality
	•	Hamilton, J. D. (1994). Time Series Analysis — Granger causality

⸻

🏁 TL;DR

This project builds a novel, academically rigorous pipeline that fuses fractal roughness, quantum-style entanglement, and network centrality into a statistical early-warning signal for market drawdowns—validated quantitatively and primed for broader research and deployment.

---
