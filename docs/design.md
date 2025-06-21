High level decisions made, systems diagrams and more

Both Higuchi‐FD and Hurst‐exponent have their merits—here’s how to choose:

Criterion	Higuchi	Hurst
Robustness to Noise	More robust on short, noisy financial series.	Sensitive to non-stationarity and outliers.
Window Size	Can work on very small windows (≥20–30 points).	Needs longer windows (≥100 points) for stability.
Interpretability	Direct fractal dimension 1 < D < 2.	Hurst 0<H<1 interprets persistence (H>0.5).
Computation Speed	A bit slower per window, but fine for N≈5 assets.	Faster, since it’s just a polyfit on lags.
Literature Precedent	Widely used in econophysics and financial papers.	Classic in time-series; less common in fractal‐QM pipelines.
Implementation	Provided out‐of‐the‐box by nolds.	You code up the lag–tau fit yourself.


⸻

Decision for POC
	•	Start with Higuchi-FD
• It’s battle-tested on rolling windows of 30–100 daily points.
• nolds.higuchi_fd handles the internal details (no need to reinvent the R/S logic).
• You’ll get a clean D_i(t) series in the typical [1.1,1.9] range.
	•	Validate with Hurst as a robustness check
Once Higuchi-based results are in, you can switch --method hurst to confirm that the systemic-entanglement signal C(t) is qualitatively similar. If both methods produce consistent early-warning spikes, your POC is stronger.

⸻

How to switch in code (src/fractal.py)

# Higuchi (default)
python src/fractal.py -i data/processed/prices.csv -w 30

# Hurst
python src/fractal.py -i data/processed/prices.csv -w 100 -m hurst

Use a longer window (e.g. 100) for Hurst to get stable estimates. Then compare the two C(t) series in network‐analysis notebook.


If you leave those few FD values above 2.0 in place, nothing in the code will blow up—the covariance, density‐matrix and QMI routines don’t enforce theoretical bounds on the inputs. However, you will carry those numerical “glitches” right through to your network:
	•	Inflated variances on days when D_i(t) spikes above 2 will boost the off-diagonal covariances \Sigma_t.
	•	When you normalize to \rho_t, those days become high-weight slices of your density matrix.
	•	The von Neumann entropy and QMI will then register artificially high entanglements on those dates—even if the market didn’t really become that “complex.”
	•	Your systemic-entanglement score C(t) will likely show spurious spikes (false alarms) wherever the FD algorithm overshot.

⸻

When it might be OK to leave them
	•	Exploratory POC: if you just want to sanity-check the plumbing of your pipeline, you can leave them and eyeball when the spikes happen—then inspect whether they coincide with real events.
	•	Sensitivity test: you can run the whole pipeline twice (once with clipping, once without) and compare the two C(t) series to see how much those out-of-range points actually move your early-warning indicator.

⸻

When you should clean them
	•	Final prototype / paper: stick to the fractal theory and clip (or winsorize) to [1,2]. That way your covariance network reflects genuine “roughness” co-movements, not numerical artifacts.
	•	Robustness: if you ever push this toward production, you’ll want a systematic winsorization or a robust covariance estimator (e.g. Minimum Covariance Determinant) to guard against any stray FD blow-ups.