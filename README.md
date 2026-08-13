# Market Regime Detection using Hidden Markov Models

A modular Python implementation of a **Gaussian Hidden Markov Model (HMM)** built from scratch using `NumPy`. This project models financial return time-series to detect hidden market regimes (such as **Bullish** vs. **Bearish** states), using the **Baum-Welch Expectation-Maximization (EM)** algorithm for parameter fitting and the **Forward-Backward algorithm** for state probability inference.

---

## Key Features

* **Custom Gaussian HMM Core**: Built completely from scratch without high-level ML framework dependencies (like `hmmlearn`).
* **Baum-Welch (EM) Parameter Estimation**: Iteratively updates initial state distribution ($\pi$), transition probabilities ($A$), and Gaussian emission parameters ($\mu_k, \sigma_k^2$).
* **Forward-Backward Inference**: Evaluates posterior state probabilities $\gamma_t(i) = P(z_t = i \mid X_{1:T})$ with scaling to prevent numerical underflow.
* **Regime-Switching Simulation**: Includes a synthetic data generator modeling financial returns under high-volatility/bearish and low-volatility/bullish states.

---

## Project Structure

```text
market-regime-detection-hmm/
├── hmm.py          # Gaussian HMM class (Baum-Welch & Forward-Backward algorithms)
├── simulator.py    # Synthetic regime-switching financial time-series generator
├── main.py         # Model training, regime inference, and visualization
└── README.md
