import numpy as np


def simulate_market_data(T=1000, seed=42):
    """Simulates market returns under Bearish and Bullish regimes."""
    np.random.seed(seed)

    # State Parameters
    # State 0 (Bearish): Mean = -0.15%, Volatility = 2.0%
    # State 1 (Bullish): Mean = +0.10%, Volatility = 0.8%
    true_means = np.array([-0.0015, 0.0010])
    true_stds = np.array([0.020, 0.008])
    true_A = np.array([[0.92, 0.08], [0.05, 0.95]])

    states = np.zeros(T, dtype=int)
    returns = np.zeros(T)

    states[0] = 1  # Start in Bullish regime
    returns[0] = np.random.normal(true_means[states[0]], true_stds[states[0]])

    for t in range(1, T):
        states[t] = np.random.choice([0, 1], p=true_A[states[t - 1]])
        returns[t] = np.random.normal(true_means[states[t]], true_stds[states[t]])

    return returns, states, true_A, true_means, true_stds