import matplotlib.pyplot as plt
import numpy as np

from hmm import GaussianHMM
from simulator import simulate_market_data


def main():
    # 1. Generate Synthetic Financial Returns Data
    returns, true_states, true_A, true_means, true_stds = simulate_market_data(
        T=1000, seed=42
    )

    # 2. Fit HMM via Baum-Welch (EM)
    hmm = GaussianHMM(n_states=2, max_iter=100)
    hmm.fit(returns)

    # 3. Perform Forward-Backward Inference
    posterior_probs = hmm.predict_proba(returns)

    # 4. Print Parameter Output
    print("\n" + "=" * 45)
    print("ESTIMATED HMM PARAMETERS")
    print("=" * 45)
    print(
        f"Bearish Regime  -> Mean: {hmm.means[0]:.5f}, Volatility: {np.sqrt(hmm.vars[0]):.5f}"
    )
    print(
        f"Bullish Regime  -> Mean: {hmm.means[1]:.5f}, Volatility: {np.sqrt(hmm.vars[1]):.5f}"
    )
    print("\nTransition Matrix A:")
    print(np.round(hmm.A, 3))

    # 5. Plot Results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # Plot Returns
    ax1.plot(returns, color="gray", alpha=0.5, label="Simulated Returns")
    ax1.set_title("Simulated Financial Return Time-Series")
    ax1.set_ylabel("Returns")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Plot State Probabilities
    ax2.plot(posterior_probs[:, 1], color="green", label="P(State = Bullish)")
    ax2.axhline(0.5, color="red", linestyle="--", alpha=0.7, label="Regime Boundary")
    ax2.set_title("Forward-Backward Inferred Probability (Bullish Regime)")
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Probability")
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()