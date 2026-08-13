import numpy as np


class GaussianHMM:
    """Hidden Markov Model with Gaussian Emissions for Market Regime Detection."""

    def __init__(self, n_states=2, max_iter=100, tol=1e-5):
        self.n_states = n_states
        self.max_iter = max_iter
        self.tol = tol

        # Model Parameters
        self.pi = None  # Initial state probability vector (n_states,)
        self.A = None  # Transition probability matrix (n_states, n_states)
        self.means = None  # Emission Gaussian means (n_states,)
        self.vars = None  # Emission Gaussian variances (n_states,)

    def _gaussian_pdf(self, x, mean, var):
        """Calculates univariate Gaussian probability density function."""
        denom = np.sqrt(2 * np.pi * var)
        num = np.exp(-0.5 * ((x - mean) ** 2) / var)
        return np.maximum(num / denom, 1e-12)

    def _compute_emissions(self, X):
        """Computes emission probability matrix B where B[t, i] = P(x_t | z_t = i)."""
        T = len(X)
        B = np.zeros((T, self.n_states))
        for i in range(self.n_states):
            B[:, i] = self._gaussian_pdf(X, self.means[i], self.vars[i])
        return B

    def _forward(self, B):
        """Forward Algorithm with scaling to prevent numerical underflow."""
        T = len(B)
        alpha_hat = np.zeros((T, self.n_states))
        c = np.zeros(T)

        alpha_raw = self.pi * B[0]
        c[0] = np.sum(alpha_raw)
        alpha_hat[0] = alpha_raw / c[0]

        for t in range(1, T):
            alpha_raw = (alpha_hat[t - 1] @ self.A) * B[t]
            c[t] = np.sum(alpha_raw)
            alpha_hat[t] = alpha_raw / c[t]

        return alpha_hat, c

    def _backward(self, B, c):
        """Backward Algorithm using scaling factors from Forward pass."""
        T = len(B)
        beta_hat = np.zeros((T, self.n_states))
        beta_hat[-1] = 1.0 / c[-1]

        for t in range(T - 2, -1, -1):
            beta_raw = self.A @ (B[t + 1] * beta_hat[t + 1])
            beta_hat[t] = beta_raw / c[t]

        return beta_hat

    def fit(self, X):
        """Estimates HMM parameters (pi, A, means, vars) using Baum-Welch (EM)."""
        T = len(X)
        np.random.seed(42)

        # 1. Initialization
        self.pi = np.ones(self.n_states) / self.n_states
        self.A = np.full((self.n_states, self.n_states), 1.0 / self.n_states)

        quantiles = np.linspace(0.2, 0.8, self.n_states)
        self.means = np.quantile(X, quantiles)
        self.vars = np.var(X) * np.ones(self.n_states)

        old_log_likelihood = -np.inf

        for iteration in range(self.max_iter):
            # --- E-STEP ---
            B = self._compute_emissions(X)
            alpha_hat, c = self._forward(B)
            beta_hat = self._backward(B, c)

            gamma = alpha_hat * beta_hat
            gamma /= np.sum(gamma, axis=1, keepdims=True)

            xi = np.zeros((T - 1, self.n_states, self.n_states))
            for t in range(T - 1):
                numerator = (
                    alpha_hat[t][:, None]
                    * self.A
                    * B[t + 1][None, :]
                    * beta_hat[t + 1][None, :]
                )
                xi[t] = numerator / np.sum(numerator)

            log_likelihood = np.sum(np.log(c))
            if np.abs(log_likelihood - old_log_likelihood) < self.tol:
                print(f"Baum-Welch converged at iteration {iteration + 1}.")
                break
            old_log_likelihood = log_likelihood

            # --- M-STEP ---
            self.pi = gamma[0]
            self.A = np.sum(xi, axis=0) / np.sum(gamma[:-1], axis=0)[:, None]

            gamma_sum = np.sum(gamma, axis=0)
            self.means = np.sum(gamma * X[:, None], axis=0) / gamma_sum
            self.vars = (
                np.sum(gamma * ((X[:, None] - self.means[None, :]) ** 2), axis=0)
                / gamma_sum
            )

        # Sort states: State 0 = Bearish (lower mean), State 1 = Bullish (higher mean)
        sort_idx = np.argsort(self.means)
        self.means = self.means[sort_idx]
        self.vars = self.vars[sort_idx]
        self.pi = self.pi[sort_idx]
        self.A = self.A[sort_idx][:, sort_idx]

    def predict_proba(self, X):
        """Computes posterior regime probabilities using Forward-Backward."""
        B = self._compute_emissions(X)
        alpha_hat, c = self._forward(B)
        beta_hat = self._backward(B, c)

        gamma = alpha_hat * beta_hat
        gamma /= np.sum(gamma, axis=1, keepdims=True)
        return gamma

    def predict(self, X):
        """Classifies each time step into the most likely regime state."""
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)