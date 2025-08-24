# dynamics_learner/gp.py
import numpy as np
from typing import Sequence, Tuple
from .base import DynamicsLearner

def rbf_kernel(X1: np.ndarray, X2: np.ndarray, lengthscale: float, variance: float):
    """
    RBF kernel (squared exponential).
    X1: [n1, d], X2: [n2, d]
    returns: [n1, n2]
    """
    X1s = np.sum(X1**2, axis=1, keepdims=True)
    X2s = np.sum(X2**2, axis=1, keepdims=True)
    d2 = X1s - 2 * X1.dot(X2.T) + X2s.T
    return variance * np.exp(-0.5 * d2 / (lengthscale**2))


class SlidingWindowGP(DynamicsLearner):
    """
    Exact GP on a sliding-window dataset.
    - stores up to maxlen samples
    - supports multi-output Y (shape [N, r_dim])
    - prediction: mean (r_dim,) and var (r_dim,)  (homoscedastic across dims here)
    """

    def __init__(self,
                 x_dim: int,
                 u_dim: int,
                 r_dim: int,
                 maxlen: int = 300,
                 lengthscale: float = 1.0,
                 variance: float = 1.0,
                 noise_variance: float = 1e-3,
                 jitter: float = 1e-8):
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.in_dim = x_dim + u_dim
        self.r_dim = r_dim
        self.maxlen = maxlen
        self.lengthscale = lengthscale
        self.variance = variance
        self.noise_variance = noise_variance
        self.jitter = jitter

        # Buffers
        self.X = np.zeros((0, self.in_dim), dtype=float)
        self.Y = np.zeros((0, self.r_dim), dtype=float)

        # Precomputed matrices
        self.K = None      # kernel matrix (N,N)
        self.L = None      # cholesky of K + noise
        self.alpha = None  # solves K^{-1}Y

    def _form_input(self, x: Sequence[float], u: Sequence[float]) -> np.ndarray:
        return np.asarray(np.concatenate([np.asarray(x).ravel(), np.asarray(u).ravel()]),
                          dtype=float).reshape(1, -1)

    def predict(self, x: Sequence[float], u: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
        if len(self.X) == 0:
            return np.zeros(self.r_dim), np.full(self.r_dim, self.variance + self.noise_variance)

        x_star = self._form_input(x, u)  # [1, in_dim]
        K_star = rbf_kernel(x_star, self.X, self.lengthscale, self.variance)  # [1, N]
        K_ss = rbf_kernel(x_star, x_star, self.lengthscale, self.variance).reshape(())

        try:
            alpha = self.alpha                      # [N, r_dim]
            mean = (K_star @ alpha).ravel()         # (r_dim,)
            v = np.linalg.solve(self.L, K_star.T)   # [N,1]
            var_scalar = K_ss - (v.T @ v).ravel()[0]
            var_scalar = max(var_scalar, 1e-12)
            var = np.full(self.r_dim, var_scalar) + self.noise_variance
            return mean, var
        except Exception:
            K = self.K + (self.noise_variance + self.jitter) * np.eye(len(self.X))
            try:
                L = np.linalg.cholesky(K)
                invK = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(L.shape[0])))
                mean = (K_star @ invK @ self.Y).ravel()
                var_scalar = K_ss - (K_star @ invK @ K_star.T).ravel()[0]
                var = np.full(self.r_dim, max(var_scalar, 1e-12) + self.noise_variance)
                return mean, var
            except np.linalg.LinAlgError:
                return (K_star @ np.zeros((self.X.shape[0], self.r_dim))).ravel(), \
                       np.full(self.r_dim, self.variance)

    def update(self, batch):
        """
        Append data from batch into the sliding window and recompute factorization.
        Keep at most maxlen samples.
        Assumes inputs/targets already normalized by caller if desired.
        """
        if len(batch) == 0:
            return
        X_new, Y_new = [], []
        for x, u, r in batch:
            X_new.append(np.concatenate([np.asarray(x).ravel(), np.asarray(u).ravel()]))
            Y_new.append(np.asarray(r).ravel())
        X_new = np.asarray(X_new, dtype=float)
        Y_new = np.asarray(Y_new, dtype=float)

        if self.X.shape[0] == 0:
            self.X = X_new.copy()
            self.Y = Y_new.copy()
        else:
            self.X = np.vstack([self.X, X_new])
            self.Y = np.vstack([self.Y, Y_new])

        if len(self.X) > self.maxlen:
            self.X = self.X[-self.maxlen:]
            self.Y = self.Y[-self.maxlen:]

        K = rbf_kernel(self.X, self.X, self.lengthscale, self.variance)
        K += (self.noise_variance + self.jitter) * np.eye(len(K))
        try:
            L = np.linalg.cholesky(K)
            v = np.linalg.solve(L, self.Y)
            alpha = np.linalg.solve(L.T, v)  # [N, r_dim]
            self.K = K
            self.L = L
            self.alpha = alpha
        except np.linalg.LinAlgError:
            # add more jitter and retry
            K += 1e-6 * np.eye(K.shape[0])
            L = np.linalg.cholesky(K)
            v = np.linalg.solve(L, self.Y)
            alpha = np.linalg.solve(L.T, v)
            self.K = K
            self.L = L
            self.alpha = alpha

    def set_mode(self, mode: str = 'eval') -> None:
        return
