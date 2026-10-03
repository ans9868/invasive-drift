"""Decoder zoo (spike counts -> kinematics). Linear + a kinematic Kalman filter.

Refs: Wiener filter / OLE (linear), Velocity-KF (Wu et al. 2006; Kim et al. 2008),
ReFIT-KF (Gilja et al. 2012 - adds cursor-position state), feedforward NN.
"""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor


def lagged(X, L):
    T, n = X.shape
    out = np.zeros((T - L, n * (L + 1)), np.float32)
    for i in range(L + 1):
        out[:, i * n:(i + 1) * n] = X[L - i:T - i]
    return out


class RidgeDec:
    name = "ridge"
    def fit(self, X, y):
        self.m = X.mean(0); self.s = X.std(0) + 1e-6
        self.mdl = Ridge(alpha=1.0).fit((X - self.m) / self.s, y); return self
    def predict(self, X):
        return self.mdl.predict((X - self.m) / self.s)


class WienerDec:
    """Linear FIR (Wiener) decoder with L history lags."""
    def __init__(self, L=5, alpha=1.0):
        self.L = L; self.alpha = alpha; self.name = f"wiener_L{L}"
    def fit(self, X, y):
        Xl = lagged(X, self.L); self.m = Xl.mean(0); self.s = Xl.std(0) + 1e-6
        self.mdl = Ridge(alpha=self.alpha).fit((Xl - self.m) / self.s, y[self.L:]); return self
    def predict(self, X):
        Xl = lagged(X, self.L); return self.mdl.predict((Xl - self.m) / self.s)


class MLPDec:
    """Nonlinear feedforward net on lagged spikes."""
    def __init__(self, L=3, hidden=128, max_iter=300):
        self.L = L; self.h = hidden; self.mi = max_iter; self.name = f"mlp_L{L}"
    def fit(self, X, y):
        Xl = lagged(X, self.L); self.m = Xl.mean(0); self.s = Xl.std(0) + 1e-6
        self.mdl = MLPRegressor(hidden_layer_sizes=(self.h, self.h), max_iter=self.mi,
                                early_stopping=True, random_state=0)
        self.mdl.fit((Xl - self.m) / self.s, y[self.L:]); return self
    def predict(self, X):
        Xl = lagged(X, self.L); return self.mdl.predict((Xl - self.m) / self.s)


class KalmanDec:
    """Kinematic Kalman filter.  state = [pos(2), vel(2), 1]  (mode='posvel', Gilja) or
    [vel(2),1] (mode='vel').  Observation z_t = C y_t (linear decode); A = kinematic."""
    def __init__(self, mode="posvel", dt=0.02):
        self.mode = mode; self.dt = dt; self.name = f"kf_{mode}"
    def _state(self, pos, vel):
        if self.mode == "posvel":
            return np.column_stack([pos, vel, np.ones(len(vel))])
        return np.column_stack([vel, np.ones(len(vel))])
    def fit(self, X, pos, vel):
        Z = self._state(pos, vel)
        self.Xm = X.mean(0); self.Xs = X.std(0) + 1e-6
        reg = Ridge(alpha=10.0).fit((X - self.Xm) / self.Xs, Z)
        self.C = reg.coef_.T; self.c0 = reg.intercept_
        Zhat = (X - self.Xm) / self.Xs @ self.C + self.c0
        dt = self.dt
        if self.mode == "posvel":
            A = np.array([[1,0,dt,0,0],[0,1,0,dt,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]], float)
        else:
            A = np.array([[1,0,0],[0,1,0],[0,0,1]], float)
        self.A = A; n = Z.shape[1]; self.n = n
        self.R = np.atleast_2d(np.cov((Z - Zhat).T)) + np.eye(n) * 1e-6
        self.W = np.atleast_2d(np.cov((Z[1:] - Z[:-1] @ A.T).T)) + np.eye(n) * 1e-6
        self.x0 = Z.mean(0); self.P0 = np.eye(n)
        return self
    def predict(self, X):
        n = self.n; A = self.A; I = np.eye(n)
        Z = (X - self.Xm) / self.Xs @ self.C + self.c0
        x = self.x0.copy(); P = self.P0.copy(); out = np.zeros((len(X), 2))
        for t in range(len(X)):
            x = A @ x; P = A @ P @ A.T + self.W
            K = P @ np.linalg.inv(P + self.R)
            x = x + K @ (Z[t] - x); P = (I - K) @ P
            out[t] = x[2:4] if self.mode == "posvel" else x[0:2]
        return out


# ---- recurrent (GRU) decoder (torch) ----
try:
    import torch
    import torch.nn as nn

    class _GRUNet(nn.Module):
        def __init__(self, n_in, hidden=64):
            super().__init__()
            self.gru = nn.GRU(n_in, hidden, batch_first=True)
            self.fc = nn.Linear(hidden, 2)
        def forward(self, x):
            o, _ = self.gru(x)
            return self.fc(o[:, -1])
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


def _windows(X, L):
    T, n = X.shape
    idx = np.arange(L)[None, :] + np.arange(T - L)[:, None]
    return X[idx]


class GRUDec:
    """Recurrent (GRU) decoder over a sliding window of L bins."""
    def __init__(self, L=10, hidden=64, epochs=4, bs=512, lr=1e-3):
        self.L = L; self.hidden = hidden; self.epochs = epochs; self.bs = bs; self.lr = lr
        self.name = f"gru_L{L}"
    def fit(self, X, y):
        import torch, torch.nn as nn
        torch.manual_seed(0)
        self.m = X.mean(0); self.s = X.std(0) + 1e-6
        Xw = _windows(((X - self.m) / self.s).astype("float32"), self.L)
        yw = y[self.L:].astype("float32")
        self.net = _GRUNet(X.shape[1], self.hidden)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        lossf = nn.MSELoss()
        Xt = torch.from_numpy(Xw); Yt = torch.from_numpy(yw); N = len(Xt)
        for _ in range(self.epochs):
            perm = torch.randperm(N)
            for i in range(0, N, self.bs):
                b = perm[i:i + self.bs]
                opt.zero_grad(); loss = lossf(self.net(Xt[b]), Yt[b]); loss.backward(); opt.step()
        return self
    def predict(self, X):
        import torch
        Xw = _windows(((X - self.m) / self.s).astype("float32"), self.L)
        self.net.eval()
        with torch.no_grad():
            return self.net(torch.from_numpy(Xw)).numpy()
