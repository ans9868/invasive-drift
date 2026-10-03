"""Adapter base class, registry, and shared numeric helpers."""
import numpy as np


class Adapter:
    name = "base"
    stage = "feature"          # "feature" | "output"
    uses_labels = False
    uses_targets = False
    uses_decoder = False
    is_trainable = False
    causal = True
    n_params = 0

    def fit(self, Zfit, ref, decoder=None, y=None, dirbin=None):
        """Fit on the FIT POOL only (Zfit: (n,d) standardised features). Returns self."""
        raise NotImplementedError

    def apply(self, A):
        """Apply to features (stage=feature) or to predictions (stage=output)."""
        raise NotImplementedError

    def describe(self):
        return dict(name=self.name, stage=self.stage, uses_labels=self.uses_labels,
                    uses_targets=self.uses_targets, uses_decoder=self.uses_decoder,
                    is_trainable=self.is_trainable, causal=self.causal, n_params=self.n_params)


REGISTRY = {}


def register(cls):
    REGISTRY[cls.name] = cls
    return cls


def get(name):
    return REGISTRY[name]()


def list_adapters():
    return sorted(REGISTRY)


def describe_all():
    return [REGISTRY[k]().describe() for k in sorted(REGISTRY)]


# ---- shared numeric helpers -------------------------------------------------

def shrunk_cov(X, lam=1e-3):
    """Covariance with trace-scaled ridge (numerically safe)."""
    C = np.cov(X, rowvar=False)
    C = np.atleast_2d(C)
    d = C.shape[0]
    return C + lam * (np.trace(C) / d + 1e-12) * np.eye(d)


def _sym_pow(C, p):
    """Matrix power C**p via eigendecomposition (C symmetric PSD, shrunk)."""
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 1e-12, None)
    return (V * (w ** p)) @ V.T


def cov_align_map(Cw, C0):
    """Affine (linear part) mapping a distribution with covariance Cw to one with C0.

    A = Cw^{-1/2} C0^{1/2}   (minimum-MSE linear map between the two covariance shapes).
    """
    return _sym_pow(Cw, -0.5) @ _sym_pow(C0, 0.5)


def row_space_projector(W):
    """Orthogonal projector onto the row space of the decoder W (2 x d or (k,d))."""
    W = np.atleast_2d(W)
    G = W @ W.T + 1e-9 * np.eye(W.shape[0])
    return W.T @ np.linalg.solve(G, W)


def procrustes(A, B):
    """Orthogonal R, plus offsets, s.t. (A - a_bar) R + b_bar ~ B (rows = points)."""
    Ac = A - A.mean(0); Bc = B - B.mean(0)
    U, _, Vt = np.linalg.svd(Bc.T @ Ac)
    R = U @ Vt
    return R, A.mean(0), B.mean(0)
