"""Feature-stage adapters (applied to standardised features Z *before* decoding)."""
import numpy as np
from .base import (Adapter, register, shrunk_cov, cov_align_map, row_space_projector,
                   procrustes, _sym_pow)

_EPS = 1e-6


@register
class Identity(Adapter):
    """No-op control: must reproduce `frozen` exactly."""
    name = "identity"
    stage = "feature"
    n_params = 0

    def fit(self, Zfit, ref, decoder=None, y=None, dirbin=None):
        return self

    def apply(self, A):
        return A


@register
class ShuffledRef(Adapter):
    """Negative control (from f26): per-unit moment match but with the reference PERMUTED across units.

    A genuine per-unit correction must beat this; if it does not, the effect is generic, not identity-based.
    """
    name = "shuffled_ref"
    stage = "feature"
    n_params = 0

    def fit(self, Zfit, ref, decoder=None, y=None, dirbin=None):
        d = Zfit.shape[1]
        self.perm = np.random.default_rng(0).permutation(d)
        self.mu0 = ref["mu0"]; self.sd0 = ref["sd0"]
        return self

    def apply(self, A):
        mw = A.mean(0); sw = A.std(0) + _EPS
        return (A - mw) / sw * self.sd0[self.perm] + self.mu0[self.perm]


@register
class MomGlobal(Adapter):
    """One global mean+std for all units (2 params)."""
    name = "mom_global"
    stage = "feature"
    n_params = 2

    def fit(self, Zfit, ref, decoder=None, y=None, dirbin=None):
        self.m0 = float(ref["mu0"].mean()); self.s0 = float(ref["sd0"].mean())
        return self

    def apply(self, A):
        return (A - A.mean()) / (A.std() + _EPS) * self.s0 + self.m0


class _MomDiag(Adapter):
    """Per-unit mean+std match (the f26 incumbent). `self_moments=True` = two-pass (non-causal)."""
    stage = "feature"
    n_params = None

    def fit(self, Zfit, ref, decoder=None, y=None, dirbin=None):
        self.mu0 = ref["mu0"]; self.sd0 = ref["sd0"]
        self.mf = Zfit.mean(0); self.sf = Zfit.std(0) + _EPS     # fit-pool moments (causal default)
        self.n_params = 2 * Zfit.shape[1]
        return self

    def apply(self, A):
        if getattr(self, "self_moments", False):
            mw = A.mean(0); sw = A.std(0) + _EPS
        else:
            mw = self.mf; sw = self.sf
        return (A - mw) / sw * self.sd0 + self.mu0


@register
class MomDiag(_MomDiag):
    """Causal per-unit moment match (moments from the FIT POOL)."""
    name = "mom_diag"
    self_moments = False


@register
class MomDiagSelf(_MomDiag):
    """Two-pass per-unit moment match (moments from the data being decoded) — the f25 win."""
    name = "mom_diag_self"
    self_moments = True
    causal = False


@register
class Zca(Adapter):
    """Full-rank covariance alignment to the reference: A = Cf^{-1/2} C0^{1/2} (min-MSE linear map).

    NOTE: this subsumes the plan's `reg_ref` (an unpaired least-squares "regression to the reference"
    is ill-posed / equals this). Kept only as `zca`; see the module docstring.
    """
    name = "zca"
    stage = "feature"

    def fit(self, Zfit, ref, decoder=None, y=None, dirbin=None):
        self.mu0 = ref["mu0"]; self.C0 = ref["C0"]
        self.mf = Zfit.mean(0)
        self.S = cov_align_map(shrunk_cov(Zfit - self.mf), self.C0)
        self.n_params = Zfit.shape[1] ** 2
        return self

    def apply(self, A):
        return (A - self.mf) @ self.S + self.mu0


@register
class CovLowrank(Adapter):
    """Mean + covariance match restricted to the top-k PCA subspace of the reference."""
    name = "cov_lowrank"
    stage = "feature"

    def fit(self, Zfit, ref, decoder=None, y=None, dirbin=None):
        self.mu0 = ref["mu0"]; self.P = ref["P"]; self.mf = Zfit.mean(0)
        U = (Zfit - self.mf) @ self.P
        self.Sk = cov_align_map(shrunk_cov(U), ref["C0k"])
        self.n_params = Zfit.shape[1] * self.P.shape[1] + self.P.shape[1] ** 2
        return self

    def apply(self, A):
        Ac = A - self.mf; U = Ac @ self.P
        return Ac + (U @ self.Sk - U) @ self.P.T + self.mu0


@register
class Subspace(Adapter):
    """Subspace alignment (SA): align the fit window's top-k principal subspace onto the reference's.

    NOTE: the plan called this cell `cca`, but CCA requires PAIRED samples (the same trials in two views).
    An unpaired fit-window vs burn-in reference provides none, so unpaired CCA is ill-posed (it reduces to
    ZCA). Orthogonal subspace alignment is the well-posed unpaired analogue, so this cell is implemented
    as SA. (`reg_ref` is likewise subsumed by `zca`.)
    """
    name = "subspace"
    stage = "feature"
    k = 5

    def fit(self, Zfit, ref, decoder=None, y=None, dirbin=None):
        self.mf = Zfit.mean(0); self.mu0 = ref["mu0"]
        Pw = np.linalg.svd(Zfit - self.mf, full_matrices=False)[2][:self.k].T
        P0 = ref["P"]
        self.T = Pw @ (Pw.T @ P0) @ P0.T
        self.n_params = 2 * Zfit.shape[1] * min(Pw.shape[1], P0.shape[1])
        return self

    def apply(self, A):
        return (A - self.mf) @ self.T + self.mu0


@register
class CentroidProc(Adapter):
    """GRAY: orthogonal Procrustes on per-direction landmark centroids, restricted to the landmark span."""
    name = "centroid_proc"
    stage = "feature"
    uses_targets = True
    k = 7

    def _centroids(self, Z, ref, dirbin):
        K = ref["dirC"].shape[0]
        C = np.zeros_like(ref["dirC"]); ok = np.zeros(K, bool)
        for kk in range(K):
            m = (dirbin == kk) if dirbin is not None else np.zeros(len(Z), bool)
            if m.sum() >= 20:
                C[kk] = Z[m].mean(0); ok[kk] = True
        return C, ok

    def fit(self, Zfit, ref, decoder=None, y=None, dirbin=None):
        Cw, okw = self._centroids(Zfit, ref, dirbin)
        com = ref["dirOK"] & okw
        self.noop = com.sum() < 3
        if self.noop:
            return self
        A, B = Cw[com], ref["dirC"][com]
        Bc = B - B.mean(0)
        _, _, VtA = np.linalg.svd(Bc, full_matrices=False)
        q = max(1, min(self.k, int(com.sum()) - 1, VtA.shape[0]))
        self.P = VtA[:q].T
        self.R, self.ab, self.bb = procrustes(A @ self.P, B @ self.P)
        self.n_params = Zfit.shape[1] * q
        return self

    def apply(self, A):
        if self.noop:
            return A
        U = A @ self.P
        return A - U @ self.P.T + ((U - self.ab) @ self.R + self.bb) @ self.P.T


@register
class NullProj(Adapter):
    """ALIGNED: per-unit moment correction, projected onto the decoder's row space (null space untouched).

    Linear decoders only (uses `decoder.coef_`); no-op otherwise.
    """
    name = "null_proj"
    stage = "feature"
    uses_decoder = True

    def fit(self, Zfit, ref, decoder=None, y=None, dirbin=None):
        self.mu0 = ref["mu0"]; self.sd0 = ref["sd0"]
        self.mf = Zfit.mean(0); self.sf = Zfit.std(0) + _EPS
        W = getattr(decoder, "coef_", None)
        self.PW = row_space_projector(W) if W is not None else None
        self.noop = self.PW is None
        self.n_params = 2 * Zfit.shape[1]
        return self

    def apply(self, A):
        if self.noop:
            return A
        mom = (A - self.mf) / self.sf * self.sd0 + self.mu0
        return A + (mom - A) @ self.PW

    self_moments = True
    causal = False
