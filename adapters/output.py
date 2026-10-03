"""Output-stage adapters (applied to the decoder's PREDICTIONS, after decoding).

These are the DECODER-ALIGNED family: the correction lives in the ~2-D space the decoder actually
produces, instead of the d-dimensional feature space (which is mostly null space for a linear decoder).
"""
import numpy as np
from .base import Adapter, register, shrunk_cov, cov_align_map


@register
class OutMom(Adapter):
    """ALIGNED + label-free: match the distribution of predicted velocity to the burn-in velocity.

    ~6 params (2x2 affine + offset) -> vastly better conditioned than a d x d feature-space map.
    """
    name = "out_mom"
    stage = "output"
    uses_decoder = True
    n_params = 6

    def fit(self, Zfit, ref, decoder=None, y=None, dirbin=None):
        self.noop = decoder is None
        if self.noop:
            return self
        V = np.atleast_2d(decoder.predict(Zfit))          # (n,2)
        self.mf = V.mean(0)
        self.S = cov_align_map(shrunk_cov(V), ref["v_cov0"])
        self.mu0 = ref["v_mu0"]
        return self

    def apply(self, V):
        if self.noop:
            return V
        return (V - self.mf) @ self.S + self.mu0


@register
class OutAffine(Adapter):
    """LABELED twin of `out_mom`: 2x2 affine fit by least squares to minimise decode error (ceiling-ish)."""
    name = "out_affine"
    stage = "output"
    uses_decoder = True
    uses_labels = True
    n_params = 6

    def fit(self, Zfit, ref, decoder=None, y=None, dirbin=None):
        self.noop = (decoder is None) or (y is None)
        if self.noop:
            return self
        V = np.atleast_2d(decoder.predict(Zfit))
        X = np.c_[V, np.ones(len(V))]
        self.A, *_ = np.linalg.lstsq(X, y, rcond=None)
        return self

    def apply(self, V):
        if self.noop:
            return V
        return np.c_[V, np.ones(len(V))] @ self.A
