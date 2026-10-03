"""Shared cell evaluation for the exp-27 grid and the staleness instrument.

ONE place that encodes:
  - the ROW-ALIGNMENT RULE (wiener/mlp/gru return len(X)-L predictions -> align targets to the TAIL)
  - adapter stage handling ("feature" applied before decoding, "output" applied after)
  - no-op detection
  - the metric set from metrics.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import metrics as M        # noqa: E402
import adapters as A       # noqa: E402,F401


def align_tail(y, p):
    """Lagged decoders (wiener L=5, mlp L=3, gru L=10) drop rows -> align y to the TAIL of p."""
    y = np.asarray(y)
    return y[len(y) - len(p):] if len(p) <= len(y) else y


def is_noop(adapter):
    return adapter is None or bool(getattr(adapter, "noop", False))


def _stage(adapter):
    return getattr(adapter, "stage", "feature")


def fit_and_apply(adapter, decoder, ref, Ztr, ytr, dirbin_tr, Zte):
    """Fit the adapter on the TRAIN rows, apply to the TEST rows. Returns (pred, noop)."""
    adapter.fit(Ztr, ref, decoder=decoder, y=ytr, dirbin=dirbin_tr)
    if is_noop(adapter):
        return np.asarray(decoder.predict(Zte)), True
    if _stage(adapter) == "feature":
        return np.asarray(decoder.predict(adapter.apply(Zte))), False
    return np.asarray(adapter.apply(np.asarray(decoder.predict(Zte)))), False


def evaluate_cell(decoder, Ztr, ytr, Zte, yte, ref=None, adapter=None, dirbin_tr=None,
                  moving_te=None):
    """Full metric dict for one (train-rows -> test-rows) cell."""
    if is_noop(adapter):
        pred, noop = np.asarray(decoder.predict(Zte)), True
    else:
        pred, noop = fit_and_apply(adapter, decoder, ref, Ztr, ytr, dirbin_tr, Zte)
    yt = align_tail(yte, pred)
    mv = align_tail(np.asarray(moving_te, bool), pred) if moving_te is not None else None
    out = {}
    out.update(M.velocity_metrics(yt, pred))
    out.update(M.direction_metrics(yt, pred, mv if mv is not None else np.ones(len(yt), bool)))
    out["lag_bins"] = M.lag_bins(yt, pred)
    out["n_eval"] = int(len(pred))
    out["noop"] = bool(noop)
    out["n_params"] = int(getattr(adapter, "n_params", 0) or 0) if adapter is not None else 0
    out.update(correction_size(adapter, decoder, Zte) if not noop else dict(corr_rel=0.0, corr_rank=0))
    out["_pred"] = pred              # for cross-decoder agreement (caller pops it)
    return out


def correction_size(adapter, decoder, Zprobe):
    """How much did this adapter actually change the decoder's input/output? (scale-invariant)"""
    if is_noop(adapter):
        return dict(corr_rel=0.0, corr_rank=0)
    base = Zprobe if _stage(adapter) == "feature" else np.asarray(decoder.predict(Zprobe))
    new = np.asarray(adapter.apply(base))
    d = new - base
    return dict(corr_rel=float(np.linalg.norm(d) / (np.linalg.norm(base) + 1e-12)),
                corr_rank=int(np.linalg.matrix_rank(d)))


def ctx_row_to_dict(ctx, ctx_names, w):
    """Window context row -> named dict (for writing into every result row)."""
    return {str(nm): float(ctx[w, i]) for i, nm in enumerate(ctx_names)}


SCHEME_COLS = ("scheme", "gap_min", "calib_min")
