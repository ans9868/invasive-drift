"""adapters — correction library for `grid_within_session_27`.

An *adapter* maps either features (before the decoder) or predictions (after it) to try to undo drift,
**without** (usually) looking at velocity labels.

Interface
---------
Each adapter declares:
  stage          "feature" (applied to Z before decode) or "output" (applied to predictions after)
  uses_labels    needs velocity labels at fit time
  uses_targets   needs the target/condition identity (gray)
  uses_decoder   needs the frozen decoder (decoder-ALIGNED methods)
  is_trainable   gradient-based (else closed-form)
  causal         fit pool precedes the eval set (always True in this harness)
  n_params       rough parameter count (for the complexity column)

Export a class per adapter and decorate with @register.

`ref` (the reference dict, built once per session from BURN-IN only) carries:
  m0, s0        (d,) raw-feature scaler used to define Z            [burn-in only]
  mu0, sd0      (d,) mean/std of Z over burn-in
  C0            (d,d) covariance of Z over burn-in
  P             (d,k) top-k PCA basis of burn-in Z
  dirC, dirOK   (K,d) landmark centroids per direction + validity mask
  v_mu0, v_cov0 (2,), (2,2) output (velocity) reference moments
"""
from .base import Adapter, REGISTRY, get, list_adapters, describe_all  # noqa: F401
from . import feature  # noqa: F401,E402  (registers feature-stage adapters)
from . import output   # noqa: F401,E402  (registers output-stage adapters)
