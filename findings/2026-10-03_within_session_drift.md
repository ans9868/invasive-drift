# Finding — Within-session drift is representational (tuning rotation)

**Date:** 2026-10-03
**Dataset:** Perich `000688` sub-C, 53 sessions
**Scripts:** `05_within_session.py`, `06_session_scan.py`, `15_within_session_mechanism.py`,
`16_drift_cause.py` (jobs 19097276, 19097464, 19105812, 19106025)

## Result
1. **Drift is real:** a frozen decoder (trained on the first ~20–30% of a session) decays over the
   session — 6/7 scanned sessions decay; e.g. R² 0.12 → 0.05 over ~1 h.
2. **It is representational (tuning rotation), and it's the main driver:**
   - `corr(decoder-decay, functional-drift) = +0.526`, `corr(decay, weight-drift) = +0.525` (n=53).
   - **weight-drift ≈ functional-drift ⇒ this is REAL functional drift, not a ridge null-space artifact.**
   - Tuning drift explains ~**28%** of the decay variance.
3. **Not the other mechanisms:**
   - **Gain**: firing rate is flat over the session → no gain change.
   - **Unit loss / isolation**: waveforms are stable within a session (`12`) → no loss.
   - **Behavior**: `corr(decay, behavior-drift) = −0.433` (wrong sign) → behavior change is not the driver.
4. **Recalibration**: expanding-window refit improves mean R² only slightly (0.317 → 0.332) but
   **dramatically in some sessions** (e.g. CO-20160914: 0.06 → 0.40).

## Interpretation
> Within-session decoder decay = **the spike→velocity encoding rotating over the session**
> (representational drift). It is functionally verified, and it is **not** gain, **not** unit loss,
> **not** driven by behavior. Partially recoverable by recalibration.

This reproduces, **within a single session and in ephys**, the decomposition's headline that
**representational drift dominates**, with gain and loss absent at this timescale (cf. Gallego 2020 —
stable population dynamics, drifting single-unit tuning).

## Caveats
- ~28% variance explained → the rest is noise / unmodeled factors.
- 2-min blocks: the "refit" comparison is noisy; use larger windows for a clean recalibratability test.
