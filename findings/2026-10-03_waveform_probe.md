# Finding — Waveform probe: stable within-session, works cross-session

**Date:** 2026-10-03
**Scripts:** `12_waveform_degradation.py`, `14_waveform_crosssession.py` (jobs 19105071, 19105197)

## Result
### Within a session (CO-20150309, 74 units, 57.7 min)
- waveform **stability ≈ 0.997–0.999** (corr of block mean vs global mean),
- **waveform drift (1−corr 1st vs 2nd half): median 0.000, max 0.016**,
- alive units 72–74 (stable); amplitude **417 → 388 µV (−7%)**.
- **⇒ unit waveforms barely change within a session** → the within-session decoder drift is **not**
  unit-loss / isolation; it is representational/gain (see the within-session finding). The −7%
  amplitude is a candidate gain/SNR effect.

### Across sessions (sub-C, 2013–2016)
- Per-unit waveform **matching by correlation works**: median matched correlation ≈ **0.996**;
  **63–100% of units match** between consecutive sessions.
- ⇒ **Perich waveforms unlock cross-session unit identity** — the Track-A/H3 "unit matching" blocker,
  without needing new data.

## Interpretation
Waveforms are the **wrong probe for within-session drift** but the **right probe for cross-session**
mechanisms: unit death / appearance / identity, and per-unit amplitude/stability over days, i.e. the
decomposition's **unit loss** and **gain** mechanisms.

## Status / caveat
- Within-session tooling done (`12`); cross-session matcher prototype done (`14`), gives per-consecutive-pair
  matching — a full tracked-chain (union-find) + per-unit amplitude trends over days is the next step.
