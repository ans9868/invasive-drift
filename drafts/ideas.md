# ideas.md — directions worth pursuing

## Idea 1 — Cross-session waveform instrument (STRONG)
**What:** metrics are always **upstream** and **mechanistic**, not the downstream/noisy decoder:
match units across days by **waveform similarity** (works: median corr ≈ 0.996, 63–100% matched),
then track per unit over time:
- **amplitude / SNR over days** → channel health (impedance/gliosis) → **gain**.
- **waveform stability over days** → unit drifting off the electrode → **isolation/loss**.
- **fraction of units lost per session** → **unit loss**.
Use those as the **degradation signal + forecast features** instead of within-session R².
This directly measures the decomposition's mechanisms (unit loss / drift / gain) and unblocks the
cross-session axis (Perich has waveforms, so the "unicorn" dataset isn't needed).
**Status:** waveform matcher implemented (`14_waveform_crosssession.py`), works.

## Idea 2 — Within-session mechanism decomposition (frozen vs refit)
**What:** the within-session decoder decay (R² 0.12→0.05) is real but **not waveform-driven**
(waveforms stable within a session). So *what* decays? Separate:
- **Refit** decoder (trained on recent data) vs **Frozen** decoder (day-1). If **refit stays high while
  frozen decays → representational drift (recalibratable tuning change)**. If **both decay → intrinsic**
  (unit degradation / noise).
- **Tuning-weight drift** (encoder weights vs block 0) → representational drift magnitude.
- **Mean firing rate** trend → gain.
**Status:** building (`15_within_session_mechanism.py`).

## Idea 3 — Waveform-based unit matching as the H3/Track-A instrument
Match units across sessions → "units available to the decoder" becomes a **measured** quantity →
the decomposition's **unit loss** bucket becomes identifiable (death vs isolation vs matcher).
Overlaps Idea 1; downstream of it.

## Idea 4 — Hierarchical / partial pooling across days
Per-session estimates on ~10 blocks are noisy; a hierarchical model (random slopes per session/day)
would give honest cross-session trends (and matches the forecast-note's hierarchical-Bayes panel).

## Idea 5 — Gain vs drift vs loss, measured directly
Within a session: gain = rate scale; drift = tuning-weight change; loss = units dropping. Test which
tracks the decoder decay (Idea 2 does this).
