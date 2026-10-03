# Finding — MLP confirms the low-rank gate (not a linear artifact)

**Date:** 2026-10-03
**Dataset:** Perich `000688` sub-C, 53 sessions
**Script:** `24_mlp_drift.py` (job 19109650) — `23` re-run with a **nonlinear (MLP)** decoder, head-to-head
with ridge. Feature set identical to `23` (instantaneous state, no lags) so the only change is
linear → nonlinear.

## Results
```
RIDGE (53): PC1=0.55  PC1+2=0.79  rank80=2.5/16 | vol AR1=+0.13 | corr(vol,R2)=-0.27, next=-0.12
MLP   (53): PC1=0.61  PC1+2=0.82  rank80=2.4/16 | vol AR1=+0.09 | corr(vol,R2)=+0.10, next=+0.02
```

## Interpretation
- **The low-rank structure is NOT an artifact of a linear readout.** A nonlinear MLP gives **PC1+2 = 0.82**
  vs ridge **0.79** — essentially identical. ✅ (This was the specific worry the MLP was run to test.)
- **Volatility stays memoryless** in both (lag-1 autocorr **+0.13 / +0.09**) and **still does not lead**
  the R² drop (next-block correlation weaker than contemporaneous in ridge; ≈0 in MLP). ❌ Confirms `23`.
- **Do NOT trust the absolute `rank80 ≈ 2.5`.** It is capped by `nb−1`; and for **well-sampled sessions**
  (nb = 28–31) both decoders give **PC1+2 ≈ 0.5–0.65** → the drift is low-rank but more like **~5–8 dims,
  not 2.** The `PC1+2` number is the solid one.

## Caveats
- **Per-block MLP fitting is unstable:** several sessions are degenerate (R² ≈ −10⁸; some probes give
  PC1 = 1.00). The MLP's *mean R²* (−1e8) is therefore meaningless — only the **low-rank/volatility**
  aggregates (which are robust to a few bad blocks) are usable.
- No capacity control beyond `(64,64), alpha=1e-3, random_state=0, early_stopping`.

## Bottom line
> The **low-rank gate survives a nonlinear decoder**. The drift is genuinely low-dimensional (~5–8 D);
> volatility remains memoryless and non-leading.

## Resources
1907 s, 1.26 GB → 4 G ample.
