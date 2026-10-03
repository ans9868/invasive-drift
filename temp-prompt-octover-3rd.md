ok next is "My lean: (i) — because the staleness instrument genuinely wants per-block context (it's a time series), and the grid wants one session-level context. It's a ctx_scope column and two small loops.” andd "A. Code changes

cache.py
Remove the window loop for the grid → add gfit*mask (first 80 % of online) and geval_mask (last 20 % of online).
Keep the block loop for staleness → rename win*_ → blk\__ (blk_mask, blk_t0, blk_t1) so it's obvious it's staleness-only.
ctx (neural context): currently per-window. → see decision below.
Bump cache_version → 3.
config.json — window_min → block_min (staleness-only); note pool_frac/eval_frac now apply to the online period, not a window.
run_grid.py (new) — no window loop: adapter × decoder × N on the single fit/eval split; adds r2_refit_oracle / r2_refit_out.
staleness.py (new) — the two schemes over blk_mask.
Selftests — update if cache keys change; keep all 4 green.
B. Docs

PLAN.md — rewrite §3 (protocol: no windows), §5 (schema: window_idx → block_idx for staleness; grid rows carry split="grid"), §5c (n_windows → n_blocks).
ideas.md — new Idea 19: "Why we removed windows from the grid" with the reasoning (locality survives; drift is slow → repetition adds little; bigger fit pool & eval; simpler; staleness already covers trajectories).
README.md — progress log 4.
C. Smoke

Re-run smoke (4 selftests + cache --n 1 --with-decoders) and report Block 2c.
D. Then

C → run_grid.py + tracer bullet (Block 4).
D → staleness.py + smoke.
Full cache (53) → full grid → analysis (Block 5/6) → P5/P6/P7.” but stop when finish C
