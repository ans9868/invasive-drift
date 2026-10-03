# invasive-drift

**Forecast — then decompose — cross-session decoder degradation in chronic invasive neural interfaces.**

The spine (the question): given a **chronic same-probe** recording (same subject, same physical probe
*not* removed), can cheap features from session *N* predict session *N+1*'s decoder drop **better than
persistence** — and can that drop be split into **unit loss / representational drift / gain shift**?

## Where things are

- `drafts/` — all thinking notes (`00`–`12`). Start with `drafts/README.md` (index) and
  `drafts/10_mvp_hackathon.md` (the MVP).
- `torch-slurm-priority-and-partitions.md` — how to get scheduled on NYU Torch (`_priority` account,
  `--partition=cpu_short`).
- `torch-jupyter-tmux-skill.md` — the notebook/tunnel runbook (tmux + `$SCRATCH`).

## Within-session work — the current result

- **`final-writeup.md`** — ⭐ the paper-style write-up: ***unsupervised within-session adaptation does not
  correct neural drift*** (12 adapters × 5 decoders × 53 sessions, 12,720 cells). **Start here.**
  Headline: P1 fails (the moment ladder is a *step, plateau, cliff*), P2 confirmed (`subspace` alignment
  costs **−0.171 R²**), and **P3 refuted** — the per-unit reference correspondence is a **placebo** (a
  shuffled control matches it, p=6.5e-07); the only clear win is **supervised**.
- `grid_within_session_27/` — the experiment. `PLAN.md` is authoritative; `README.md` holds progress logs
  1–8 (7 = the std-floor bug and fix, 8 = the re-run + verification).
- `adapters/` — the adapter library (moment-order ladder, direction-only family, the shuffled control).
- `findings/` — dated finding notes; **`findings/README.md` is the index.** For this work see
  `2026-10-03_adapter_grid_framing.md` (the pre-registered P1–P5, written *before* the run) and
  `2026-10-03_adapter_grid_results.md` (the outcome), plus `2026-10-03_prior_art_falcon_nomad.md`
  (FALCON + NoMAD extraction).
- `ai-context/` — session handoffs for whoever picks this up next.

## Key decisions (see `drafts/`)

- **Scope = cross-session** (chronic same-probe), *not* cross-insertion (`drafts/08`).
- **Primary dataset = Perich & Miller DANDI `000688`**; **Step-1 baseline = reproduce FALCON**
  (`drafts/06`).
- **Reduced panel** (spikes-only): rate, unit count, match-survival, factor-angle, noise corr, decode
  entropy (`drafts/06` Part 3).
- **The test = does the panel beat persistence?** (`drafts/04`, `drafts/09`).
- **Prior art:** unoccupied; **MINDFUL** (`10.1038/s42003-024-06784-4`) is the bar (`drafts/11`).

## On Torch (HPC)

Repo is cloned at `$SCRATCH/invasive-drift`:

```bash
ssh torch
cd $SCRATCH/invasive-drift && git pull --ff-only   # sync source (push local -> pull here)
```

Interactive compute node (see `torch-slurm-priority-and-partitions.md`):

```bash
srun -A torch_pr_60_tandon_priority --partition=cpu_short \
     --cpus-per-task=4 --mem=32G --time=04:00:00 --pty bash
```

## Sync workflow

Local repo (`ans9868/invasive-drift`, public) is the source of truth; Torch pulls
(`git push` locally → `git pull` on Torch).
