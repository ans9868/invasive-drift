# MVP — forecast next-session decoder failure (see `../drafts/10_mvp_hackathon.md`)

The one question: **do cheap spike-derived features from session N predict session N+1's decoder drop
better than persistence?**

## Layout
```
mvp/
  README.md          # this file (runbook)
  requirements.txt   # python deps
  scripts/
    00_check_env.py      # verify env + imports
    01_list_data.py      # list available sessions (FALCON / Perich)
    02_inspect_nwb.py    # inspect ONE NWB (validate I/O)  [DONE]
    03_decode_health.py  # decoder -> health(t) curve      [TODO, after 02]
    04_forecast.py       # reduced panel vs persistence    [TODO]
  out/               # figures + json (gitignored)
```

Build order: **validate I/O (02 on one real NWB) → decoder (03) → forecast (04).** Do not write the
decoder until 02 confirms the unit/trial/behaviour structure.

## 1. Environment (on a compute node, tmux)
```bash
ssh torch && tmux new -s mvp
# then, inside tmux:
srun -A torch_pr_60_tandon_priority --partition=cpu_short \
     --cpus-per-task=4 --mem=32G --time=04:00:00 --pty bash

# conda env (one-time)
conda create -y -p $SCRATCH/conda_storage/invasive-drift python=3.11
conda activate $SCRATCH/conda_storage/invasive-drift
pip install -r $SCRATCH/invasive-drift/mvp/requirements.txt
python $SCRATCH/invasive-drift/mvp/scripts/00_check_env.py
```

## 2. Data (smallest first)
- **FALCON M1-A (`000941`)** for the first end-to-end sanity (~300 MB, defined split).
- **Perich `000688`** for the longitudinal spine (a one-subject subset).

## 3. Run
```bash
cd $SCRATCH/invasive-drift/mvp
python scripts/01_list_data.py
python scripts/02_decode_health.py     # prints health(t); writes out/health.png
python scripts/03_features_forecast.py # prints panel vs persistence; writes out/forecast.png
```

## 4. Deliverable
One figure (`out/forecast.png`) + one number ("panel AUROC … vs persistence …").
