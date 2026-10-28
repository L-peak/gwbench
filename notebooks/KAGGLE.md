# Running gwbench on Kaggle

The local pipeline is CPU-only. Use Kaggle's free GPU for the global GRU
experiments.

## 1. Create a private Kaggle Dataset

A ready-to-upload archive is at `dist/gwbench-data.zip` (5.6 MB). It contains:

```
gwbench/                       # package
scripts/                       # pipeline scripts
requirements.txt
data/processed/panel_core_clean.parquet  # 74 wells, clean core
data/processed/panel_core.parquet        # 89 wells
data/processed/panel_full.parquet        # 279 wells
```

Create a private dataset named `gwbench-data` and upload the zip (or the
extracted folder). Kaggle may keep the zip as a single file; the bootstrap
cell below unzips it.

## 2. Create a Kaggle Notebook

- Add the `gwbench-data` dataset as input.
- Set **Accelerator** to GPU (T4 ×2 or P100).
- Enable **Internet** only if you need to install packages; torch is already
  available in Kaggle images.
- Set the session to run in the background / "Save & Run All" for long jobs.

## 3. Bootstrap the code

In the first cell, either unzip the archive:

```python
!unzip -q /kaggle/input/gwbench-data/gwbench-data.zip -d /kaggle/working/
!cd /kaggle/working && python -c "import gwbench; print(gwbench.__version__)"
```

or, if Kaggle already extracted the folder, copy it:

```python
!cp -r /kaggle/input/gwbench-data/gwbench /kaggle/working/
!cp -r /kaggle/input/gwbench-data/scripts /kaggle/working/
!cp /kaggle/input/gwbench-data/requirements.txt /kaggle/working/
!cd /kaggle/working && python -c "import gwbench; print(gwbench.__version__)"
```

## 4. Train the 30-day model

```python
!cd /kaggle/working && python scripts/07_train_gru.py \
  --panel /kaggle/input/gwbench-data/panel_core_clean.parquet \
  --split-mode temporal \
  --lookback 365 --horizon 30 \
  --stride-train 7 --stride-eval 1 \
  --epochs 40 --batch-size 256 --lr 1e-3 \
  --hidden 64 --layers 2 --dropout 0.2 \
  --num-workers 2 \
  --out-dir /kaggle/working/gru_h30
```

The script writes:

- `best_model.pt`, `last_model.pt`
- `config.json`
- `test_predictions.parquet`
- `test_metrics_per_site.csv`
- `test_metrics.json`

Download `/kaggle/working/gru_h30` after the run.

## 5. Experiment matrix

Run these in order; each model is independent:

| Experiment | Horizon | Split | Lookback | Notes |
|---|---:|---|---:|---|
| `gru_h30_temporal` | 30 | temporal | 365 | headline result |
| `gru_h90_temporal` | 90 | temporal | 365 | harder, persistence is near zero |
| `gru_h30_site` | 30 | site | 365 | add `--no-site-embedding`; treat as a lower bound |
| `gru_h30_short` | 30 | temporal | 180 | lookback ablation |
| `gru_h30_seed1/2` | 30 | temporal | 365 | seed variance |

For site split, the script stratifies by `state_cd` and holds out whole wells.
Pass `--no-site-embedding` for leave-site-out runs: an untrained embedding for
an unseen well is random noise, so the embedded model is not a fair transfer
test.

## 6. Time budget

On a T4, the 74-well core panel with lookback 365 and 40 epochs is roughly
1-2 hours per run. Kaggle gives 30 GPU hours per week, so the matrix above fits
comfortably. Keep each session under 12 hours; `best_model.pt` and
`last_model.pt` are written during training.

## 7. What to compare

`test_metrics.json` contains the model and two baselines on exactly the same
target dates:

- `y_pred_*` — global GRU
- `persistence_*` — last observation before the anchor
- `seasonal_naive_*` — value 365 days before the target

The headline question: does the GRU beat persistence at 30 and 90 days, and
does the gain survive a leave-site-out split?
