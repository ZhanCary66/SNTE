# SNTE — Symmetric Neural–speech Temporal Encoding

Code for the paper *"Symmetric Neural–speech Temporal Encoding for Cross-Subject
Match–Mismatch Classification with Non-Invasive Brain Recordings."*

SNTE performs five-way match–mismatch classification: given a 5-second neural
window (EEG or MEG) and five candidate speech segments, it identifies the
segment the participant actually heard. The model applies the same sequence of
dilated convolutions — kernel size 3, dilations (1, 3, 9) — to both modalities
with **separate parameters**, then scores each candidate from multi-statistic
correlation descriptors in the shared latent space.

## Repository layout

| File | Contents |
|---|---|
| `model.py` | SNTE: symmetric dilated encoders and the correlation scorer |
| `baselines/` | The five baselines compared in the paper |
| `dataset.py` | Data pipeline, fixed split and random cross-subject splits |
| `config.py` | Final hyperparameters of SNTE and every baseline |
| `main.py` | Training / evaluation entry point |
| `run_paper.py` | Emits the exact commands behind Tables 1–3 |
| `collect_results.py` | Aggregates result JSONs into the paper's tables |

## Installation

```bash
pip install -r requirements.txt   # torch, numpy
```

Training requires a CUDA GPU; a CPU run is only possible with `--integration`.

## Data

The datasets are not redistributed here; obtain them from their original
sources (see the paper for details). `dataset.py` expects this layout:

```
$SNTE_DATA_ROOT/
├── neural_lp30_64hz/<dataset>/sub-<id>_<stimulus>_LP-30_64Hz.npy
└── stimuli/<dataset>/
    ├── mel10_64Hz/<stimulus>.npy                   # [T, 10]
    └── wav2vec_l14_pca64_64Hz/<stimulus>.npy       # [T, 64]
```

with `<dataset>` one of `SparKULee`, `PKUEEG`, `SEM4Lang`. Point
`SNTE_DATA_ROOT` at it (defaults to `./data/ICASSP_shared_v1`), then check the
layout:

```bash
export SNTE_DATA_ROOT=/path/to/ICASSP_shared_v1
python main.py --dataset SparKULee --check-data
```

The 74-dimensional speech input is the concatenation of the PCA-reduced
wav2vec 2.0 layer-14 features (64) and a Mel spectrum (10), both at 64 Hz.

## Quick start

```bash
# smoke test: tiny subset, CPU, no GPU needed
python main.py --model snte --dataset SparKULee --integration --device cpu

# one real run (fixed split, training seed 0)
python main.py --model snte --dataset SparKULee --seed 0

# model-only forward check
python model.py     # prints the output shape and parameter count
```

## Reproducing the paper

`run_paper.py` knows the exact configuration of every reported run.

| Table | Runs | Protocol |
|---|---|---|
| 1 — main results | 6 models × 3 datasets × 5 splits | 5 random cross-subject splits, training seed 2 |
| 2 — match head | 3 heads × 3 datasets × 5 splits | same 5 splits |
| 3 — component ablation | 6 variants × 3 datasets × 3 seeds | fixed split |

```bash
python run_paper.py --table all --emit sh          # print every command
python run_paper.py --table 3 --emit slurm         # write slurm/ scripts
python collect_results.py                          # print the three tables
```

189 runs in total, roughly 30 GPU-hours on an A800. For a single-GPU machine,
run the emitted commands sequentially; each writes
`results/<table>/<label>/<dataset>_seed<seed>.json`.

> The main-results row for SNTE and the `perstat` row of the match-head table
> are the *same* configuration (`snte_split<N>` and `perstat_split<N>`), so one
> set of runs covers both; the original campaign ran it once.

### A note on the splits

The "5 random cross-subject splits" (split seeds 101–105) are **independent
random re-partitions**, not the disjoint folds of a k-fold cross-validation:
each seed shuffles the participants and re-applies the same per-split subject
counts as the fixed split, so the test sets of two seeds overlap and the same
participant may be tested under several seeds. Keep this in mind when pooling
per-subject results across seeds — for the paper's statistics, repeated
observations of one participant were collapsed to a single value first.

## Results

Test subject-macro accuracy (%), chance level 20%, mean ± std over the 5 random
cross-subject splits:

| Model | SparrKULee | PKUEEG | SEM4Lang |
|---|---:|---:|---:|
| **SNTE** | **71.43 ± 3.78** | **56.58 ± 7.23** | **79.93 ± 5.75** |
| Eeg2Vec | 68.44 ± 3.68 | 54.76 ± 7.83 | 73.86 ± 5.28 |
| CCA | 66.42 ± 3.29 | 52.31 ± 5.24 | 71.00 ± 5.57 |
| ConvConcatNet | 66.02 ± 3.93 | 52.97 ± 6.39 | 70.32 ± 5.26 |
| VLAAI | 65.71 ± 3.41 | 54.09 ± 6.93 | 73.69 ± 6.08 |
| BrainMagic | 60.19 ± 6.37 | 52.89 ± 6.04 | 74.86 ± 7.04 |

SNTE ranks first on all 15 split–dataset combinations and has 1.61M parameters
(SparrKULee), against 4.48M for ConvConcatNet and 6.43M for BrainMagic.

## Configuration

`config.py` holds the final hyperparameters: SNTE uses `embed_dim=256`,
`dropout=0.5`, and AdamW with learning rate `1e-3`, weight decay `1e-2`,
50 epochs, batch size 64 and early-stopping patience 10. The baselines are
tuned toward their natural operating point under the same protocol, so the
comparison isolates the match head and encoder.

The architecture switches in `main.py` (`--head`, `--no-standardize`,
`--neural-encoder`, `--speech-encoder`, `--tied-encoder`,
`--neural-dilations`, `--speech-dilations`) reproduce the paper's ablations.
Their defaults are the reported model.

One implementation detail worth flagging: the scorer still builds its
learned temperature and the per-lag attention layer even though the paper's
configuration uses a single scale and a single lag. This is deliberate — it
keeps the module creation order, and therefore the parameter count (1.61M) and
the seeded initialization, identical to the code that produced the reported
numbers, so a rerun with the same seed reproduces them.

## Citation

```bibtex
@inproceedings{snte2027,
  title     = {Symmetric Neural--speech Temporal Encoding for Cross-Subject
               Match--Mismatch Classification with Non-Invasive Brain Recordings},
  author    = {Zhang, Zifeng and Xu, Xiran and Yan, Yujie and Li, Songyi and
               Zheng, Linze and Liang, Jinghua and Xiao, Boda and Dong, Mochu and Chen, Jing},
  booktitle = {Proc. IEEE ICASSP},
  year      = {2027}
}
```
