# SNTE — Symmetric Neural–speech Temporal Encoding

Code for *Symmetric Neural–speech Temporal Encoding for Cross-Subject
Match–Mismatch Classification with Non-Invasive Brain Recordings*.

Given a five-second EEG or MEG window and five speech candidates, SNTE outputs
five matching scores. Neural and speech encoders use the same convolutional
structure with separate parameters. The scorer combines time-mean products,
time-maximum products and time-mean absolute differences of normalized features.

## Installation

Use Python 3.10 or newer:

```bash
pip install -r requirements.txt
```

Formal training requires a CUDA device supporting bfloat16. CPU execution is
available for the small `--integration` check. Dependency constraints are
compatible lower bounds, not a record of the original experiment environment.

## Input data

Obtain data from the original sources cited in the paper; PKUEEG is available at
[OpenNeuro v1.0.3](https://openneuro.org/datasets/ds008834/versions/1.0.3).

```text
$SNTE_DATA_ROOT/
├── neural_lp30_64hz/<dataset>/sub-<id>_<stimulus>_LP-30_64Hz.npy
└── stimuli/<dataset>/
    ├── wav2vec_l14_pca64_64Hz/<stimulus>.npy   # [T, 64]
    └── mel10_64Hz/<stimulus>.npy              # [T, 10]
```

Neural arrays have 64, 57 or 306 channels, respectively; time-by-channel and
channel-by-time layouts are accepted. SEM4Lang uses the 204 planar-gradiometer
channels. Both modalities must already be time-aligned and sampled at 64 Hz.
Speech inputs concatenate wav2vec-L14-PCA64 before Mel10, after separate
stimulus-wise z-scores. Neural inputs receive recording-wise channel z-scores
and, where enabled by the model, window-wise z-scores.

Only complete, non-overlapping 320-sample windows within both recordings are
used. Each positive speech window is paired with four other windows from the
same stimulus. If fewer than four distinct negative windows exist, sampling is
with replacement. Candidate plans use fixed split-specific seeds.

```bash
export SNTE_DATA_ROOT=/path/to/ICASSP_shared_v1
python main.py --dataset SparKULee --check-data
```

The layout check covers the selected dataset's file counts, identities and
stimulus correspondence. Array shape and finite-value checks also run during
loading; these checks cannot establish the correctness of upstream alignment.

## Training and evaluation

```bash
# One-epoch pipeline check on a small subset of the supplied input data
python main.py --model snte --dataset SparKULee --integration --device cpu --workers 0

# Fixed cross-subject split, training seed 0
python main.py --model snte --dataset SparKULee --seed 0

# One random subject repartition
python main.py --model snte --dataset SparKULee --seed 2 --split-seed 101
```

`config.py` contains model-specific dimensions, dropout and training settings.
`main.py` selects the checkpoint by validation subject-macro accuracy, breaking
ties by validation loss, and then evaluates the selected checkpoint on test.
Results contain metrics, effective configuration and ordered subject mappings;
checkpoints use a separate `.pt` file. Existing outputs are not overwritten.
Use `--data-root` and `--output /path/to/run.json` to select explicit locations.
Integration results are marked separately and excluded from paper aggregation.

## Paper experiments

| Entry | Protocol |
|---|---|
| Table 1 | Six models, three datasets, repartition seeds 101–105; training seed 2 |
| Table 2 | Three SNTE scoring heads under the same five repartitions |
| Table 3 | Six components/variants on the fixed split; training seeds 0–2 |
| Figure 2 | The nine Table 3/full results, with 17/5/2 test subjects |

```bash
# Print commands; this does not run or submit jobs
python run_paper.py --table all --emit sh --data-root "$SNTE_DATA_ROOT" --output-root /path/to/results

# Alternatively generate Slurm scripts; select a partition for your cluster
python run_paper.py --table 3 --emit slurm --partition YOUR_PARTITION --data-root "$SNTE_DATA_ROOT" --output-root /path/to/results

python collect_results.py --results /path/to/results
python analyze_results.py --results /path/to/results
python plot_fig2.py --results /path/to/results --output /path/to/figure2.png --csv /path/to/figure2.csv
```

There are 174 unique training jobs covering 189 table references: Table 2/perstat
reuses Table 1/SNTE. The loaders also accept the historical Table 2/perstat path;
if both copies exist, their scientific metadata and results must agree.
`--table` and `--datasets` select subsets for command generation, collection and
analysis. Collection and analysis reject missing, invalid or integration runs.
`collect_results.py --partial` reports coverage only, not an incomplete table.

The five repartitions are not disjoint k-folds; the same subject can be tested
more than once. Table 1's nominal Wilcoxon comparisons use subject–repartition
observations. Head comparisons average repeated differences per subject before
pooling; fixed-split component comparisons average over training seeds per
subject. The analyzer labels these conventions separately. Averaging repeated
observations does not remove all dependence from shared training or overlapping
partitions. Figure 2 uses subject three-seed means, IQR/median boxes, min–max
whiskers of those means, and each subject's seed range.

### Older result files

New results include `split_unit_names` and `candidate_seeds`. Legacy results
without identity or architecture metadata need an explicit `--subject-map` JSON
for collection, analysis and plotting. Its keys are paths relative to the results
root, and each entry supplies the missing known metadata:

```json
{
  "table3/full/PKUEEG_seed0.json": {
    "split_unit_names": {"train": ["..."], "val": ["..."], "test": ["..."]},
    "architecture": {"...": "actual recorded architecture fields"}
  }
}
```

Replace the placeholders with complete, lexicographically ordered actual IDs
and the architecture fields defined in `results_io.py`. Conflicts with recorded
identities or SNTE settings are rejected. The known schema-0 default baseline
architecture bookkeeping error may be corrected only to the executed baseline
architecture, with a nonempty `provenance` string explaining the source.
Overlays do not modify files or authorize replacing differing hyperparameters.
No raw data or checkpoint is required to analyze self-describing results.

## Implementation scope

The baselines are adaptations to this five-way task, using a shared time-mean
cosine matcher and model-specific dimensions and regularization. CCA denotes a
33-tap linear neural encoder with a learned speech encoder, not a closed-form
CCA solver. Window normalization is enabled in the four other adapted baselines.
The three heads and six component variants are defined in `run_paper.py`.

Validation-based checkpoint selection describes a single training run. Historical
training-seed and some baseline-setting choices considered fixed-split test
performance and relative performance margins; this is not a validation-only
hyperparameter-selection protocol. Release settings should be checked against
original run metadata when verifying historical numbers: fixed-split
ConvConcatNet logs differ from the released dimension/dropout configuration,
and the final repartition configuration is not established by those logs.

`config.py` defines SNTE with `embed_dim=256`, `dropout=0.5`, and AdamW with
learning rate `1e-3`, weight decay `1e-2`, up to 50 epochs, batch size 64 and
early-stopping patience 10.

Seeds are recorded, but cuDNN benchmark mode and nondeterministic algorithms are
enabled. Identical numerical results across environments are not guaranteed.
The tests below use synthetic inputs and do not verify the paper's accuracies.

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests cover model/data behavior, invalid inputs, output protection, result
identity and completeness, statistical pairing, experiment reuse and Figure 2.
The before-change fixture is version-bound; incompatible numerical checks are
explicitly skipped rather than treated as passed.

## Citation

```bibtex
@inproceedings{snte2027,
  title     = {Symmetric Neural--speech Temporal Encoding for Cross-Subject
               Match--Mismatch Classification with Non-Invasive Brain Recordings},
  author    = {Zhang, Zifeng and Xu, Xiran and Yan, Yujie and Li, Songyi and
               Zheng, Linze and Liang, Jinghua and Xiao, Boda and Dong, Mochu and
               Chen, Jing},
  booktitle = {Proc. IEEE ICASSP},
  year      = {2027}
}
```
