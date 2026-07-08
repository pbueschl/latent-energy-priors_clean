# Latent Energy Priors for MedShape SDF

Publication-facing reference implementation for the MedShape SDF experiments in
the off-grid workshop paper. The artifact studies post-hoc latent priors for
test-time optimization in latent-conditioned implicit neural representations.

This checkout contains code, configs, and command wrappers only. It does not
contain datasets, checkpoints, logs, generated metrics, private paths, or paper
result files.

## Paper Reproduction Path

The paper path is the MedShape diverse SDF workflow:

```text
public mesh manifest -> class-balanced split -> per-shape SDF samples
  -> DeepSDF autodecoder -> conditional latent energy prior
  -> budgeted validation grids -> locked final sparse-completion evaluation
  -> class aggregates, plots, and bootstrap intervals
```

The public comparison uses four test-time objectives:

```text
no prior:         min_z L_observation(z)
tuned L2:         min_z L_observation(z) + lambda_L2 ||z||_2^2
conditional:      min_z L_observation(z) + lambda_C E_phi(z, observation)
L2+conditional:   min_z L_observation(z) + lambda_L2 ||z||_2^2
                                             + lambda_C E_phi(z, observation)
```

The conditional energy prior is trained after the DeepSDF model from train-split
latents and sparse train-shape observations. The decoder is frozen during
held-out sparse inference.

## Install

The command wrappers under `scripts/` can be run directly from a source checkout,
for example `python3 scripts/train_deepsdf.py --help`. For normal use, install
dependencies in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Mesh preprocessing requires optional geometry dependencies:

```bash
pip install -e ".[preprocess]"
```

Plotting requires:

```bash
pip install -e ".[viz]"
```

Core runtime dependencies are `torch`, `numpy`, `pyyaml`, and `tqdm`.

## Expected Data Layout

Use relative, local-only paths. The default paper configs expect:

```text
data/
  manifests/
    medshape_diverse.csv
  raw/
    medshape_diverse/
      class_a/
        shape_001.stl
      class_b/
        shape_002.stl
  splits/
    medshape_diverse.json
  sdf/
    medshape_diverse/
      shape_001.npz
outputs/
  paper/
    medshape_diverse/
```

The manifest CSV must include `class`, `filename`, `shape_id`, and `url`
columns. Per-shape SDF `.npz` files must contain:

```text
points: float32 array, shape [N, 3]
sdf:    float32 array, shape [N] or [N, 1]
```

All `data/` and `outputs/` paths are ignored by git.

## Paper Configs

Paper-ready YAML files live in `configs/paper/`:

```text
medshape_diverse_deepsdf.yaml
medshape_diverse_conditional_energy.yaml
medshape_diverse_l2_validation_grid.yaml
medshape_diverse_conditional_validation_grid.yaml
medshape_diverse_final_eval.yaml
```

The final eval config intentionally uses relative checkpoint placeholders under
`outputs/paper/medshape_diverse/`. If using released checkpoints, set the wrapper
environment variables documented below instead of editing machine-specific paths
into tracked configs.

The validation and final-evaluation configs store a scalar
`sparse_sweep.observation_points` because the core runner accepts one value per
process. The validation wrappers loop over `OBS_LIST="16 32 64 128"` for
weight selection, while the final paper evaluation loops over
`OBS_LIST="16 32 64 128 256"` and writes one CSV per observation count.

## Command Wrappers

Prepare a split and SDF samples:

```bash
bash scripts/paper/prepare_medshape_sdf.sh \
  data/manifests/medshape_diverse.csv \
  data/raw/medshape_diverse
```

Train the DeepSDF autodecoder:

```bash
bash scripts/paper/train_deepsdf.sh
```

Train the conditional energy prior:

```bash
bash scripts/paper/train_conditional_energy.sh
```

Run the small L2 validation grid:

```bash
bash scripts/paper/run_l2_validation_grid.sh
```

This writes per-observation outputs such as:

```text
outputs/paper/medshape_diverse/l2_val/obs_16/sparse_sweep.csv
outputs/paper/medshape_diverse/l2_val/obs_16/selection.json
```

The L2 grid is intentionally coarse (`1e-5, 1e-4, 1e-3`) so the baseline is
validated without spending most of the tuning budget on the Gaussian prior.
Default selection uses validation `eval_l1`. If selecting by grid SDF metrics or
Dice/IoU, enable the corresponding validation metrics explicitly and report that
selection rule.

Then tune the conditional energy weight with the selected L2 value:

```bash
L2_LAMBDA=0.001 \
L2_VARIANT=l2_lam_1e_3 \
bash scripts/paper/run_conditional_validation_grid.sh
```

This evaluates Conditional-only and L2+Conditional variants over
`CONDITIONAL_GRID`, which defaults to
`1e-5, 3e-5, 1e-4`.
This narrow range follows existing MedShape validation evidence: `1e-3` was too
strong, while useful settings clustered around `1e-5` for conditional-only and
`3e-5` for the L2+conditional final configuration.

Both validation wrappers accept `MAX_PARALLEL`; for example,
`MAX_PARALLEL=4` runs four observation settings concurrently on the selected
device. Use this only when memory and disk I/O can handle multiple sparse
sweeps.

Final MedShape paper numbers require running these validation grids before test
evaluation. After selecting the L2 and conditional values on validation, run
final evaluation:

```bash
bash scripts/paper/run_final_eval.sh
```

You can also inject the selected L2 value without editing YAML. For example:

```bash
OBS_LIST=16 \
L2_LAMBDA=0.001 \
L2_VARIANT=l2_lam_1e_3 \
CONDITIONAL_LAMBDA=0.00001 \
CONDITIONAL_VARIANT=conditional_lam_1e_5 \
L2_CONDITIONAL_LAMBDA=0.00003 \
L2_CONDITIONAL_VARIANT=l2_conditional_l2_1e_3_cond_3e_5 \
bash scripts/paper/run_final_eval.sh
```

`run_final_eval.sh` separates the conditional-only weight from the
L2+conditional weight, because validation can select different useful
conditional strengths for these two prior families.

To run the shuffled-context diagnostic, keep the same observation loss and
replace only the conditional-energy context with sparse observations from a
different shape in the same split:

```bash
INCLUDE_SHUFFLED_CONTEXT=1 bash scripts/paper/run_final_eval.sh
INCLUDE_SHUFFLED_CONTEXT=1 bash scripts/paper/bootstrap_and_plots.sh
```

This appends `conditional_energy_shuffled_context` and
`l2_conditional_energy_shuffled_context` variants. The output CSV records
`context_shape_id` and `context_is_shuffled`, so the diagnostic can be audited
without changing the primary paper defaults.

Final evaluation also writes per-observation outputs:

```text
outputs/paper/medshape_diverse/final/obs_16/sparse_sweep.csv
```

Aggregate, plot, and bootstrap final outputs:

```bash
bash scripts/paper/bootstrap_and_plots.sh
```

By default this reports bootstrap intervals for
`eval_l1 grid_mean_abs_sdf_error eval_dice eval_iou`. Lower-is-better metrics
are bootstrapped as baseline minus variant, while Dice/IoU/accuracy are run with
`--higher-is-better`. Outputs include the metric name, for example
`bootstrap_eval_dice_vs_tuned_l2.csv`. The default bootstrap count is 10000.

Common environment overrides:

```bash
DEVICE=cuda
SDF_ROOT=data/sdf/medshape_diverse
SPLIT_FILE=data/splits/medshape_diverse.json
DEEPSDF_CHECKPOINT=outputs/paper/medshape_diverse/deepsdf/deepsdf_final.pt
CONDITIONAL_ENERGY_CHECKPOINT=outputs/paper/medshape_diverse/conditional_energy/conditional_energy_prior_best_loss.pt
OBS_LIST="16 32 64 128 256"
MAX_PARALLEL=1
OUT_ROOT=outputs/paper/medshape_diverse/final
L2_LAMBDA=0.001
L2_VARIANT=l2_lam_1e_3
CONDITIONAL_LAMBDA=0.00001
CONDITIONAL_VARIANT=conditional_lam_1e_5
L2_CONDITIONAL_LAMBDA=0.00003
L2_CONDITIONAL_VARIANT=l2_conditional_l2_1e_3_cond_3e_5
CONDITIONAL_GRID="1e_5:0.00001 3e_5:0.00003 1e_4:0.0001"
NUM_BOOTSTRAP=10000
BOOTSTRAP_METRICS="eval_l1 grid_mean_abs_sdf_error eval_dice eval_iou"
PLOT_METRICS=eval_l1,grid_mean_abs_sdf_error,eval_dice,eval_iou,latent_norm,conditional_energy
```

Use `DEEPSDF_EPOCHS` or `CONDITIONAL_ENERGY_EPOCHS` only for smoke/debug runs;
paper runs should use the config values unless the paper explicitly reports a
different protocol.

## Existing Checkpoints Versus Full Retraining

For full retraining, run the wrappers in order from preprocessing through final
evaluation. To use released checkpoints, place them under an ignored output
directory and set:

```bash
DEEPSDF_CHECKPOINT=outputs/released/deepsdf_final.pt
CONDITIONAL_ENERGY_CHECKPOINT=outputs/released/conditional_energy_prior_best_loss.pt
```

The final sparse sweep still needs the public SDF root and split file so it can
sample held-out sparse observations and evaluation points.

## Grid Metrics Caveat

`configs/paper/medshape_diverse_final_eval.yaml` enables grid SDF and binary
mask metrics at 250, 500, 1000, 1500, and 2000 optimization steps. The
public-safe default permits non-watertight meshes for GT grid evaluation with
`allow_non_watertight_gt: true`; treat those GT Dice/IoU numbers as exploratory
or secondary unless the paper explicitly states this caveat. For primary
MedShape grid metrics, prefer a documented watertight subset or set
`allow_non_watertight_gt: false` and report the resulting evaluated set.

## Secondary Utilities

Legacy ShapeNet and occupancy workflows are not part of the primary paper
artifact path. Prefer the `configs/paper/` and `scripts/paper/` workflow for
MedShape SDF reproducibility.

## Development Checks

Lightweight checks:

```bash
python3 -m compileall -q lep scripts tests
python3 -m unittest discover -s tests
```

The test files use synthetic temporary data and do not require MedShape data or
private checkpoints. If `pytest` is available, `python3 -m pytest -q` exercises
the same tests more completely because several tests use pytest fixtures.
