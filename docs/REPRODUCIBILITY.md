# MedShape SDF Reproducibility

This document describes the anonymous public workflow for reproducing the
MedShape SDF paper experiments. Keep local data, checkpoints, logs, and result
CSVs under ignored directories such as `data/` and `outputs/`.

## Inputs

Required local inputs:

- A public MedShape-style manifest CSV with `class`, `filename`, `shape_id`, and
  `url` columns.
- Public mesh files arranged as class subdirectories, matching the manifest.
- Either released DeepSDF and conditional-energy checkpoints, or enough compute
  to retrain them from the public meshes.

No tracked file should contain workstation paths, user names, private datasets,
or generated result values.

## Recommended Workflow

1. Create the split and preprocess meshes:

   ```bash
   bash scripts/paper/prepare_medshape_sdf.sh \
     data/manifests/medshape_diverse.csv \
     data/raw/medshape_diverse
   ```

2. Train or place the DeepSDF checkpoint:

   ```bash
   bash scripts/paper/train_deepsdf.sh
   ```

3. Train or place the conditional energy checkpoint:

   ```bash
   bash scripts/paper/train_conditional_energy.sh
   ```

4. Tune the L2 baseline on validation only:

   ```bash
   bash scripts/paper/run_l2_validation_grid.sh
   ```

   The default grid is intentionally coarse: `1e-5, 1e-4, 1e-3`.
   The wrapper runs `OBS_LIST="16 32 64 128"` by default and writes
   `outputs/paper/medshape_diverse/l2_val/obs_<N>/sparse_sweep.csv` plus
   `selection.json` and `ranking.csv` for each observation setting. Select the
   best L2 variant using validation `eval_l1` unless the paper states a
   different metric. If selection uses grid SDF/Dice metrics, enable and report
   that selection rule explicitly.

5. Tune the conditional energy weight with the selected L2 value:

   ```bash
   L2_LAMBDA=0.001 \
   L2_VARIANT=l2_lam_1e_3 \
   bash scripts/paper/run_conditional_validation_grid.sh
   ```

   The conditional grid defaults to `1e-5, 3e-5, 1e-4` and evaluates both
   `conditional_energy` and `l2_conditional_energy`. This narrow range keeps the
   validation budget comparable to the L2 grid while covering the useful region
   indicated by existing pilot runs. Use `MAX_PARALLEL=4` (or a smaller value)
   to run observation settings concurrently when the GPU has free capacity.

6. Lock the final config and run test evaluation:

   ```bash
   bash scripts/paper/run_final_eval.sh
   ```

   Final variants should be `no_prior`, tuned `l2`, `conditional_energy`, and
   `l2_conditional_energy`. Final MedShape paper numbers require both validation
   grids above before locking `configs/paper/medshape_diverse_final_eval.yaml`.
   The final wrapper runs `OBS_LIST="16 32 64 128 256"` by default and writes
   `outputs/paper/medshape_diverse/final/obs_<N>/sparse_sweep.csv`.

   The selected L2 and conditional weights can be injected without editing YAML:

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

   The final wrapper intentionally separates the conditional-only weight from
   the L2+conditional weight, because validation can select different useful
   conditional strengths for the two prior families.

   To run the shuffled-context diagnostic, append diagnostic variants without
   changing the default final methods:

   ```bash
   INCLUDE_SHUFFLED_CONTEXT=1 bash scripts/paper/run_final_eval.sh
   ```

   This adds `conditional_energy_shuffled_context` and
   `l2_conditional_energy_shuffled_context`. The sparse observation loss still
   uses the current test shape, but the conditional energy receives sparse
   observations from a different shape in the same split when one is available.
   The output CSV records `context_shape_id` and `context_is_shuffled`; if the
   split contains only one shape, the context falls back to the matching
   observations and `context_is_shuffled` is false. Use the same environment
   variable with `scripts/paper/bootstrap_and_plots.sh` to include these
   diagnostics in the default variant list.

7. Report aggregate metrics, plots, and bootstrap intervals:

   ```bash
   bash scripts/paper/bootstrap_and_plots.sh
   ```

   By default this bootstraps `eval_l1`, `grid_mean_abs_sdf_error`,
   `eval_dice`, and `eval_iou`. L1/grid-error metrics are lower-is-better;
   Dice/IoU/accuracy metrics are called with `--higher-is-better`. Each
   bootstrap file includes the metric name, for example
   `bootstrap_eval_iou_vs_no_prior.json`. The default bootstrap count is 10000.

## Outputs To Report

For the paper table or appendix, report:

- Overall final-step sparse-completion metrics.
- Per-class final-step metrics.
- Step curves for reconstruction quality and latent diagnostics.
- Bootstrap confidence intervals versus `no_prior`.
- Bootstrap confidence intervals for `L2+conditional` versus tuned `L2`.
- The selected L2 and conditional validation variants and selection metrics.
- The observation count for each reported table row or curve.
- The metric direction for each bootstrap result when adding metrics beyond the
  wrapper defaults.

## Grid Metrics Caveat

The final eval config includes grid SDF and binary mask metrics at 250, 500,
1000, 1500, and 2000 optimization steps with `allow_non_watertight_gt: true`.
This is useful for broad exploratory MedShape coverage, but GT Dice/IoU on
non-watertight meshes must be described as exploratory or secondary. For primary
MedShape grid-metric claims, use a documented watertight subset or set
`allow_non_watertight_gt: false` and report which shapes were evaluated.

## Using Released Checkpoints

Place released artifacts under an ignored folder, for example:

```text
outputs/released/
  deepsdf_final.pt
  conditional_energy_prior_best_loss.pt
```

Then run:

```bash
DEEPSDF_CHECKPOINT=outputs/released/deepsdf_final.pt \
CONDITIONAL_ENERGY_CHECKPOINT=outputs/released/conditional_energy_prior_best_loss.pt \
bash scripts/paper/run_final_eval.sh
```

Using released checkpoints skips model training but does not skip validation:
run `scripts/paper/run_l2_validation_grid.sh` against the released DeepSDF
checkpoint before locking final L2 variants. Do not commit copied checkpoints,
generated CSVs, plots, or logs.

## Smoke Runs

For local smoke checks, override epochs and sample counts:

```bash
SDF_SAMPLES=4096 SURFACE_SAMPLES=4096 PREPROCESS_WORKERS=1 \
bash scripts/paper/prepare_medshape_sdf.sh \
  data/manifests/medshape_diverse.csv \
  data/raw/medshape_diverse

DEEPSDF_EPOCHS=2 bash scripts/paper/train_deepsdf.sh
CONDITIONAL_ENERGY_EPOCHS=2 bash scripts/paper/train_conditional_energy.sh
```

Do not use smoke outputs as paper evidence.
