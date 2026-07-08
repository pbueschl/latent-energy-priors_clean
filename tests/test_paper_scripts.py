import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PaperScriptTextTest(unittest.TestCase):
    def test_final_eval_wrapper_exposes_tuned_l2_variant_env(self):
        script = (ROOT / "scripts" / "paper" / "run_final_eval.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn('OBS_LIST="${OBS_LIST:-16 32 64 128 256}"', script)
        self.assertIn('L2_LAMBDA="${L2_LAMBDA:-0.001}"', script)
        self.assertIn('L2_VARIANT="${L2_VARIANT:-l2_lam_1e_3}"', script)
        self.assertIn('CONDITIONAL_LAMBDA="${CONDITIONAL_LAMBDA:-0.00001}"', script)
        self.assertIn(
            'CONDITIONAL_VARIANT="${CONDITIONAL_VARIANT:-conditional_lam_1e_5}"',
            script,
        )
        self.assertIn(
            'L2_CONDITIONAL_LAMBDA="${L2_CONDITIONAL_LAMBDA:-0.00003}"',
            script,
        )
        self.assertIn(
            'L2_CONDITIONAL_VARIANT="${L2_CONDITIONAL_VARIANT:-l2_conditional_l2_1e_3_cond_3e_5}"',
            script,
        )
        self.assertIn('INCLUDE_SHUFFLED_CONTEXT="${INCLUDE_SHUFFLED_CONTEXT:-0}"', script)
        self.assertIn(
            'CONDITIONAL_SHUFFLED_VARIANT="${CONDITIONAL_SHUFFLED_VARIANT:-conditional_shuffled_context_lam_1e_5}"',
            script,
        )
        self.assertIn(
            'L2_CONDITIONAL_SHUFFLED_VARIANT="${L2_CONDITIONAL_SHUFFLED_VARIANT:-l2_conditional_shuffled_context_l2_1e_3_cond_3e_5}"',
            script,
        )
        self.assertIn('"name": l2_variant', script)
        self.assertIn('"lambda_l2": l2_lambda', script)
        self.assertIn('"name": l2_conditional_variant', script)
        self.assertIn('"lambda_conditional_energy": l2_conditional_lambda', script)
        self.assertIn('"method": "conditional_energy_shuffled_context"', script)
        self.assertIn('"method": "l2_conditional_energy_shuffled_context"', script)

    def test_validation_wrappers_expose_parallel_and_conditional_grid(self):
        l2_script = (ROOT / "scripts" / "paper" / "run_l2_validation_grid.sh").read_text(
            encoding="utf-8"
        )
        conditional_script = (
            ROOT / "scripts" / "paper" / "run_conditional_validation_grid.sh"
        ).read_text(encoding="utf-8")

        self.assertIn('MAX_PARALLEL="${MAX_PARALLEL:-1}"', l2_script)
        self.assertIn('MAX_PARALLEL="${MAX_PARALLEL:-1}"', conditional_script)
        self.assertIn(
            'CONDITIONAL_GRID="${CONDITIONAL_GRID:-1e_5:0.00001 3e_5:0.00003 1e_4:0.0001}"',
            conditional_script,
        )
        self.assertIn('"method": "conditional_energy"', conditional_script)
        self.assertIn('"method": "l2_conditional_energy"', conditional_script)
        self.assertIn('wait -n', conditional_script)

    def test_bootstrap_wrapper_handles_metric_defaults_and_directions(self):
        script = (ROOT / "scripts" / "paper" / "bootstrap_and_plots.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn(
            'BOOTSTRAP_METRICS="${BOOTSTRAP_METRICS:-eval_l1 grid_mean_abs_sdf_error eval_dice eval_iou}"',
            script,
        )
        self.assertIn(
            'HIGHER_IS_BETTER_METRICS="${HIGHER_IS_BETTER_METRICS:-eval_dice eval_iou eval_accuracy}"',
            script,
        )
        self.assertIn('DIRECTION_ARG=(--higher-is-better)', script)
        self.assertIn('INCLUDE_SHUFFLED_CONTEXT="${INCLUDE_SHUFFLED_CONTEXT:-0}"', script)
        self.assertIn('CONDITIONAL_SHUFFLED_VARIANT', script)
        self.assertIn('L2_CONDITIONAL_SHUFFLED_VARIANT', script)
        self.assertIn('TUNED_L2_VARIANTS="${TUNED_L2_VARIANTS:-${L2_CONDITIONAL_VARIANT}}"', script)
        self.assertIn('bootstrap_${METRIC}_vs_no_prior.json', script)
        self.assertIn('bootstrap_${METRIC}_vs_tuned_l2.csv', script)
        self.assertIn('--metrics "${PLOT_METRICS}"', script)
        self.assertIn('NUM_BOOTSTRAP="${NUM_BOOTSTRAP:-10000}"', script)


if __name__ == "__main__":
    unittest.main()
