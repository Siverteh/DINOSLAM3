from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys
from unittest import mock

import torch.nn as nn
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "refocus_vo" / "src"))

from refocus_vo.dino_dpvo.config import DinoDPVOConfig
from refocus_vo.sweeps.run_assoc9_sweep import (
    _baseline_pack_candidates,
    _finalize_existing_run_status,
    _find_live_training_process,
    _load_manifest,
    _materialize_pure_pack_config,
    _mean_metrics_from_eval_csv,
    _required_usable_dev_steps,
    _skip_validation,
    _state_entry_template,
    _state_entry_to_leaderboard_row,
    _top_dev_candidates,
    _train_env,
    _worse_on_both_streak,
)
from refocus_vo.train_dino_dpvo_frontend import _build_optimizer, _lr_scale_for_step, _parse_dpvo_opts


class _DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4))
        self.head = nn.Linear(4, 2)


class Assoc9SweepV1Tests(unittest.TestCase):
    def test_manifest_has_10_runs(self) -> None:
        root = Path(__file__).resolve().parents[1]
        manifest_path = root / "refocus_vo" / "configs" / "dino_dpvo_assoc9_sweep_v1.yaml"
        payload, runs = _load_manifest(manifest_path)
        self.assertEqual(payload["name"], "dino_dpvo_assoc9_sweep_v1")
        self.assertEqual(len(runs), 10)
        self.assertEqual(len(payload["evaluation"]["sequences"]), 9)

    def test_dual_manifest_has_9_runs_and_secondary_eval_sequences(self) -> None:
        root = Path(__file__).resolve().parents[1]
        manifest_path = root / "refocus_vo" / "configs" / "dino_dpvo_paper_lowtex_dual_sweep_v1.yaml"
        payload, runs = _load_manifest(manifest_path)
        self.assertEqual(payload["name"], "dino_dpvo_paper_lowtex_dual_sweep_v1")
        self.assertEqual(len(runs), 9)
        secondary_sequences = payload["sweep"]["config_overrides"]["eval"]["secondary_eval_sequences"]
        self.assertEqual(
            secondary_sequences,
            [
                "freiburg1_desk",
                "freiburg1_plant",
                "freiburg3_large_cabinet",
                "freiburg3_walking_static",
            ],
        )

    def test_top5_repro_manifest_has_5_runs_and_explicit_seed(self) -> None:
        root = Path(__file__).resolve().parents[1]
        manifest_path = root / "refocus_vo" / "configs" / "dino_dpvo_paper_lowtex_top5_repro_sweep_v1.yaml"
        payload, runs = _load_manifest(manifest_path)
        self.assertEqual(payload["name"], "dino_dpvo_paper_lowtex_top5_repro_sweep_v1")
        self.assertEqual(len(runs), 5)
        self.assertEqual(int(payload["runner"]["seed"]), 13)
        self.assertTrue(bool(payload["runner"]["deterministic"]))

    def test_room_noroom_manifest_has_12_runs_and_legacy_repro(self) -> None:
        root = Path(__file__).resolve().parents[1]
        manifest_path = root / "refocus_vo" / "configs" / "dino_dpvo_paper_room_noroom_family_sweep_v1.yaml"
        payload, runs = _load_manifest(manifest_path)
        self.assertEqual(payload["name"], "dino_dpvo_paper_room_noroom_family_sweep_v1")
        self.assertEqual(len(runs), 12)
        self.assertTrue(bool(payload["runner"]["legacy_repro"]))
        self.assertNotIn("seed", payload["runner"])
        self.assertEqual(
            payload["sweep"]["config_overrides"]["eval"]["secondary_from_primary_exclude_sequences"],
            ["freiburg1_room"],
        )
        self.assertEqual(
            payload["evaluation"]["secondary_from_primary_exclude_sequences"],
            ["freiburg1_room"],
        )

    def test_focus071_lr_manifest_has_10_runs_and_patience_2(self) -> None:
        root = Path(__file__).resolve().parents[1]
        manifest_path = root / "refocus_vo" / "configs" / "dino_dpvo_focus071_lr_only_sweep_v1.yaml"
        payload, runs = _load_manifest(manifest_path)
        self.assertEqual(payload["name"], "dino_dpvo_focus071_lr_only_sweep_v1")
        self.assertEqual(len(runs), 10)
        self.assertEqual(int(payload["sweep"]["worse_on_both_patience"]), 2)
        self.assertTrue(bool(payload["runner"]["legacy_repro"]))

    def test_focus071_tumwin_manifest_has_10_runs_and_skip_validation(self) -> None:
        root = Path(__file__).resolve().parents[1]
        manifest_path = root / "refocus_vo" / "configs" / "dino_dpvo_focus071_tumwin_final_sweep_v1.yaml"
        payload, runs = _load_manifest(manifest_path)
        self.assertEqual(payload["name"], "dino_dpvo_focus071_tumwin_final_sweep_v1")
        self.assertEqual(len(runs), 10)
        self.assertEqual(int(payload["sweep"]["worse_on_both_patience"]), 1)
        self.assertTrue(bool(payload["runner"]["legacy_repro"]))
        self.assertTrue(_skip_validation(payload))
        primary_sequences = payload["sweep"]["config_overrides"]["eval"]["primary_eval_sequences"]
        secondary_sequences = payload["sweep"]["config_overrides"]["eval"]["secondary_eval_sequences"]
        self.assertEqual(len(primary_sequences), 16)
        self.assertEqual(len(secondary_sequences), 8)

    def test_focus071_arch5x2_manifest_has_10_runs_and_best_of_mode(self) -> None:
        root = Path(__file__).resolve().parents[1]
        manifest_path = root / "refocus_vo" / "configs" / "dino_dpvo_focus071_arch5x2_tumwin_sweep_v1.yaml"
        payload, runs = _load_manifest(manifest_path)
        self.assertEqual(payload["name"], "dino_dpvo_focus071_arch5x2_tumwin_sweep_v1")
        self.assertEqual(len(runs), 10)
        eval_cfg = payload["sweep"]["config_overrides"]["eval"]
        self.assertEqual(eval_cfg["selection_mode"], "best_of_pure_hybrid")
        self.assertTrue(bool(eval_cfg["save_best_hybrid"]))
        self.assertTrue(bool(eval_cfg["run_hybrid_dev_eval"]))

    def test_all_run_configs_use_9seq_assoc_eval_and_5k_steps(self) -> None:
        root = Path(__file__).resolve().parents[1]
        manifest_path = root / "refocus_vo" / "configs" / "dino_dpvo_assoc9_sweep_v1.yaml"
        payload, runs = _load_manifest(manifest_path)
        expected_sequences = payload["evaluation"]["sequences"]

        for run in runs:
            cfg = yaml.safe_load(run.config_path.read_text(encoding="utf-8"))
            self.assertEqual(cfg["training"]["train_steps"], 5000)
            self.assertEqual(cfg["training"]["eval_every"], 1000)
            self.assertEqual(cfg["eval"]["coverage_gate"], 0.95)
            self.assertEqual(cfg["eval"]["primary_eval_sequences"], expected_sequences)
            self.assertEqual(cfg["eval"]["pure100_eval_sequences"], expected_sequences)

    def test_build_optimizer_separates_backbone_lr(self) -> None:
        model = _DummyModel()
        cfg = DinoDPVOConfig(
            method_id="demo",
            feature_type="DEMO",
            raw={
                "training": {
                    "learning_rate": 2e-5,
                    "dino_backbone_lr": 1e-6,
                    "weight_decay": 1e-6,
                }
            },
        )
        optimizer = _build_optimizer(model, cfg)
        self.assertEqual(len(optimizer.param_groups), 2)
        lrs = sorted(float(group["lr"]) for group in optimizer.param_groups)
        self.assertEqual(lrs, [1e-06, 2e-05])

    def test_parse_dpvo_opts_expands_key_value_pairs(self) -> None:
        self.assertEqual(
            _parse_dpvo_opts("BUFFER_SIZE=512 PATCHES_PER_FRAME=128 REMOVAL_WINDOW=24"),
            ["BUFFER_SIZE", "512", "PATCHES_PER_FRAME", "128", "REMOVAL_WINDOW", "24"],
        )

    def test_materialize_pure_pack_config_applies_pure_overrides(self) -> None:
        root = Path(__file__).resolve().parents[1]
        source_cfg = root / "refocus_vo" / "configs" / "dino_dpvo_assoc9_anchor_90_10_v1.yaml"
        with tempfile.TemporaryDirectory() as td:
            out_cfg = Path(td) / "pack.yaml"
            _materialize_pure_pack_config(source_cfg, out_cfg, "assoc9_test_pack")
            payload = yaml.safe_load(out_cfg.read_text(encoding="utf-8"))
        model_cfg = payload["model"]
        self.assertEqual(model_cfg["native_fraction"], 0.0)
        self.assertEqual(model_cfg["dino_fraction"], 1.0)
        self.assertTrue(model_cfg["enforce_unique_semantic"])
        self.assertEqual(model_cfg["max_semantic_per_cell"], 2)
        self.assertEqual(payload["method_id"], "assoc9_anchor_90_10_v1_assoc9_test_pack")

    def test_state_entry_to_leaderboard_row_surfaces_running_without_dev_rows(self) -> None:
        root = Path(__file__).resolve().parents[1]
        manifest_path = root / "refocus_vo" / "configs" / "dino_dpvo_assoc9_sweep_v1.yaml"
        _, runs = _load_manifest(manifest_path)
        run = runs[0]
        with tempfile.TemporaryDirectory() as td:
            output_dir = Path(td) / "train" / run.run_id
            runtime_cfg = Path(td) / "generated.yaml"
            entry = _state_entry_template(run, output_dir=output_dir, runtime_config_path=runtime_cfg)
            entry["status"] = "running"
            row = _state_entry_to_leaderboard_row(run, entry)
        self.assertEqual(row["status"], "running")
        self.assertEqual(row["best_assoc"], "")
        self.assertEqual(row["last_step"], "")

    def test_finalize_existing_run_status_marks_completed_when_total_steps_reached(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            cfg_path = td_path / "cfg.yaml"
            cfg_path.write_text(
                yaml.safe_dump({"training": {"train_steps": 15000}}, sort_keys=False),
                encoding="utf-8",
            )
            metrics_path = td_path / "train_metrics.csv"
            metrics_path.write_text(
                "step,split,external_mean_ate,external_mean_ate_associated,external_mean_coverage\n"
                "15000,dev,0.1,0.1,1.0\n",
                encoding="utf-8",
            )
            status = _finalize_existing_run_status(output_dir=td_path, runtime_config_path=cfg_path, rows=[])
        self.assertEqual(status, "completed")

    @mock.patch("refocus_vo.sweeps.run_assoc9_sweep.subprocess.run")
    def test_find_live_training_process_prefers_shell_pid(self, mock_run: mock.Mock) -> None:
        output_dir = Path("/tmp/demo_output").resolve()
        mock_run.return_value = mock.Mock(
            stdout=(
                f"111 1 111 bash /home/coder/DINOSLAM3/refocus_vo/scripts/train_dino_dpvo_frontend.sh\n"
                f"222 111 111 /home/coder/DINOSLAM3/refocus_vo/.micromamba/envs/dpvo/bin/python -m refocus_vo.train_dino_dpvo_frontend --output-dir {output_dir}\n"
                f"333 222 111 /home/coder/DINOSLAM3/refocus_vo/.micromamba/envs/dpvo/bin/python -m refocus_vo.train_dino_dpvo_frontend --output-dir {output_dir}\n"
            )
        )
        live = _find_live_training_process(output_dir)
        self.assertIsNotNone(live)
        assert live is not None
        self.assertEqual(live["process_pid"], 111)
        self.assertEqual(live["trainer_pid"], 222)

    def test_baseline_pack_candidates_supports_current_and_extra_baselines(self) -> None:
        evaluation_cfg = {
            "current_champion": {
                "run_id": "broad_best",
                "checkpoint": "refocus_vo/runs/train/dino_dpvo_final_frontend_raw_v1/best.pt",
                "config": "refocus_vo/configs/dino_dpvo_proposals_100_0_fullcover_fixed96_v1.yaml",
                "kind": "current_runtime_winner",
            },
            "baseline_candidates": [
                {
                    "run_id": "paper_best",
                    "checkpoint": "refocus_vo/runs/sweeps/dino_dpvo_freiburg1_paper_sweep_v1_15k/train/assoc9_no_gradient_90_10_v1/best_pure100.pt",
                    "config": "refocus_vo/configs/dino_dpvo_assoc9_no_gradient_90_10_pure_eval_v1.yaml",
                    "kind": "current_paper_best",
                }
            ],
        }
        candidates = _baseline_pack_candidates(evaluation_cfg)
        self.assertEqual([candidate.candidate_id for candidate in candidates], ["broad_best", "paper_best"])
        self.assertEqual(candidates[0].kind, "current_runtime_winner")
        self.assertEqual(candidates[1].kind, "current_paper_best")

    def test_train_env_propagates_seed_and_deterministic(self) -> None:
        root = Path(__file__).resolve().parents[1]
        manifest_path = root / "refocus_vo" / "configs" / "dino_dpvo_paper_lowtex_top5_repro_sweep_v1.yaml"
        payload, runs = _load_manifest(manifest_path)
        with mock.patch.dict("os.environ", {}, clear=True):
            env = _train_env(runs[0], Path("/tmp/demo_top5_output"), payload)
        self.assertEqual(env["SEED"], "13")
        self.assertEqual(env["PYTHONHASHSEED"], "13")
        self.assertEqual(env["DETERMINISTIC"], "1")
        self.assertEqual(env["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")

    def test_train_env_propagates_legacy_repro_without_extra_seed_exports(self) -> None:
        root = Path(__file__).resolve().parents[1]
        manifest_path = root / "refocus_vo" / "configs" / "dino_dpvo_paper_room_noroom_family_sweep_v1.yaml"
        payload, runs = _load_manifest(manifest_path)
        with mock.patch.dict("os.environ", {}, clear=True):
            env = _train_env(runs[0], Path("/tmp/demo_room_noroom_output"), payload)
        self.assertEqual(env["LEGACY_REPRO"], "1")
        self.assertNotIn("DETERMINISTIC", env)
        self.assertNotIn("PYTHONHASHSEED", env)

    def test_required_usable_dev_steps_defaults_and_sorts(self) -> None:
        self.assertEqual(_required_usable_dev_steps({}), [1000])
        self.assertEqual(_required_usable_dev_steps({"required_usable_dev_steps": [1500, 500, 500]}), [500, 1500])

    def test_skip_validation_reads_both_sections(self) -> None:
        self.assertFalse(_skip_validation({}))
        self.assertTrue(_skip_validation({"evaluation": {"skip_validation": True}}))
        self.assertTrue(_skip_validation({"sweep": {"skip_validation": True}}))

    def test_top_dev_candidates_honors_limit(self) -> None:
        rows = [
            {"run_id": "a", "best_assoc": "0.090000", "best_ate": "0.220000", "best_coverage": "0.99"},
            {"run_id": "b", "best_assoc": "0.088000", "best_ate": "0.221000", "best_coverage": "0.99"},
            {"run_id": "c", "best_assoc": "0.087000", "best_ate": "0.219000", "best_coverage": "0.99"},
        ]
        top = _top_dev_candidates(rows, coverage_gate=0.95, limit=2)
        self.assertEqual([row["run_id"] for row in top], ["c", "b"])

    def test_top_dev_candidates_can_apply_lowtex_guardrail(self) -> None:
        rows = [
            {
                "run_id": "paper_only",
                "best_assoc": "0.077000",
                "best_ate": "0.210000",
                "best_coverage": "0.99",
                "best_lowtex_assoc": "0.041000",
                "best_lowtex_coverage": "0.99",
            },
            {
                "run_id": "balanced",
                "best_assoc": "0.078000",
                "best_ate": "0.211000",
                "best_coverage": "0.99",
                "best_lowtex_assoc": "0.035500",
                "best_lowtex_coverage": "0.99",
            },
        ]
        top = _top_dev_candidates(
            rows,
            coverage_gate=0.95,
            secondary_coverage_gate=0.95,
            secondary_assoc_guardrail=0.036,
            limit=2,
        )
        self.assertEqual([row["run_id"] for row in top], ["balanced"])

    def test_mean_metrics_from_eval_csv_can_exclude_sequences(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "metrics_summary.csv"
            path.write_text(
                "sequence,ate_rmse,ate_rmse_associated,coverage\n"
                "freiburg1_360,0.10,0.10,1.0\n"
                "freiburg1_room,0.40,0.40,1.0\n"
                "freiburg1_xyz,0.02,0.02,1.0\n",
                encoding="utf-8",
            )
            all_metrics = _mean_metrics_from_eval_csv(path)
            no_room_metrics = _mean_metrics_from_eval_csv(path, exclude_sequences={"freiburg1_room"})
        self.assertAlmostEqual(all_metrics["assoc"], (0.10 + 0.40 + 0.02) / 3.0)
        self.assertAlmostEqual(no_room_metrics["assoc"], (0.10 + 0.02) / 2.0)

    def test_worse_on_both_streak_counts_consecutive_regressions(self) -> None:
        rows = [
            {
                "step": 500,
                "external_mean_ate_associated": 0.070,
                "external_mean_coverage": 0.99,
                "lowtex_mean_ate_associated": 0.048,
                "lowtex_mean_coverage": 0.99,
            },
            {
                "step": 1000,
                "external_mean_ate_associated": 0.072,
                "external_mean_coverage": 0.99,
                "lowtex_mean_ate_associated": 0.049,
                "lowtex_mean_coverage": 0.99,
            },
            {
                "step": 1500,
                "external_mean_ate_associated": 0.073,
                "external_mean_coverage": 0.99,
                "lowtex_mean_ate_associated": 0.050,
                "lowtex_mean_coverage": 0.99,
            },
        ]
        self.assertEqual(
            _worse_on_both_streak(rows, coverage_gate=0.95, secondary_coverage_gate=0.95),
            2,
        )

    def test_lr_scale_for_step_supports_triangular(self) -> None:
        scale_start = _lr_scale_for_step(
            1,
            6000,
            {"type": "triangular", "start_scale": 0.75, "peak_scale": 1.60, "peak_step": 1500, "end_scale": 0.50},
        )
        scale_peak = _lr_scale_for_step(
            1500,
            6000,
            {"type": "triangular", "start_scale": 0.75, "peak_scale": 1.60, "peak_step": 1500, "end_scale": 0.50},
        )
        scale_end = _lr_scale_for_step(
            6000,
            6000,
            {"type": "triangular", "start_scale": 0.75, "peak_scale": 1.60, "peak_step": 1500, "end_scale": 0.50},
        )
        self.assertAlmostEqual(scale_start, 0.75)
        self.assertAlmostEqual(scale_peak, 1.60, places=2)
        self.assertAlmostEqual(scale_end, 0.50, places=2)


if __name__ == "__main__":
    unittest.main()
