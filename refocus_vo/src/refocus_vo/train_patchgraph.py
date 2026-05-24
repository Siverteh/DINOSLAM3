from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from refocus_vo.data import TartanAirWindowDataset
from refocus_vo.patchgraph import DinoPatchGraphVO, DinoPatchTeacher, compute_patchgraph_losses, load_patchgraph_config


def _make_loader(dataset: TartanAirWindowDataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,
    )


def _build_model(cfg) -> DinoPatchGraphVO:
    model_cfg = cfg.model
    return DinoPatchGraphVO(
        dino_name_or_path=str(model_cfg.get("dino_name_or_path")),
        dino_layers=tuple(int(v) for v in model_cfg.get("dino_layers", [6, 11])),
        dino_dtype=str(model_cfg.get("dino_dtype", "bf16")),
        patch_size=int(model_cfg.get("patch_size", 16)),
        num_patches=int(model_cfg.get("num_patches", 64)),
        max_nodes_per_object_ratio=float(model_cfg.get("max_nodes_per_object_ratio", 0.20)),
        k_mutual_neighbors=int(model_cfg.get("k_mutual_neighbors", 4)),
        dino_hidden_dim=int(model_cfg.get("dino_hidden_dim", 192)),
        local_patch_dim=int(model_cfg.get("local_patch_dim", 64)),
        edge_dim=int(model_cfg.get("edge_dim", 192)),
        graph_hidden_dim=int(model_cfg.get("graph_hidden_dim", 256)),
        min_match_cosine=float(model_cfg.get("min_match_cosine", 0.40)),
        max_history=int(model_cfg.get("max_history", 3)),
        lag_embedding_dim=int(model_cfg.get("lag_embedding_dim", 16)),
        enable_offset_refinement=bool(model_cfg.get("enable_offset_refinement", False)),
        use_multiframe_graph=bool(model_cfg.get("use_multiframe_graph", False)),
        device=str(cfg.training.get("device", "cuda")),
    ).to(torch.device(str(cfg.training.get("device", "cuda"))))


def load_patchgraph_state_dict(model: DinoPatchGraphVO, state_dict: dict[str, torch.Tensor]) -> tuple[list[str], list[str]]:
    compat_state: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("pose_head."):
            compat_state["pair_pose_head." + key[len("pose_head."):]] = value
        else:
            compat_state[key] = value
    missing, unexpected = model.load_state_dict(compat_state, strict=False)
    return list(missing), list(unexpected)


def _append_metrics_row(path: Path, step: int, split: str, metrics: dict[str, float]) -> None:
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                int(step),
                split,
                f"{float(metrics.get('loss', math.nan)):.6f}",
                f"{float(metrics.get('selector_bce', math.nan)):.6f}",
                f"{float(metrics.get('persistence_l1', math.nan)):.6f}",
                f"{float(metrics.get('edge_valid_bce', math.nan)):.6f}",
                f"{float(metrics.get('confidence_bce', math.nan)):.6f}",
                f"{float(metrics.get('offset_l1', math.nan)):.6f}",
                f"{float(metrics.get('rotation_l1', math.nan)):.6f}",
                f"{float(metrics.get('translation_dir_l1', math.nan)):.6f}",
                f"{float(metrics.get('translation_scale_l1', math.nan)):.6f}",
                f"{float(metrics.get('coverage_reg', math.nan)):.6f}",
                f"{float(metrics.get('semantic_consistency', math.nan)):.6f}",
                f"{float(metrics.get('mean_selected_teacher_score', math.nan)):.6f}",
                f"{float(metrics.get('matches_per_pair', math.nan)):.6f}",
            ]
        )


@torch.no_grad()
def _evaluate_dev(
    model: DinoPatchGraphVO,
    teacher: DinoPatchTeacher,
    loader: DataLoader,
    loss_weights: dict[str, float],
) -> dict[str, float]:
    model.eval()
    agg: dict[str, float] = {}
    count = 0
    for batch in loader:
        pred = model(batch["images"])
        _, metrics, _ = compute_patchgraph_losses(pred, batch, teacher, loss_weights)
        for key, value in metrics.items():
            agg[key] = agg.get(key, 0.0) + float(value)
        count += 1
        if count >= 16:
            break
    if count == 0:
        return {"loss": math.nan}
    return {k: v / count for k, v in agg.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the DINO patch-graph VO model on TartanAir.")
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--subset-config", default=None)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-train-windows", type=int, default=None)
    ap.add_argument("--max-dev-windows", type=int, default=None)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg_path = Path(args.config).expanduser().resolve() if args.config else (
        Path(__file__).resolve().parents[2] / "configs" / "dino_patchgraph_v1.yaml"
    )
    subset_cfg_path = Path(args.subset_config).expanduser().resolve() if args.subset_config else (
        Path(__file__).resolve().parents[2] / "configs" / "tartanair_subset_v1.yaml"
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = output_dir / "train_metrics.csv"

    cfg = load_patchgraph_config(cfg_path)
    import yaml

    subset_cfg = yaml.safe_load(subset_cfg_path.read_text(encoding="utf-8")) or {}
    subset_block = subset_cfg.get("subset", {})
    window_block = subset_cfg.get("windowing", {})

    device = str(args.device or cfg.training.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    cfg.raw.setdefault("training", {})["device"] = device

    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    image_size = tuple(int(v) for v in cfg.model.get("image_size", [240, 320]))
    train_ds = TartanAirWindowDataset(
        args.dataset_root,
        split="train",
        subset_environments=tuple(subset_block.get("environments", [])),
        difficulties=tuple(subset_block.get("difficulties", ["Easy", "Hard"])),
        max_trajectories_per_env_difficulty=int(subset_block.get("max_trajectories_per_env_difficulty", 1)),
        n_frames=int(window_block.get("n_frames", 4)),
        image_size=image_size,
        max_windows=int(args.max_train_windows or window_block.get("train_windows", 80000)),
        dev_ratio=float(window_block.get("dev_windows", 8000)) / max(
            1.0,
            float(window_block.get("train_windows", 80000)) + float(window_block.get("dev_windows", 8000)),
        ),
        seed=int(args.seed),
    )
    dev_ds = TartanAirWindowDataset(
        args.dataset_root,
        split="dev",
        subset_environments=tuple(subset_block.get("environments", [])),
        difficulties=tuple(subset_block.get("difficulties", ["Easy", "Hard"])),
        max_trajectories_per_env_difficulty=int(subset_block.get("max_trajectories_per_env_difficulty", 1)),
        n_frames=int(window_block.get("n_frames", 4)),
        image_size=image_size,
        max_windows=int(args.max_dev_windows or (window_block.get("train_windows", 80000) + window_block.get("dev_windows", 8000))),
        dev_ratio=float(window_block.get("dev_windows", 8000)) / max(
            1.0,
            float(window_block.get("train_windows", 80000)) + float(window_block.get("dev_windows", 8000)),
        ),
        seed=int(args.seed),
    )

    batch_size = int(cfg.training.get("batch_size", 4))
    num_workers = int(cfg.training.get("num_workers", 4))
    train_loader = _make_loader(train_ds, batch_size, num_workers, True)
    dev_loader = _make_loader(dev_ds, batch_size, max(0, num_workers // 2), False)

    model = _build_model(cfg)
    teacher = DinoPatchTeacher(
        patch_size=int(cfg.model.get("patch_size", 16)),
        num_patches=int(cfg.model.get("num_patches", 64)),
        max_nodes_per_object_ratio=float(cfg.model.get("max_nodes_per_object_ratio", 0.20)),
        k_mutual_neighbors=int(cfg.model.get("k_mutual_neighbors", 4)),
        teacher_weights=dict(cfg.teacher),
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg.training.get("learning_rate", 2e-4)),
        weight_decay=float(cfg.training.get("weight_decay", 1e-6)),
    )

    with metrics_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "step",
                "split",
                "loss",
                "selector_bce",
                "persistence_l1",
                "edge_valid_bce",
                "confidence_bce",
                "offset_l1",
                "rotation_l1",
                "translation_dir_l1",
                "translation_scale_l1",
                "coverage_reg",
                "semantic_consistency",
                "mean_selected_teacher_score",
                "matches_per_pair",
            ]
        )

    best_dev = float("inf")
    total_steps = int(cfg.training.get("train_steps_stage1", 10000)) + int(cfg.training.get("train_steps_stage2", 90000))
    stage1_steps = int(cfg.training.get("train_steps_stage1", 10000))
    eval_every = int(cfg.training.get("eval_every", 1000))
    grad_clip = float(cfg.training.get("grad_clip", 5.0))

    step = 0
    train_iter = iter(train_loader)
    while step < total_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        model.train()
        pred = model(
            batch["images"],
            teacher_scores=None,
            teacher=teacher,
            depths=batch["depths"],
            poses=batch["poses"],
            intrinsics=batch["intrinsics"],
            use_teacher_for_selection=bool(step < stage1_steps),
        )
        loss, metrics, teacher_scores = compute_patchgraph_losses(pred, batch, teacher, dict(cfg.losses))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        _append_metrics_row(metrics_csv, step + 1, "train", metrics)
        if (step + 1) % int(cfg.training.get("log_every", 50)) == 0:
            print(
                f"[step {step + 1}/{total_steps}] "
                f"loss={metrics['loss']:.4f} selector={metrics['selector_bce']:.4f} "
                f"pose_r={metrics['rotation_l1']:.4f} pose_t={metrics['translation_dir_l1']:.4f} "
                f"matches={metrics['matches_per_pair']:.2f}"
            )

        if (step + 1) % eval_every == 0 or (step + 1) == total_steps:
            dev_metrics = _evaluate_dev(model, teacher, dev_loader, dict(cfg.losses))
            _append_metrics_row(metrics_csv, step + 1, "dev", dev_metrics)
            ckpt_path = ckpt_dir / f"step_{step + 1:06d}.pt"
            torch.save(
                {
                    "config_path": str(cfg_path),
                    "subset_config_path": str(subset_cfg_path),
                    "state_dict": model.state_dict(),
                    "step": step + 1,
                    "dev_metrics": dev_metrics,
                    "teacher_scores_shape": tuple(int(v) for v in teacher_scores.shape),
                },
                ckpt_path,
            )
            dev_loss = float(dev_metrics.get("loss", float("inf")))
            if math.isfinite(dev_loss) and dev_loss < best_dev:
                best_dev = dev_loss
                torch.save(
                    {
                        "config_path": str(cfg_path),
                        "subset_config_path": str(subset_cfg_path),
                        "state_dict": model.state_dict(),
                        "step": step + 1,
                        "dev_metrics": dev_metrics,
                    },
                    output_dir / "best.pt",
                )
            print(f"[dev {step + 1}] loss={dev_metrics.get('loss', float('nan')):.4f}")
        step += 1

    print(f"Training complete. Best checkpoint: {output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
