from __future__ import annotations
import argparse
from dino_slam3.utils.config import load_config
from dino_slam3.training.trainer import train

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--init-ckpt", type=str, default=None, help="Initialize model weights from checkpoint.")
    ap.add_argument("--resume-ckpt", type=str, default=None, help="Resume training from checkpoint.")
    ap.add_argument("--experiment-id", type=str, default=None, help="Stable experiment ID used for SQLite/artifact tracking.")
    ap.add_argument("--prompt-tag", type=str, default=None, help="Short label describing the training idea/prompt.")
    ap.add_argument("--parent-experiment-id", type=str, default=None, help="Optional lineage pointer to parent experiment.")
    ap.add_argument("--ablation-tag", type=str, default=None, help="Optional ablation variant tag.")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.init_ckpt:
        cfg.setdefault("training", {})["init_checkpoint"] = args.init_ckpt
    if args.resume_ckpt:
        cfg.setdefault("training", {})["resume_checkpoint"] = args.resume_ckpt
    if args.experiment_id:
        cfg.setdefault("run", {})["experiment_id"] = args.experiment_id
    if args.prompt_tag:
        cfg.setdefault("run", {})["prompt_tag"] = args.prompt_tag
    if args.parent_experiment_id:
        cfg.setdefault("run", {})["parent_experiment_id"] = args.parent_experiment_id
    if args.ablation_tag:
        cfg.setdefault("run", {})["ablation_tag"] = args.ablation_tag
    train(cfg)

if __name__ == "__main__":
    main()
