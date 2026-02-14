from __future__ import annotations
import argparse
from dino_slam3.utils.config import load_config
from dino_slam3.training.trainer import train

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--init-ckpt", type=str, default=None, help="Initialize model weights from checkpoint.")
    ap.add_argument("--resume-ckpt", type=str, default=None, help="Resume training from checkpoint.")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.init_ckpt:
        cfg.setdefault("training", {})["init_checkpoint"] = args.init_ckpt
    if args.resume_ckpt:
        cfg.setdefault("training", {})["resume_checkpoint"] = args.resume_ckpt
    train(cfg)

if __name__ == "__main__":
    main()
