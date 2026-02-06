from __future__ import annotations
import argparse
from dino_slam3.utils.config import load_config
from dino_slam3.utils.seed import seed_everything
from dino_slam3.training.trainer import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg.get("seed", 42)))
    train(cfg)


if __name__ == "__main__":
    main()
