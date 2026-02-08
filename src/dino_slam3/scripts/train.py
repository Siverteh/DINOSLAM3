from __future__ import annotations
import argparse
from dino_slam3.utils.config import load_config
from dino_slam3.training.trainer import train

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    train(cfg)

if __name__ == "__main__":
    main()
