from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from dino_slam3.data.tum_rgbd import TUMRGBDDataset, CameraIntrinsics
from dino_slam3.models.network import LocalFeatureNet
from dino_slam3.losses.geometry_supervision import compute_losses
from dino_slam3.utils.rich_logging import print_epoch_header, print_metrics_table
from dino_slam3.utils.config import ensure_dir


def _make_loaders(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    dcfg = cfg["dataset"]
    cam = cfg["camera"]
    intr = CameraIntrinsics(
        fx=float(cam["fx"]), fy=float(cam["fy"]), cx=float(cam["cx"]), cy=float(cam["cy"])
    )

    def build(seqs, is_train: bool):
        datasets = []
        for s in seqs:
            datasets.append(
                TUMRGBDDataset(
                    dataset_root=dcfg["root"],
                    sequence=s,
                    intrinsics=intr,
                    input_size=int(dcfg["input_size"]),
                    frame_spacing=int(dcfg["frame_spacing"]),
                    max_frames=dcfg.get("max_frames"),
                    augmentation=dcfg.get("augmentation"),
                    is_train=is_train,
                )
            )
        return ConcatDataset(datasets)

    train_ds = build(dcfg["train_sequences"], True)
    val_ds = build(dcfg["val_sequences"], False)

    tcfg = cfg["training"]

    # ---- dataloader knobs (defaults tuned for big machines / H100) ----
    batch_size = int(tcfg["batch_size"])
    num_workers = int(tcfg.get("num_workers", 16))  # raise this on 256GB RAM machines
    pin_memory = bool(tcfg.get("pin_memory", True))
    persistent_workers = bool(tcfg.get("persistent_workers", True))
    prefetch_factor = int(tcfg.get("prefetch_factor", 4))

    # If num_workers == 0, PyTorch disallows persistent_workers/prefetch_factor
    dl_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    if num_workers > 0:
        dl_kwargs["persistent_workers"] = persistent_workers
        dl_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        **dl_kwargs,
    )

    val_kwargs = dict(dl_kwargs)
    val_kwargs["drop_last"] = False
    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        **val_kwargs,
    )

    return train_loader, val_loader


def _device(cfg: Dict[str, Any]) -> torch.device:
    d = cfg.get("device", "auto")
    if d == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(d)


def _build_model(cfg: Dict[str, Any]) -> LocalFeatureNet:
    m = cfg["model"]
    fine = m.get("fine_cnn", {})
    heads = m.get("heads", {})
    net = LocalFeatureNet(
        patch_size=int(m.get("patch_size", 16)),
        descriptor_dim=int(m.get("descriptor_dim", 128)),
        fine_channels=int(fine.get("channels", 64)),
        fine_blocks=int(fine.get("num_blocks", 6)),
        fine_stride=int(fine.get("out_stride", 4)),
        use_offset=bool(heads.get("offset", {}).get("enabled", True)),
        use_reliability=bool(heads.get("reliability", {}).get("enabled", True)),
        freeze_backbone=True,
    )
    return net


def train(cfg: Dict[str, Any]) -> None:
    device = _device(cfg)
    train_loader, val_loader = _make_loaders(cfg)
    model = _build_model(cfg).to(device)

    opt = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    sch = CosineAnnealingLR(
        opt,
        T_max=int(cfg["training"]["epochs"]),
        eta_min=float(cfg["training"].get("lr_min", 1e-6)),
    )

    out_dir = ensure_dir(Path(cfg["run"]["out_dir"]) / cfg["run"]["name"])
    ckpt_dir = ensure_dir(out_dir / "checkpoints")

    epochs = int(cfg["training"]["epochs"])
    stride = int(cfg["model"]["fine_cnn"]["out_stride"])

    # ---- AMP settings (H100: BF16 is usually best) ----
    mp_enabled = bool(cfg["training"].get("mixed_precision", True)) and device.type == "cuda"
    amp_dtype_str = str(cfg["training"].get("amp_dtype", "bf16")).lower()
    amp_dtype = torch.bfloat16 if amp_dtype_str in ("bf16", "bfloat16") else torch.float16

    scaler = torch.amp.GradScaler("cuda", enabled=mp_enabled)

    best_val = float("inf")
    best_path = ckpt_dir / "best_model.pt"
    last_path = ckpt_dir / "last.pt"

    log_every = int(cfg["training"].get("log_every", 50))
    save_last = bool(cfg["training"].get("save_last", True))

    for epoch in range(1, epochs + 1):
        lr = opt.param_groups[0]["lr"]
        print_epoch_header(epoch, epochs, lr)

        # ----------------- TRAIN -----------------
        model.train()
        train_metrics = {
            "loss_total": 0.0,
            "loss_contrastive": 0.0,
            "loss_repeat": 0.0,
            "loss_offset": 0.0,
            "loss_reg": 0.0,
        }
        n_train = 0

        pbar = tqdm(train_loader, desc="train", leave=False)
        for it, batch in enumerate(pbar, start=1):
            # move batch to GPU
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type="cuda",
                dtype=amp_dtype,
                enabled=scaler.is_enabled(),
            ):
                out1 = model(batch["rgb1"])
                out2 = model(batch["rgb2"])
                losses = compute_losses(batch, out1, out2, stride=stride, cfg=cfg["loss"])
                loss = losses["loss_total"]

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            bs = int(batch["rgb1"].shape[0])
            n_train += bs
            for k in train_metrics:
                train_metrics[k] += float(losses[k].detach().cpu()) * bs

            if log_every > 0 and (it % log_every == 0):
                pbar.set_postfix(
                    {
                        "loss_total": f"{train_metrics['loss_total']/max(n_train,1):.4f}",
                        "loss_contrastive": f"{train_metrics['loss_contrastive']/max(n_train,1):.4f}",
                    }
                )

        for k in train_metrics:
            train_metrics[k] /= max(n_train, 1)

        # ----------------- VAL -----------------
        model.eval()
        val_metrics = {k: 0.0 for k in train_metrics}
        n_val = 0
        with torch.no_grad(), torch.amp.autocast(
            device_type="cuda",
            dtype=amp_dtype,
            enabled=scaler.is_enabled(),
        ):
            for batch in tqdm(val_loader, desc="val", leave=False):
                for k, v in batch.items():
                    if torch.is_tensor(v):
                        batch[k] = v.to(device, non_blocking=True)

                out1 = model(batch["rgb1"])
                out2 = model(batch["rgb2"])
                losses = compute_losses(batch, out1, out2, stride=stride, cfg=cfg["loss"])

                bs = int(batch["rgb1"].shape[0])
                n_val += bs
                for k in val_metrics:
                    val_metrics[k] += float(losses[k].detach().cpu()) * bs

        for k in val_metrics:
            val_metrics[k] /= max(n_val, 1)

        print_metrics_table("Train", train_metrics)
        print_metrics_table("Val", val_metrics)

        sch.step()

        # ----------------- SAVE BEST -----------------
        val_key = "loss_total"
        val_score = float(val_metrics[val_key])

        # Always save last (optional)
        if save_last:
            torch.save(
                {
                    "epoch": epoch,
                    "best_val": best_val,
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "config": cfg,
                },
                last_path,
            )

        if val_score < best_val:
            best_val = val_score
            torch.save(
                {
                    "epoch": epoch,
                    "best_val": best_val,
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "config": cfg,
                },
                best_path,
            )
