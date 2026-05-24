from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle
from PIL import Image

try:
    from scipy import stats
except Exception:  # pragma: no cover - scipy exists in the DPVO env, but keep this portable.
    stats = None


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO_ROOT / "chapter4_visualizations"
PATCH_DIR = OUT_ROOT / "patch_overlays"
SEPARATE_PATCH_DIR = PATCH_DIR / "separate_images"
VAR_DIR = OUT_ROOT / "variance_stats"
DATA_DIR = OUT_ROOT / "source_data"
HANDOFF_ROOT = OUT_ROOT.parent / "THESIS_HANDOFF_MULTISCALE_32X4_75_25_V1"
FOCUS_DPVO_5X_ROOT = REPO_ROOT / "refocus_vo/runs/eval/tum_rgbd_freiburg123_dpvo_vs_focus071_v1_rerun6"

PATCH_SIZE_MODEL = 16
MODEL_W = 320
MODEL_H = 240
DISPLAY_GRID_ROWS = 6
DISPLAY_GRID_COLS = 8
DISPLAY_PATCH_BUDGET = 96
BEST_MULTISCALE_LABEL = "Multiscale 50/50"
LEGACY_WINNER_REPEAT_ID = "multiscale_32x4_v1_hybrid75_25"

SEQUENCES = [
    "freiburg1_desk",
    "freiburg2_desk_with_person",
    "freiburg3_walking_static",
    "freiburg3_large_cabinet",
]

SEQUENCE_LABELS = {
    "freiburg1_desk": "F1 desk",
    "freiburg2_desk_with_person": "F2 desk + person",
    "freiburg3_walking_static": "F3 walking static",
    "freiburg3_large_cabinet": "F3 large cabinet",
}

PATCH_DIAGNOSTICS = {
    "Multiscale 75/25": REPO_ROOT
    / "refocus_vo/runs/eval/tum_rgbd_freiburg123_arch_ratio_ablation_v1/screening/multiscale_32x4_v1/hybrid75_25/patch_diagnostics.jsonl",
    "Multiscale 50/50": REPO_ROOT
    / "refocus_vo/runs/eval/tum_rgbd_freiburg123_arch_ratio_ablation_v1/screening/multiscale_32x4_v1/hybrid50_50/patch_diagnostics.jsonl",
    "Pure multiscale": REPO_ROOT
    / "refocus_vo/runs/eval/tum_rgbd_freiburg123_arch_ratio_ablation_v1/screening/multiscale_32x4_v1/pure100/patch_diagnostics.jsonl",
}

PATCH_VARIANTS = [
    "Native DPVO",
    "Multiscale 75/25",
    "Multiscale 50/50",
    "Pure multiscale",
]

SEPARATE_PATCH_VARIANTS = [
    "Native DPVO",
    "Multiscale 50/50",
    "Pure multiscale",
]

COLORS = {
    "native": "#2B8CBE",
    "dino": "#E88C30",
    "unknown": "#6B7280",
    "freiburg1": "#4C78A8",
    "freiburg2": "#F58518",
    "freiburg3": "#54A24B",
    "winner": "#2F855A",
    "dpvo": "#4C78A8",
    "focus": "#B279A2",
    "micro4": "#E45756",
}

METHOD_COLORS = {
    "Native DPVO": COLORS["dpvo"],
    "Focus071": COLORS["focus"],
    BEST_MULTISCALE_LABEL: COLORS["winner"],
}

FAMILY_COLS = {
    "freiburg1": "freiburg1_mean_ate_rmse_associated",
    "freiburg2": "freiburg2_mean_ate_rmse_associated",
    "freiburg3": "freiburg3_mean_ate_rmse_associated",
}

FONT_BASE = 14
FONT_TICK = 16
FONT_AXIS = 20
FONT_TITLE = 24
FONT_SUPTITLE = 32
FONT_LEGEND = 18
FONT_ANNOTATION = 17
FONT_OVERLAY_TITLE = 16


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": FONT_BASE,
            "axes.titlesize": FONT_TITLE,
            "axes.labelsize": FONT_AXIS,
            "xtick.labelsize": FONT_TICK,
            "ytick.labelsize": FONT_TICK,
            "legend.fontsize": FONT_LEGEND,
            "figure.titlesize": FONT_SUPTITLE,
        }
    )


def trim_png_whitespace(path: Path, *, threshold: int = 250) -> None:
    with Image.open(path) as image:
        rgba = image.convert("RGBA")
        arr = np.asarray(rgba)
        alpha = arr[:, :, 3] > 0
        non_white = alpha & np.any(arr[:, :, :3] < threshold, axis=2)
        if not bool(non_white.any()):
            return
        ys, xs = np.where(non_white)
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        if bbox == (0, 0, rgba.width, rgba.height):
            return
        rgba.crop(bbox).save(path)


def save_png(fig: plt.Figure, path: Path, *, dpi: int = 180) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    trim_png_whitespace(path)


def save_canvas_png(fig: plt.Figure, path: Path, *, dpi: int = 180) -> None:
    fig.savefig(path, dpi=dpi, pad_inches=0)
    trim_png_whitespace(path)


def ensure_dirs() -> None:
    for path in (PATCH_DIR, SEPARATE_PATCH_DIR, VAR_DIR, DATA_DIR):
        path.mkdir(parents=True, exist_ok=True)


def clean_regenerated_stats() -> None:
    for path in VAR_DIR.glob("*.png"):
        path.unlink()
    for path in DATA_DIR.glob("*.csv"):
        path.unlink()


def clean_axes(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def load_first_frame(sequence: str) -> Image.Image:
    rgb_dir = REPO_ROOT / f"src/dino_slam3/data/tum_rgbd/rgbd_dataset_{sequence}/rgb"
    images = sorted(rgb_dir.glob("*.png"))
    if not images:
        raise FileNotFoundError(f"No RGB frames found for {sequence}: {rgb_dir}")
    return Image.open(images[0]).convert("RGB")


def read_first_frame_patches(
    path: Path,
    *,
    sequences: list[str],
    max_patches_per_sequence: int = 192,
) -> dict[str, list[dict]]:
    wanted = {f'"sequence": "{seq}"': seq for seq in sequences}
    records: dict[str, list[dict]] = {seq: [] for seq in sequences}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            seq = None
            for needle, candidate in wanted.items():
                if needle in line:
                    seq = candidate
                    break
            if seq is None or len(records[seq]) >= max_patches_per_sequence:
                continue
            item = json.loads(line)
            if int(item.get("insertion_frame", -1)) != 0:
                continue
            if str(item.get("status", "")) not in {"ok", "partial_low_coverage"}:
                continue
            records[seq].append(item)
            if all(len(records[s]) >= max_patches_per_sequence for s in sequences):
                break
    return records


def display_patch_counts(variant: str) -> dict[str, int]:
    if variant == "Native DPVO":
        return {"native": DISPLAY_PATCH_BUDGET, "dino": 0}
    if variant == "Multiscale 75/25":
        return {"native": 72, "dino": 24}
    if variant == "Multiscale 50/50":
        return {"native": 48, "dino": 48}
    if variant == "Pure multiscale":
        return {"native": 0, "dino": DISPLAY_PATCH_BUDGET}
    raise KeyError(f"Unknown patch variant: {variant}")


def select_display_patches(records: list[dict], variant: str) -> list[dict]:
    selected: list[dict] = []
    for source_name, count in display_patch_counts(variant).items():
        if count <= 0:
            continue
        selected.extend(
            [record for record in records if str(record.get("source", "unknown")) == source_name][:count]
        )
    return selected


def load_patch_sets() -> dict[str, dict[str, list[dict]]]:
    raw_loaded = {
        name: read_first_frame_patches(path, sequences=SEQUENCES)
        for name, path in PATCH_DIAGNOSTICS.items()
    }
    native_from_reference = {
        seq: [r for r in raw_loaded["Multiscale 75/25"][seq] if r.get("source") == "native"]
        for seq in SEQUENCES
    }
    raw_loaded["Native DPVO"] = native_from_reference
    return {
        name: {seq: select_display_patches(raw_loaded[name][seq], name) for seq in SEQUENCES}
        for name in PATCH_VARIANTS
    }


def patch_xy(record: dict, image_w: int, image_h: int) -> tuple[float, float]:
    sx = float(image_w) / float(MODEL_W)
    sy = float(image_h) / float(MODEL_H)
    return float(record["initial_x"]) * sx, float(record["initial_y"]) * sy


def draw_patch_overlay(
    ax: plt.Axes,
    image: Image.Image,
    patches: list[dict],
    *,
    title: str,
    label_overlay: bool = True,
) -> None:
    ax.imshow(image)
    clean_axes(ax)

    w, h = image.size
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")
    sx = w / MODEL_W
    sy = h / MODEL_H
    box_w = PATCH_SIZE_MODEL * sx
    box_h = PATCH_SIZE_MODEL * sy

    for col in range(1, DISPLAY_GRID_COLS):
        x = col * w / DISPLAY_GRID_COLS
        ax.axvline(x, color="white", lw=0.45, alpha=0.22)
    for row in range(1, DISPLAY_GRID_ROWS):
        y = row * h / DISPLAY_GRID_ROWS
        ax.axhline(y, color="white", lw=0.45, alpha=0.22)

    selected = patches[:DISPLAY_PATCH_BUDGET]
    for source_name in ("native", "dino", "unknown"):
        group = [p for p in selected if str(p.get("source", "unknown")) == source_name]
        if not group:
            continue
        xs, ys = zip(*(patch_xy(p, w, h) for p in group))
        color = COLORS[source_name]
        ax.scatter(
            xs,
            ys,
            s=18 if source_name == "dino" else 15,
            c=color,
            edgecolors="white",
            linewidths=0.35,
            alpha=0.95,
            label=source_name,
            zorder=3,
            clip_on=True,
        )
        for x, y in zip(xs, ys):
            rect = plt.Rectangle(
                (x - box_w / 2, y - box_h / 2),
                box_w,
                box_h,
                fill=False,
                edgecolor=color,
                linewidth=0.75,
                alpha=0.55,
                zorder=2,
                clip_on=True,
            )
            ax.add_patch(rect)

    counts = Counter(str(p.get("source", "unknown")) for p in selected)
    suffix = " / ".join(f"{k}:{counts[k]}" for k in ("native", "dino") if counts[k])
    if label_overlay:
        ax.text(
            0.015,
            0.975,
            f"{title}\n{suffix}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=FONT_OVERLAY_TITLE,
            color="white",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="black", edgecolor="none", alpha=0.58),
            zorder=5,
        )


def make_patch_overlay_grid(patch_sets: dict[str, dict[str, list[dict]]]) -> None:
    fig, axes = plt.subplots(
        len(SEQUENCES),
        len(PATCH_VARIANTS),
        figsize=(19.2, 14.4),
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    for row, seq in enumerate(SEQUENCES):
        image = load_first_frame(seq)
        for col, variant in enumerate(PATCH_VARIANTS):
            draw_patch_overlay(
                axes[row, col],
                image,
                patch_sets[variant][seq],
                title=f"{SEQUENCE_LABELS[seq]} | {variant}",
                label_overlay=True,
            )
    save_canvas_png(fig, PATCH_DIR / "patch_overlay_comparison_grid.png", dpi=180)
    plt.close(fig)

    for seq in SEQUENCES:
        fig, axes = plt.subplots(1, len(PATCH_VARIANTS), figsize=(19.2, 3.6), constrained_layout=False)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
        image = load_first_frame(seq)
        for col, variant in enumerate(PATCH_VARIANTS):
            draw_patch_overlay(
                axes[col],
                image,
                patch_sets[variant][seq],
                title=variant,
                label_overlay=True,
            )
        save_canvas_png(fig, PATCH_DIR / f"patch_overlay_{seq}.png", dpi=180)
        plt.close(fig)


def make_separate_patch_overlays(patch_sets: dict[str, dict[str, list[dict]]]) -> None:
    slugs = {
        "Native DPVO": "native_dpvo",
        "Multiscale 50/50": "multiscale_50_50",
        "Pure multiscale": "pure_multiscale",
    }
    for stale_file in SEPARATE_PATCH_DIR.glob("*__*.png"):
        stale_file.unlink()

    for seq in SEQUENCES:
        image = load_first_frame(seq)
        for variant in SEPARATE_PATCH_VARIANTS:
            dpi = 200
            fig = plt.figure(figsize=(image.width / dpi, image.height / dpi), dpi=dpi)
            ax = fig.add_axes([0, 0, 1, 1])
            draw_patch_overlay(
                ax,
                image,
                patch_sets[variant][seq],
                title=f"{SEQUENCE_LABELS[seq]} | {variant}",
                label_overlay=False,
            )
            save_canvas_png(fig, SEPARATE_PATCH_DIR / f"{seq}__{slugs[variant]}.png", dpi=dpi)
            plt.close(fig)


def make_patch_density_heatmaps(patch_sets: dict[str, dict[str, list[dict]]]) -> None:
    bins_x = np.linspace(0, MODEL_W, 17)
    bins_y = np.linspace(0, MODEL_H, 13)
    maps: dict[str, np.ndarray] = {}
    for variant in PATCH_VARIANTS:
        xs: list[float] = []
        ys: list[float] = []
        for seq in SEQUENCES:
            for record in patch_sets[variant][seq]:
                xs.append(float(record["initial_x"]))
                ys.append(float(record["initial_y"]))
        hist, _, _ = np.histogram2d(ys, xs, bins=[bins_y, bins_x])
        maps[variant] = hist

    vmax = max(float(m.max()) for m in maps.values()) or 1.0
    fig, axes = plt.subplots(1, len(PATCH_VARIANTS), figsize=(18.2, 4.8), constrained_layout=True)
    for ax, variant in zip(axes, PATCH_VARIANTS):
        im = ax.imshow(maps[variant], cmap="magma", vmin=0, vmax=vmax, origin="upper")
        ax.set_title(variant, fontsize=FONT_TITLE)
        ax.set_xlabel("image x-bin")
        ax.set_ylabel("image y-bin")
        ax.set_xticks(range(0, 16, 4))
        ax.set_yticks(range(0, 12, 3))
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, pad=0.01)
    cbar.set_label("patch count across selected first frames")
    fig.suptitle("Spatial Patch Density Across TUM Examples", fontsize=FONT_SUPTITLE, y=1.05)
    save_png(fig, PATCH_DIR / "patch_density_heatmaps.png", dpi=180)
    plt.close(fig)


def stream_patch_summary(
    name: str,
    path: Path,
    *,
    source_filter: str | None = None,
    max_records_per_sequence: int = 25000,
) -> dict:
    seq_counts = defaultdict(int)
    total = 0
    source_counts = Counter()
    survived5 = 0
    boundary = 0
    unstable = 0
    track_ages: list[float] = []
    utilities: list[float] = []
    wanted = {f'"sequence": "{seq}"': seq for seq in SEQUENCES}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            seq = None
            for needle, candidate in wanted.items():
                if needle in line:
                    seq = candidate
                    break
            if seq is None or seq_counts[seq] >= max_records_per_sequence:
                continue
            item = json.loads(line)
            source = str(item.get("source", "unknown"))
            if source_filter is not None and source != source_filter:
                continue
            if str(item.get("status", "")) not in {"ok", "partial_low_coverage"}:
                continue
            seq_counts[seq] += 1
            total += 1
            source_counts[source] += 1
            survived5 += int(bool(item.get("survived_5", False)))
            boundary += int(bool(item.get("near_depth_boundary", False)))
            unstable += int(bool(item.get("unstable_motion_proxy", False)))
            if item.get("track_age") is not None:
                track_ages.append(float(item["track_age"]))
            if item.get("selection_utility") is not None:
                utilities.append(float(item["selection_utility"]))
            if all(seq_counts[s] >= max_records_per_sequence for s in SEQUENCES):
                break
    total_safe = max(total, 1)
    return {
        "variant": name,
        "total_records": total,
        "native_count": source_counts["native"],
        "dino_count": source_counts["dino"],
        "survived5_rate": survived5 / total_safe,
        "near_depth_boundary_rate": boundary / total_safe,
        "unstable_motion_proxy_rate": unstable / total_safe,
        "mean_track_age": float(np.mean(track_ages)) if track_ages else math.nan,
        "mean_selection_utility": float(np.mean(utilities)) if utilities else math.nan,
    }


def make_patch_diagnostic_summary() -> None:
    rows = [
        stream_patch_summary("Native DPVO", PATCH_DIAGNOSTICS["Multiscale 75/25"], source_filter="native"),
        stream_patch_summary(BEST_MULTISCALE_LABEL, PATCH_DIAGNOSTICS[BEST_MULTISCALE_LABEL]),
    ]
    summary = pd.DataFrame(rows)
    summary.to_csv(DATA_DIR / "patch_diagnostic_summary_selected_sequences.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0), constrained_layout=True)
    x = np.arange(len(summary))
    axes[0].bar(x, summary["native_count"], label="native", color=COLORS["native"])
    axes[0].bar(x, summary["dino_count"], bottom=summary["native_count"], label="dino", color=COLORS["dino"])
    axes[0].set_xticks(x, summary["variant"], rotation=12, ha="right")
    axes[0].set_ylabel("records sampled")
    axes[0].set_title("Patch source mix")
    axes[0].legend(frameon=False)

    width = 0.26
    axes[1].bar(x - width, summary["survived5_rate"], width, color="#2F855A", label="survived 5")
    axes[1].bar(x, summary["near_depth_boundary_rate"], width, color="#E45756", label="near depth boundary")
    axes[1].bar(x + width, summary["unstable_motion_proxy_rate"], width, color="#F58518", label="motion proxy")
    axes[1].set_xticks(x, summary["variant"], rotation=12, ha="right")
    axes[1].set_ylim(0, 1.0)
    axes[1].set_ylabel("rate")
    axes[1].set_title("Patch diagnostic rates")
    axes[1].legend(frameon=False, fontsize=FONT_LEGEND)

    axes[2].bar(x, summary["mean_track_age"], color="#6B7280")
    axes[2].set_xticks(x, summary["variant"], rotation=12, ha="right")
    axes[2].set_ylabel("frames")
    axes[2].set_title("Mean track age")

    fig.suptitle("Patch Diagnostics: Native DPVO Source vs Multiscale 50/50", fontsize=FONT_SUPTITLE, y=1.05)
    save_png(fig, PATCH_DIR / "patch_source_survival_summary.png", dpi=180)
    plt.close(fig)


def paired_tests(candidate: np.ndarray, baseline: np.ndarray) -> dict[str, float]:
    delta = candidate - baseline
    output = {
        "mean_delta": float(delta.mean()),
        "std_delta": float(delta.std(ddof=1)),
        "n": float(len(delta)),
    }
    if stats is not None:
        output["paired_t_p_two_sided"] = float(stats.ttest_rel(candidate, baseline).pvalue)
        try:
            output["wilcoxon_p_two_sided"] = float(stats.wilcoxon(candidate, baseline).pvalue)
        except ValueError:
            output["wilcoxon_p_two_sided"] = math.nan
        output["sign_test_p_two_sided"] = float(stats.binomtest(int((delta < 0).sum()), len(delta), 0.5).pvalue)
    return output


def ratio_ablation_table() -> pd.DataFrame:
    rows = [
        ("multiscale_32x4", "90 / 10", 0.90, 0.10, 0.153, 26),
        ("multiscale_32x4", "75 / 25", 0.75, 0.25, 0.131, 29),
        ("multiscale_32x4", "50 / 50", 0.50, 0.50, 0.115, 31),
        ("multiscale_32x4", "25 / 75", 0.25, 0.75, 0.146, 26),
        ("multiscale_32x4", "0 / 100", 0.00, 1.00, 0.212, 25),
        ("micro4_grid", "90 / 10", 0.90, 0.10, 0.155, 33),
        ("micro4_grid", "75 / 25", 0.75, 0.25, 0.150, 26),
        ("micro4_grid", "50 / 50", 0.50, 0.50, 0.140, 30),
        ("micro4_grid", "25 / 75", 0.25, 0.75, 0.174, 27),
        ("micro4_grid", "0 / 100", 0.00, 1.00, 0.241, 19),
        ("multiscale_24x5", "90 / 10", 0.90, 0.10, 0.145, 29),
        ("multiscale_24x5", "75 / 25", 0.75, 0.25, 0.139, 30),
        ("multiscale_24x5", "50 / 50", 0.50, 0.50, 0.128, 32),
        ("multiscale_24x5", "25 / 75", 0.25, 0.75, 0.149, 31),
        ("multiscale_24x5", "0 / 100", 0.00, 1.00, 0.186, 24),
    ]
    df = pd.DataFrame(
        rows,
        columns=[
            "architecture",
            "ratio",
            "native_fraction",
            "dino_fraction",
            "mean_ate_rmse_associated",
            "wins_vs_native_dpvo_median",
        ],
    )
    df["is_best_mean"] = df["mean_ate_rmse_associated"] == df["mean_ate_rmse_associated"].min()
    df["is_best_wins"] = df["wins_vs_native_dpvo_median"] == df["wins_vs_native_dpvo_median"].max()
    return df


def load_repeat_headline_frame() -> pd.DataFrame:
    repeat_path = HANDOFF_ROOT / "RESULTS/derived/winner_repeat_distribution.csv"
    df = pd.read_csv(repeat_path)
    return pd.DataFrame(
        {
            "repeat": np.arange(1, len(df) + 1),
            "Native DPVO": df["dpvo_historical_full_mean_ate_rmse_associated"].to_numpy(float),
            "Focus071": df["old_focus071_historical_full_mean_ate_rmse_associated"].to_numpy(float),
            BEST_MULTISCALE_LABEL: df["winner_full_mean_ate_rmse_associated"].to_numpy(float),
        }
    )


def write_repeat_significance(headline: pd.DataFrame) -> dict[str, dict[str, float]]:
    comparisons = {
        f"{BEST_MULTISCALE_LABEL} vs Native DPVO": paired_tests(
            headline[BEST_MULTISCALE_LABEL].to_numpy(float),
            headline["Native DPVO"].to_numpy(float),
        ),
        f"{BEST_MULTISCALE_LABEL} vs Focus071": paired_tests(
            headline[BEST_MULTISCALE_LABEL].to_numpy(float),
            headline["Focus071"].to_numpy(float),
        ),
    }
    with (DATA_DIR / "repeat_significance_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        keys = list(next(iter(comparisons.values())).keys())
        writer.writerow(["comparison", *keys])
        for label, tests in comparisons.items():
            writer.writerow([label, *[tests[k] for k in keys]])
    return comparisons


def plot_repeat_variance(headline: pd.DataFrame, methods: list[str], filename: str) -> None:
    plot_df = headline[["repeat", *methods]].copy()
    plot_df.to_csv(DATA_DIR / f"{Path(filename).stem}.csv", index=False)

    repeats = plot_df["repeat"].to_numpy(int)
    tests = write_repeat_significance(headline)
    fig, axes = plt.subplots(1, 2, figsize=(15.8, 6.2), constrained_layout=False)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.16, top=0.69, wspace=0.24)
    axes[0].set_xlim(float(repeats.min()) - 0.45, float(repeats.max()) + 0.45)
    legend_handles: list[Patch] = []
    for method in methods:
        values = plot_df[method].to_numpy(float)
        low = float(values.min())
        high = float(values.max())
        legend_handles.append(
            Patch(
                facecolor=METHOD_COLORS[method],
                edgecolor=METHOD_COLORS[method],
                alpha=0.22,
                label=method,
            )
        )
        axes[0].axhspan(
            low,
            high,
            color=METHOD_COLORS[method],
            alpha=0.08,
            linewidth=0,
            zorder=0,
        )
        axes[0].scatter(
            repeats,
            values,
            s=95 if method == BEST_MULTISCALE_LABEL else 80,
            color=METHOD_COLORS[method],
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
    axes[0].set_xlabel("repeat")
    axes[0].set_xticks(repeats)
    axes[0].set_ylabel("full mean ATE RMSE associated")
    axes[0].set_title("Five-repeat headline variance")
    axes[0].grid(alpha=0.25)
    fig.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.08, 0.84),
        frameon=False,
        ncol=len(methods),
        handlelength=1.4,
        columnspacing=1.4,
    )

    axes[1].axhline(0, color="#111827", lw=1.0)
    delta_dpvo = plot_df[BEST_MULTISCALE_LABEL].to_numpy(float) - plot_df["Native DPVO"].to_numpy(float)
    if "Focus071" in methods:
        delta_focus = plot_df[BEST_MULTISCALE_LABEL].to_numpy(float) - plot_df["Focus071"].to_numpy(float)
        width = 0.35
        axes[1].bar(repeats - width / 2, delta_dpvo, width, color=COLORS["dpvo"], label="MS - DPVO")
        axes[1].bar(repeats + width / 2, delta_focus, width, color=COLORS["focus"], label="MS - Focus071")
        axes[1].legend(frameon=False, fontsize=FONT_LEGEND)
        p_text = (
            f"vs DPVO p={tests[f'{BEST_MULTISCALE_LABEL} vs Native DPVO']['paired_t_p_two_sided']:.4f}\n"
            f"vs Focus071 p={tests[f'{BEST_MULTISCALE_LABEL} vs Focus071']['paired_t_p_two_sided']:.4f}"
        )
    else:
        axes[1].bar(
            repeats,
            delta_dpvo,
            color=[COLORS["winner"] if d < 0 else "#E45756" for d in delta_dpvo],
        )
        p_text = (
            f"paired t p={tests[f'{BEST_MULTISCALE_LABEL} vs Native DPVO']['paired_t_p_two_sided']:.4f}\n"
            f"Wilcoxon p={tests[f'{BEST_MULTISCALE_LABEL} vs Native DPVO']['wilcoxon_p_two_sided']:.4f}\n"
            f"sign p={tests[f'{BEST_MULTISCALE_LABEL} vs Native DPVO']['sign_test_p_two_sided']:.4f}"
        )
    axes[1].set_xlabel("repeat")
    axes[1].set_xticks(repeats)
    axes[1].set_ylabel("multiscale - baseline")
    axes[1].text(
        0.98,
        0.03,
        p_text,
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=FONT_ANNOTATION,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#D1D5DB", alpha=0.9),
    )
    axes[1].set_title("Paired repeat deltas")
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle("Repeat Stability and Significance Context", fontsize=FONT_SUPTITLE, y=0.98)
    save_png(fig, VAR_DIR / filename, dpi=180)
    plt.close(fig)


def make_repeat_variance_plots() -> None:
    headline = load_repeat_headline_frame()
    plot_repeat_variance(
        headline,
        ["Native DPVO", "Focus071", BEST_MULTISCALE_LABEL],
        "repeat_variance_focus071_multiscale_dpvo.png",
    )
    plot_repeat_variance(
        headline,
        ["Native DPVO", BEST_MULTISCALE_LABEL],
        "repeat_variance_multiscale_dpvo.png",
    )


def load_family_repeat_frame() -> pd.DataFrame:
    final_path = HANDOFF_ROOT / "RESULTS/final_dual_finalists_5x/summary/repeat_summary.csv"
    old_path = HANDOFF_ROOT / "RESULTS/old_focus071_5x/summary/repeat_summary.csv"
    final = pd.read_csv(final_path)
    final = final[final["finalist_id"] == LEGACY_WINNER_REPEAT_ID].copy()
    final["method"] = BEST_MULTISCALE_LABEL
    old = pd.read_csv(old_path)
    old["method"] = old["method"].map({"dpvo_native": "Native DPVO", "focus071_best": "Focus071"})
    cols = ["method", "repeat_id", *FAMILY_COLS.values()]
    return pd.concat([old[cols], final[cols]], ignore_index=True)


def make_family_repeat_variance_plot(methods: list[str], filename: str) -> None:
    df = load_family_repeat_frame()
    df = df[df["method"].isin(methods)].copy()
    df.to_csv(DATA_DIR / f"{Path(filename).stem}.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.8), sharey=False, constrained_layout=False)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.26, top=0.74, wspace=0.32)
    for ax, (family, col) in zip(axes, FAMILY_COLS.items()):
        for idx, method in enumerate(methods, start=1):
            values = df[df["method"] == method][col].to_numpy(float)
            color = METHOD_COLORS[method]
            low = float(values.min())
            high = float(values.max())
            median = float(np.median(values))
            ax.add_patch(
                Rectangle(
                    (idx - 0.22, low),
                    0.48,
                    max(high - low, 1e-6),
                    facecolor=color,
                    edgecolor="#111827",
                    linewidth=0.9,
                    alpha=0.32,
                    zorder=0,
                )
            )
            ax.hlines(
                median,
                idx - 0.22,
                idx + 0.26,
                colors="#F58518",
                linewidth=2.0,
                zorder=2,
            )
            jitter = np.linspace(-0.09, 0.09, len(values))
            ax.scatter(
                np.full(len(values), idx) + jitter,
                values,
                s=38,
                color=color,
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
        ax.set_xlim(0.5, len(methods) + 0.5)
        ax.set_xticks(np.arange(1, len(methods) + 1), methods)
        ax.set_title(family)
        ax.set_ylabel("family mean ATE")
        ax.tick_params(axis="x", labelrotation=15)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Per-family Variance Across the Five Repeats", fontsize=FONT_SUPTITLE, y=0.98)
    save_png(fig, VAR_DIR / filename, dpi=180)
    plt.close(fig)


def make_family_repeat_variance_plots() -> None:
    make_family_repeat_variance_plot(
        ["Native DPVO", "Focus071", BEST_MULTISCALE_LABEL],
        "family_repeat_variance_focus071_multiscale_dpvo.png",
    )
    make_family_repeat_variance_plot(
        ["Native DPVO", BEST_MULTISCALE_LABEL],
        "family_repeat_variance_multiscale_dpvo.png",
    )


def load_sequence_comparison_frame() -> pd.DataFrame:
    seq_path = HANDOFF_ROOT / "RESULTS/final_dual_finalists_5x/summary/per_sequence_median.csv"
    df = pd.read_csv(seq_path)
    winner = df[df["finalist_id"] == LEGACY_WINNER_REPEAT_ID].copy()
    return pd.DataFrame(
        {
            "sequence": winner["sequence"],
            "family": winner["family"],
            "Native DPVO": winner["baseline_dpvo_assoc_median"].to_numpy(float),
            "Focus071": winner["baseline_old_focus071_assoc_median"].to_numpy(float),
            BEST_MULTISCALE_LABEL: winner["median_ate_rmse_associated"].to_numpy(float),
        }
    )


def sequence_long_form(seq: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    rows = []
    for _, row in seq.iterrows():
        for method in methods:
            rows.append(
                {
                    "sequence": row["sequence"],
                    "family": row["family"],
                    "method": method,
                    "ate_rmse_associated": row[method],
                }
            )
    return pd.DataFrame(rows)


def make_sequence_distribution_plot(seq: pd.DataFrame, methods: list[str], filename: str) -> None:
    dist = sequence_long_form(seq, methods)
    dist.to_csv(DATA_DIR / f"{Path(filename).stem}.csv", index=False)
    fig, ax = plt.subplots(figsize=(10.2, 5.8), constrained_layout=True)
    groups = [dist[dist["method"] == name]["ate_rmse_associated"].to_numpy(float) for name in methods]
    bp = ax.boxplot(groups, tick_labels=methods, showfliers=False, patch_artist=True)
    for patch, method in zip(bp["boxes"], methods):
        patch.set_facecolor(METHOD_COLORS[method])
        patch.set_alpha(0.45)
    for i, values in enumerate(groups, start=1):
        rng = np.random.default_rng(7 + i)
        ax.scatter(
            rng.normal(i, 0.045, size=len(values)),
            values,
            s=14,
            color=METHOD_COLORS[methods[i - 1]],
            alpha=0.62,
            zorder=3,
        )
    ax.set_yscale("log")
    ax.set_ylabel("per-sequence median ATE RMSE associated, log scale")
    ax.set_title("Sequence-level Error Distribution")
    ax.grid(axis="y", alpha=0.25, which="both")
    save_png(fig, VAR_DIR / filename, dpi=180)
    plt.close(fig)


def make_sequence_delta_plots(seq: pd.DataFrame) -> None:
    deltas = seq[["sequence", "family"]].copy()
    deltas["delta_multiscale_vs_dpvo"] = seq[BEST_MULTISCALE_LABEL] - seq["Native DPVO"]
    deltas["delta_multiscale_vs_focus071"] = seq[BEST_MULTISCALE_LABEL] - seq["Focus071"]
    deltas = deltas.sort_values("delta_multiscale_vs_dpvo", ascending=True)
    deltas.to_csv(DATA_DIR / "sequence_deltas_multiscale_focus071_dpvo.csv", index=False)

    fig_h = max(8.0, 0.25 * len(deltas))
    fig, ax = plt.subplots(figsize=(12.4, fig_h), constrained_layout=True)
    colors = [COLORS.get(str(f), "#6B7280") for f in deltas["family"]]
    ax.barh(deltas["sequence"], deltas["delta_multiscale_vs_dpvo"], color=colors)
    ax.axvline(0, color="#111827", lw=1.0)
    ax.set_xlabel(f"{BEST_MULTISCALE_LABEL} median ATE - native DPVO median ATE")
    ax.set_ylabel("")
    ax.set_title("Where Multiscale 50/50 Gains or Loses by Sequence")
    ax.grid(axis="x", alpha=0.25)
    save_png(fig, VAR_DIR / "sequence_delta_vs_dpvo_waterfall.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(17.2, fig_h), sharey=True, constrained_layout=True)
    for ax, column, title, color in [
        (axes[0], "delta_multiscale_vs_dpvo", "Multiscale - native DPVO", COLORS["dpvo"]),
        (axes[1], "delta_multiscale_vs_focus071", "Multiscale - Focus071", COLORS["focus"]),
    ]:
        ax.barh(deltas["sequence"], deltas[column], color=color, alpha=0.86)
        ax.axvline(0, color="#111827", lw=1.0)
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("Paired Sequence Deltas for Multiscale 50/50", fontsize=FONT_SUPTITLE, y=1.01)
    save_png(fig, VAR_DIR / "sequence_delta_focus071_multiscale_dpvo.png", dpi=180)
    plt.close(fig)


def make_sequence_distribution_plots() -> None:
    seq = load_sequence_comparison_frame()
    seq.to_csv(DATA_DIR / "sequence_comparison_focus071_multiscale_dpvo.csv", index=False)
    make_sequence_distribution_plot(
        seq,
        ["Native DPVO", "Focus071", BEST_MULTISCALE_LABEL],
        "sequence_ate_distribution_focus071_multiscale_dpvo.png",
    )
    make_sequence_distribution_plot(
        seq,
        ["Native DPVO", BEST_MULTISCALE_LABEL],
        "sequence_ate_distribution_multiscale_dpvo.png",
    )
    make_sequence_delta_plots(seq)


def make_ratio_ablation_plot() -> None:
    df = ratio_ablation_table()
    df.to_csv(DATA_DIR / "ratio_ablation_table.csv", index=False)
    order = ["90 / 10", "75 / 25", "50 / 50", "25 / 75", "0 / 100"]
    arch_order = ["multiscale_32x4", "micro4_grid", "multiscale_24x5"]
    df["ratio"] = pd.Categorical(df["ratio"], categories=order, ordered=True)
    df["architecture"] = pd.Categorical(df["architecture"], categories=arch_order, ordered=True)
    df = df.sort_values(["architecture", "ratio"])

    ms32 = df[df["architecture"] == "multiscale_32x4"].copy()
    ms32.to_csv(DATA_DIR / "multiscale_32x4_ratio_ablation.csv", index=False)

    x = np.arange(len(ms32))
    fig, ax1 = plt.subplots(figsize=(11.2, 5.8), constrained_layout=True)
    bar_colors = [COLORS["winner"] if bool(row.is_best_mean) else "#7A869A" for row in ms32.itertuples()]
    ax1.bar(x, ms32["mean_ate_rmse_associated"], color=bar_colors, alpha=0.88)
    ax1.set_xticks(x, [str(v) for v in ms32["ratio"]])
    ax1.set_ylabel("mean ATE RMSE associated")
    ax1.set_xlabel("native / DINO patch ratio")
    ax1.set_title("Multiscale 32x4 Ratio Ablation")
    ax1.grid(axis="y", alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(
        x,
        ms32["wins_vs_native_dpvo_median"],
        color="#111827",
        marker="o",
        lw=2.0,
        label="wins vs DPVO",
    )
    ax2.set_ylabel("wins vs native DPVO median (38 seq)")
    ax2.set_ylim(0, 38)
    for idx, row in enumerate(ms32.itertuples()):
        ax1.text(
            idx,
            row.mean_ate_rmse_associated * 0.52,
            f"{row.mean_ate_rmse_associated:.3f}",
            ha="center",
            va="center",
            fontsize=FONT_ANNOTATION,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=1.0),
            zorder=5,
        )
    ax2.legend(loc="upper right", frameon=False)
    save_png(fig, VAR_DIR / "multiscale_ratio_ablation.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(18.4, 6.0), sharey=True, constrained_layout=False)
    fig.subplots_adjust(left=0.07, right=0.93, bottom=0.24, top=0.76, wspace=0.30)
    for ax, arch in zip(axes, arch_order):
        sub = df[df["architecture"] == arch].copy()
        x = np.arange(len(sub))
        colors = [COLORS["winner"] if bool(row.is_best_mean) else "#7A869A" for row in sub.itertuples()]
        ax.bar(x, sub["mean_ate_rmse_associated"], color=colors, alpha=0.9)
        ax.set_ylim(0, df["mean_ate_rmse_associated"].max() * 1.16)
        ax.set_xticks(x, [str(v) for v in sub["ratio"]], rotation=20, ha="right")
        ax.set_title(str(arch))
        ax.grid(axis="y", alpha=0.25)
        ax_wins = ax.twinx()
        ax_wins.plot(x, sub["wins_vs_native_dpvo_median"], color="#111827", marker="o", lw=1.8)
        ax_wins.set_ylim(0, 38)
        if ax is axes[-1]:
            ax_wins.set_ylabel("wins vs native DPVO median")
        else:
            ax_wins.set_yticklabels([])
        for idx, row in enumerate(sub.itertuples()):
            ax.text(
                idx,
                row.mean_ate_rmse_associated + 0.006,
                f"{row.mean_ate_rmse_associated:.3f}",
                ha="center",
                va="bottom",
                fontsize=FONT_ANNOTATION,
            )
    axes[0].set_ylabel("mean ATE RMSE associated")
    fig.suptitle("Single-pass Ratio Ablation Table", fontsize=FONT_SUPTITLE, y=0.98)
    save_png(fig, VAR_DIR / "ratio_ablation_by_architecture.png", dpi=180)
    plt.close(fig)


def read_repeat_metric_frames() -> pd.DataFrame:
    rows = []
    sources = [
        (
            BEST_MULTISCALE_LABEL,
            HANDOFF_ROOT / "RESULTS/final_dual_finalists_5x/repeats" / LEGACY_WINNER_REPEAT_ID,
        ),
        ("Native DPVO", FOCUS_DPVO_5X_ROOT / "dpvo_native"),
        ("Focus071", FOCUS_DPVO_5X_ROOT / "focus071_best"),
    ]
    for method, root in sources:
        for repeat in range(1, 6):
            path = root / f"repeat_{repeat:02d}" / "dpvo_style_metrics_summary.csv"
            frame = pd.read_csv(path)
            frame["method"] = method
            frame["repeat"] = repeat
            rows.append(frame[["sequence", "method", "repeat", "ate_rmse_associated"]])
    return pd.concat(rows, ignore_index=True)


def make_repeat_sequence_std_heatmap_for_methods(
    pivot: pd.DataFrame,
    methods: list[str],
    filename: str,
) -> None:
    plot = pivot[methods].copy()
    plot = plot.sort_values(BEST_MULTISCALE_LABEL, ascending=False)
    plot.to_csv(DATA_DIR / f"{Path(filename).stem}.csv")

    fig_h = max(9.5, 0.32 * len(plot))
    fig, ax = plt.subplots(figsize=(9.0, fig_h), constrained_layout=True)
    im = ax.imshow(plot.to_numpy(float), aspect="auto", cmap="YlOrRd")
    ax.set_xticks(np.arange(plot.shape[1]), plot.columns)
    ax.set_yticks(np.arange(plot.shape[0]), plot.index, fontsize=FONT_TICK)
    ax.set_title("Per-sequence Repeat Standard Deviation")
    cbar = fig.colorbar(im, ax=ax, shrink=0.72)
    cbar.set_label("std of ATE over five repeats")
    save_png(fig, VAR_DIR / filename, dpi=180)
    plt.close(fig)


def make_repeat_sequence_std_heatmaps() -> None:
    data = read_repeat_metric_frames()
    pivot = data.pivot_table(
        index="sequence",
        columns="method",
        values="ate_rmse_associated",
        aggfunc="std",
    )
    pivot.to_csv(DATA_DIR / "per_sequence_repeat_std_all_methods.csv")
    make_repeat_sequence_std_heatmap_for_methods(
        pivot,
        ["Native DPVO", "Focus071", BEST_MULTISCALE_LABEL],
        "per_sequence_repeat_std_focus071_multiscale_dpvo.png",
    )
    make_repeat_sequence_std_heatmap_for_methods(
        pivot,
        ["Native DPVO", BEST_MULTISCALE_LABEL],
        "per_sequence_repeat_std_multiscale_dpvo.png",
    )


def main() -> None:
    configure_plot_style()
    ensure_dirs()
    clean_regenerated_stats()
    patch_sets = load_patch_sets()
    make_patch_overlay_grid(patch_sets)
    make_separate_patch_overlays(patch_sets)
    make_patch_density_heatmaps(patch_sets)
    make_patch_diagnostic_summary()
    make_repeat_variance_plots()
    make_family_repeat_variance_plots()
    make_sequence_distribution_plots()
    make_ratio_ablation_plot()
    make_repeat_sequence_std_heatmaps()


if __name__ == "__main__":
    main()
