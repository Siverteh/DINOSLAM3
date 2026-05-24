from __future__ import annotations

from typing import Any
import math
import warnings

import torch
import torch.nn.functional as F

from .frontend import DinoDPVOFrameOutput


SOURCE_NATIVE = 0
SOURCE_DINO = 1


def pixel_to_dpvo_coords(pixel_xy: torch.Tensor, *, dpvo_res: int = 4) -> torch.Tensor:
    return (pixel_xy / float(dpvo_res)) - 0.5


def _repeat_to_budget(value: torch.Tensor, patch_budget: int) -> torch.Tensor:
    if value.shape[0] >= patch_budget:
        return value[:patch_budget]
    if value.shape[0] == 0:
        raise ValueError("Cannot pad an empty proposal set to the requested DPVO patch budget.")
    repeat = (patch_budget + value.shape[0] - 1) // value.shape[0]
    tiled = value.repeat((repeat,) + (1,) * (value.dim() - 1))
    return tiled[:patch_budget]


def _grid_cell_xy(
    pixel_xy: torch.Tensor,
    *,
    image_height: int,
    image_width: int,
    grid_rows: int,
    grid_cols: int,
) -> torch.Tensor:
    if pixel_xy.numel() == 0:
        return torch.zeros((0, 2), dtype=torch.long, device=pixel_xy.device)
    x = torch.floor(pixel_xy[:, 0] / max(float(image_width), 1.0) * float(grid_cols)).long()
    y = torch.floor(pixel_xy[:, 1] / max(float(image_height), 1.0) * float(grid_rows)).long()
    x = x.clamp(0, max(int(grid_cols) - 1, 0))
    y = y.clamp(0, max(int(grid_rows) - 1, 0))
    return torch.stack([x, y], dim=1)


def _proposal_utility(
    frame_output: DinoDPVOFrameOutput,
    *,
    static_score_weight: float,
) -> torch.Tensor:
    proposal = frame_output.proposal
    if proposal.patch_indices.numel() == 0:
        return torch.zeros((0,), dtype=torch.float32, device=proposal.pixel_xy.device)
    selector = torch.sigmoid(frame_output.selector_logits.reshape(-1)[proposal.patch_indices])
    staticness = torch.sigmoid(frame_output.staticness_logits.reshape(-1)[proposal.patch_indices])
    return selector * ((1.0 - float(static_score_weight)) + (float(static_score_weight) * staticness))


def _sample_native_reduced_coords(
    *,
    count: int,
    image_height: int,
    image_width: int,
    dpvo_res: int,
    device: torch.device,
) -> torch.Tensor:
    count = int(count)
    if count <= 0:
        return torch.zeros((0, 2), dtype=torch.float32, device=device)
    ht = max(int(image_height) // int(dpvo_res), 3)
    wt = max(int(image_width) // int(dpvo_res), 3)
    x = torch.randint(1, wt - 1, size=(count,), device=device)
    y = torch.randint(1, ht - 1, size=(count,), device=device)
    return torch.stack([x, y], dim=1).float()


def _build_metadata(
    *,
    pixel_xy: torch.Tensor,
    source_labels: torch.Tensor,
    utilities: torch.Tensor,
    image_height: int,
    image_width: int,
    grid_rows: int,
    grid_cols: int,
    repeated_patch_flags: torch.Tensor | None = None,
    unique_semantic_count_before_repeat: torch.Tensor | None = None,
    dedupe_radius_used: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    metadata = {
        "source_labels": source_labels.detach().to(dtype=torch.long, device="cpu"),
        "initial_pixel_xy": pixel_xy.detach().to(dtype=torch.float32, device="cpu"),
        "initial_cell_xy": _grid_cell_xy(
            pixel_xy,
            image_height=image_height,
            image_width=image_width,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
        ).detach().to(dtype=torch.long, device="cpu"),
        "selection_utility": utilities.detach().to(dtype=torch.float32, device="cpu"),
    }
    if repeated_patch_flags is not None:
        metadata["repeated_patch_flags"] = repeated_patch_flags.detach().to(dtype=torch.float32, device="cpu")
    if unique_semantic_count_before_repeat is not None:
        metadata["unique_semantic_count_before_repeat"] = unique_semantic_count_before_repeat.detach().to(dtype=torch.float32, device="cpu")
    if dedupe_radius_used is not None:
        metadata["dedupe_radius_used"] = dedupe_radius_used.detach().to(dtype=torch.float32, device="cpu")
    return metadata


def _fill_with_native(
    *,
    reduced_coords: list[torch.Tensor],
    pixel_coords: list[torch.Tensor],
    source_labels: list[torch.Tensor],
    utilities: list[torch.Tensor],
    quality_list: list[torch.Tensor],
    descriptor_biases: list[torch.Tensor] | None,
    gmap_descriptor_biases: list[torch.Tensor] | None,
    missing: int,
    image_height: int,
    image_width: int,
    dpvo_res: int,
    imap_dim: int | None,
    gmap_dim: int | None,
    device: torch.device,
) -> None:
    if int(missing) <= 0:
        return
    native_reduced = _sample_native_reduced_coords(
        count=int(missing),
        image_height=image_height,
        image_width=image_width,
        dpvo_res=dpvo_res,
        device=device,
    )
    native_pixel = (native_reduced + 0.5) * float(dpvo_res)
    reduced_coords.append(native_reduced)
    pixel_coords.append(native_pixel)
    source_labels.append(torch.full((native_reduced.shape[0],), SOURCE_NATIVE, dtype=torch.long, device=device))
    utilities.append(torch.ones((native_reduced.shape[0],), dtype=torch.float32, device=device))
    quality_list.append(torch.ones((native_reduced.shape[0], 1), dtype=torch.float32, device=device))
    if descriptor_biases is not None and imap_dim is not None:
        descriptor_biases.append(torch.zeros((native_reduced.shape[0], imap_dim), dtype=torch.float32, device=device))
    if gmap_descriptor_biases is not None and gmap_dim is not None:
        gmap_descriptor_biases.append(torch.zeros((native_reduced.shape[0], gmap_dim), dtype=torch.float32, device=device))


def _normalize_rows(value: torch.Tensor) -> torch.Tensor:
    if value.numel() == 0:
        return value
    return F.normalize(value.float(), dim=-1, eps=1e-6)


def _weighted_random_order(
    utility: torch.Tensor,
    *,
    indices: list[int] | None = None,
) -> list[int]:
    if utility.numel() == 0:
        return []
    device = utility.device
    if indices is None:
        pool = torch.arange(utility.shape[0], device=device, dtype=torch.long)
    else:
        if not indices:
            return []
        pool = torch.as_tensor(indices, device=device, dtype=torch.long)
    weights = utility[pool].float().clamp_min(1e-6)
    if not torch.isfinite(weights).all().item():
        weights = torch.ones_like(weights)
    order = torch.multinomial(weights, num_samples=int(pool.shape[0]), replacement=False)
    return pool[order].tolist()


def _take_optional_rows(value: torch.Tensor | None, indices: torch.Tensor) -> torch.Tensor | None:
    if value is None:
        return None
    return value[indices]


def _quality_insert_mask(quality: torch.Tensor, *, threshold: float) -> torch.Tensor:
    if quality.numel() == 0:
        return torch.zeros((0,), dtype=torch.bool, device=quality.device)
    if float(threshold) <= 0.0:
        return torch.ones((quality.shape[0],), dtype=torch.bool, device=quality.device)
    return quality >= float(threshold)


def _process_external_quality(
    quality: torch.Tensor,
    *,
    config: dict[str, Any],
) -> torch.Tensor:
    if quality.numel() == 0:
        return quality.reshape(-1, 1)
    raw = quality.float().clamp(0.0, 1.0)
    mode = str(config.get("quality_mode", "none")).lower()
    smoothing = float(config.get("quality_smoothing", 0.0))
    if smoothing > 0.0:
        smoothing = max(0.0, min(1.0, smoothing))
        raw = ((1.0 - smoothing) * raw) + (smoothing * raw.mean())
    power = float(config.get("quality_edge_power", 1.0))
    processed = raw.clamp_min(1e-6).pow(max(power, 1e-6))
    if mode == "hard_gate":
        ba_threshold = float(config.get("quality_ba_threshold", 0.0))
        if ba_threshold > 0.0:
            processed = torch.where(raw >= ba_threshold, processed, torch.zeros_like(processed))
    return processed[:, None]


def _semantic_candidate_state(
    frame_output: DinoDPVOFrameOutput,
    *,
    static_score_weight: float,
    dpvo_res: int,
    image_height: int,
    image_width: int,
    grid_rows: int,
    grid_cols: int,
) -> dict[str, torch.Tensor | None]:
    proposal = frame_output.proposal
    pixel_xy = proposal.pixel_xy
    quality = frame_output.qualities
    return {
        "pixel_xy": pixel_xy,
        "coarse_pixel_xy": proposal.coarse_pixel_xy,
        "reduced_xy": pixel_to_dpvo_coords(pixel_xy, dpvo_res=dpvo_res),
        "utility": _proposal_utility(frame_output, static_score_weight=static_score_weight),
        "cell_xy": _grid_cell_xy(
            pixel_xy,
            image_height=image_height,
            image_width=image_width,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
        ),
        "quality": quality,
        "descriptor": proposal.descriptors,
        "descriptor_bias": frame_output.descriptor_bias,
        "gmap_descriptor_bias": frame_output.gmap_descriptor_bias,
        "register_context": frame_output.register_context,
    }


def _register_anchor_bonus(
    *,
    state: dict[str, torch.Tensor | None],
    config: dict[str, Any],
) -> torch.Tensor:
    target = str(config.get("register_context_target", "fused") or "fused").lower()
    if target not in {"anchor_refresh", "both"}:
        descriptor = state.get("descriptor")
        if descriptor is None:
            return torch.zeros((0,), dtype=torch.float32)
        return torch.zeros((descriptor.shape[0],), dtype=torch.float32, device=descriptor.device)

    descriptor = state.get("descriptor")
    register_context = state.get("register_context")
    scale = float(config.get("register_context_scale", 0.0) or 0.0)
    if (
        descriptor is None
        or register_context is None
        or descriptor.numel() == 0
        or register_context.numel() == 0
        or scale <= 0.0
    ):
        if descriptor is None:
            return torch.zeros((0,), dtype=torch.float32)
        return torch.zeros((descriptor.shape[0],), dtype=torch.float32, device=descriptor.device)

    context = register_context.reshape(-1)
    if context.shape[0] != descriptor.shape[1]:
        return torch.zeros((descriptor.shape[0],), dtype=torch.float32, device=descriptor.device)

    descriptor_n = _normalize_rows(descriptor)
    context_n = F.normalize(context.float(), dim=0, eps=1e-6)
    bonus = (descriptor_n @ context_n).clamp(min=0.0)
    # Keep the register-token signal as a mild preference so anchors still respect
    # the learned selector/staticness scores instead of replacing them.
    return 0.25 * float(scale) * bonus


def _rank_semantic_candidates(
    *,
    utility: torch.Tensor,
    cell_xy: torch.Tensor,
    max_per_cell: int,
) -> list[int]:
    ranked = torch.argsort(utility, descending=True)
    per_cell: dict[tuple[int, int], int] = {}
    chosen: list[int] = []
    fallback: list[int] = []
    for idx in ranked.tolist():
        cell = tuple(int(v) for v in cell_xy[idx].tolist())
        count = per_cell.get(cell, 0)
        if count < int(max_per_cell):
            chosen.append(idx)
            per_cell[cell] = count + 1
        else:
            fallback.append(idx)
    return chosen + fallback


def _multiscale_reorder(
    ordered_indices: list[int],
    *,
    pixel_xy: torch.Tensor,
    utility: torch.Tensor,
    image_height: int,
    image_width: int,
    region_rows: int,
    region_cols: int,
    points_per_region: int,
) -> list[int]:
    if not ordered_indices or int(region_rows) <= 0 or int(region_cols) <= 0 or int(points_per_region) <= 0:
        return ordered_indices
    region_xy = _grid_cell_xy(
        pixel_xy,
        image_height=image_height,
        image_width=image_width,
        grid_rows=int(region_rows),
        grid_cols=int(region_cols),
    )
    region_members: dict[tuple[int, int], list[int]] = {}
    region_score: dict[tuple[int, int], float] = {}
    for idx in ordered_indices:
        region = tuple(int(v) for v in region_xy[idx].tolist())
        region_members.setdefault(region, []).append(int(idx))
        region_score[region] = max(region_score.get(region, -math.inf), float(utility[idx].item()))
    region_order = sorted(region_members.keys(), key=lambda key: region_score[key], reverse=True)
    used: set[int] = set()
    reordered: list[int] = []
    per_region = max(1, int(points_per_region))
    for region in region_order:
        member_indices = sorted(region_members[region], key=lambda idx: float(utility[idx].item()), reverse=True)
        for idx in member_indices[:per_region]:
            if idx not in used:
                reordered.append(idx)
                used.add(idx)
    for idx in ordered_indices:
        if idx not in used:
            reordered.append(int(idx))
    return reordered


def _select_semantic_indices(
    *,
    ordered_indices: list[int],
    pixel_xy: torch.Tensor,
    patch_target: int,
    dedupe_radius_px: float,
    native_pixel_xy: torch.Tensor | None = None,
) -> list[int]:
    keep: list[int] = []
    native_xy = native_pixel_xy if native_pixel_xy is not None else pixel_xy.new_zeros((0, 2))
    for idx in ordered_indices:
        if len(keep) >= int(patch_target):
            break
        candidate = pixel_xy[idx : idx + 1]
        if native_xy.numel() > 0:
            dist = torch.cdist(candidate, native_xy)
            if bool((dist <= float(dedupe_radius_px)).any().item()):
                continue
        if keep:
            prior = pixel_xy[torch.as_tensor(keep, device=pixel_xy.device, dtype=torch.long)]
            dist = torch.cdist(candidate, prior)
            if bool((dist <= float(dedupe_radius_px)).any().item()):
                continue
        keep.append(int(idx))
    return keep


def _adaptive_select_semantic_indices(
    *,
    ordered_indices: list[int],
    pixel_xy: torch.Tensor,
    patch_target: int,
    dedupe_schedule_px: list[float],
    native_pixel_xy: torch.Tensor | None = None,
) -> tuple[list[int], float]:
    if not dedupe_schedule_px:
        dedupe_schedule_px = [0.0]

    best_keep: list[int] = []
    best_radius = float(dedupe_schedule_px[-1])
    for radius in dedupe_schedule_px:
        keep = _select_semantic_indices(
            ordered_indices=ordered_indices,
            pixel_xy=pixel_xy,
            patch_target=int(patch_target),
            dedupe_radius_px=float(radius),
            native_pixel_xy=native_pixel_xy,
        )
        if len(keep) >= len(best_keep):
            best_keep = keep
            best_radius = float(radius)
        if len(keep) >= int(patch_target):
            return keep, float(radius)
    return best_keep, best_radius


def _micro_patch_offsets(
    *,
    count: int,
    pattern: str,
    spread_px: float,
    device: torch.device,
) -> torch.Tensor:
    count = max(1, int(count))
    if count == 1:
        return torch.zeros((1, 2), dtype=torch.float32, device=device)
    spread = float(spread_px)
    mode = str(pattern or "grid").lower()
    if mode == "cross":
        candidates = [
            (0.0, 0.0),
            (-1.0, 0.0),
            (1.0, 0.0),
            (0.0, -1.0),
            (0.0, 1.0),
            (-1.0, -1.0),
            (1.0, -1.0),
            (-1.0, 1.0),
            (1.0, 1.0),
            (-2.0, 0.0),
            (2.0, 0.0),
            (0.0, -2.0),
            (0.0, 2.0),
        ]
        coords = candidates[:count]
        return torch.tensor(coords, dtype=torch.float32, device=device) * spread

    side = max(2, int(math.ceil(math.sqrt(count))))
    xs = torch.linspace(-1.0, 1.0, steps=side, device=device)
    ys = torch.linspace(-1.0, 1.0, steps=side, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    offsets = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
    score = offsets.square().sum(dim=1)
    order = torch.argsort(score, descending=True)
    offsets = offsets[order][:count]
    return offsets * spread


def _expand_micro_regions(
    *,
    state: dict[str, torch.Tensor | None],
    keep_idx: torch.Tensor,
    patch_target: int,
    image_height: int,
    image_width: int,
    config: dict[str, Any],
) -> dict[str, torch.Tensor | None]:
    count = max(1, int(config.get("micro_patch_count", 1)))
    if count <= 1 or keep_idx.numel() == 0:
        return {
            "pixel_xy": state["pixel_xy"][keep_idx],  # type: ignore[index]
            "utility": state["utility"][keep_idx],  # type: ignore[index]
            "quality": state["quality"][keep_idx],  # type: ignore[index]
            "descriptor_bias": _take_optional_rows(state["descriptor_bias"], keep_idx),  # type: ignore[arg-type]
            "gmap_descriptor_bias": _take_optional_rows(state["gmap_descriptor_bias"], keep_idx),  # type: ignore[arg-type]
        }
    center_mode = str(config.get("micro_patch_center_mode", "refined")).lower()
    base_pixel = state["coarse_pixel_xy"] if center_mode == "coarse" else state["pixel_xy"]
    assert base_pixel is not None
    selected_pixel = base_pixel[keep_idx]
    offsets = _micro_patch_offsets(
        count=count,
        pattern=str(config.get("micro_patch_pattern", "grid")),
        spread_px=float(config.get("micro_patch_spread_px", 4.0)),
        device=selected_pixel.device,
    )
    expanded_pixel = (selected_pixel[:, None, :] + offsets[None, :, :]).reshape(-1, 2)
    expanded_pixel = torch.stack(
        [
            expanded_pixel[:, 0].clamp(0.0, float(image_width - 1)),
            expanded_pixel[:, 1].clamp(0.0, float(image_height - 1)),
        ],
        dim=1,
    )
    utility = _repeat_to_budget(state["utility"][keep_idx].repeat_interleave(count), int(patch_target))  # type: ignore[index]
    quality = _repeat_to_budget(state["quality"][keep_idx].repeat_interleave(count), int(patch_target))  # type: ignore[index]
    out: dict[str, torch.Tensor | None] = {
        "pixel_xy": _repeat_to_budget(expanded_pixel, int(patch_target)),
        "utility": utility,
        "quality": quality,
        "descriptor_bias": None,
        "gmap_descriptor_bias": None,
    }
    descriptor_bias = state["descriptor_bias"]
    if descriptor_bias is not None:
        out["descriptor_bias"] = _repeat_to_budget(descriptor_bias[keep_idx].repeat_interleave(count, dim=0), int(patch_target))
    gmap_descriptor_bias = state["gmap_descriptor_bias"]
    if gmap_descriptor_bias is not None:
        out["gmap_descriptor_bias"] = _repeat_to_budget(gmap_descriptor_bias[keep_idx].repeat_interleave(count, dim=0), int(patch_target))
    return out


def _ordered_semantic_candidates(
    *,
    state: dict[str, torch.Tensor | None],
    image_height: int,
    image_width: int,
    max_per_cell: int,
    config: dict[str, Any],
) -> list[int]:
    utility = state["utility"]
    cell_xy = state["cell_xy"]
    assert utility is not None and cell_xy is not None
    ordered = _rank_semantic_candidates(
        utility=utility,
        cell_xy=cell_xy,
        max_per_cell=max_per_cell,
    )
    insert_threshold = float(config.get("quality_insert_threshold", 0.0))
    mask = _quality_insert_mask(state["quality"], threshold=insert_threshold)  # type: ignore[arg-type]
    if mask.numel() > 0 and not bool(mask.all().item()):
        allowed = {idx for idx in torch.nonzero(mask, as_tuple=False).reshape(-1).tolist()}
        ordered = [idx for idx in ordered if idx in allowed]
    ordered = _multiscale_reorder(
        ordered,
        pixel_xy=state["pixel_xy"],  # type: ignore[arg-type]
        utility=utility,
        image_height=image_height,
        image_width=image_width,
        region_rows=int(config.get("multiscale_region_rows", 0)),
        region_cols=int(config.get("multiscale_region_cols", 0)),
        points_per_region=int(config.get("multiscale_points_per_region", 0)),
    )
    return ordered


def _patch_sampler_mode(config: dict[str, Any]) -> str:
    return str(config.get("patch_sampler_mode", "semantic_grid") or "semantic_grid").strip().lower()


def _sampler_ordered_candidates(
    *,
    state: dict[str, torch.Tensor | None],
    ordered: list[int],
    config: dict[str, Any],
) -> list[int]:
    utility = state["utility"]
    cell_xy = state["cell_xy"]
    assert utility is not None and cell_xy is not None
    mode = _patch_sampler_mode(config)
    if mode == "semantic_grid":
        return ordered
    if mode == "stratified_random":
        grouped: dict[tuple[int, int], list[int]] = {}
        for idx in ordered:
            cell = tuple(int(v) for v in cell_xy[idx].tolist())
            grouped.setdefault(cell, []).append(int(idx))
        head: list[int] = []
        for cell in sorted(grouped):
            picks = _weighted_random_order(utility, indices=grouped[cell])
            if picks:
                head.append(int(picks[0]))
        used = set(head)
        tail = _weighted_random_order(utility, indices=[idx for idx in ordered if idx not in used])
        return head + [idx for idx in tail if idx not in used]
    if mode == "geometry_semantic_mix":
        random_order = _weighted_random_order(utility, indices=ordered)
        mixed: list[int] = []
        used: set[int] = set()
        max_len = max(len(ordered), len(random_order))
        for pos in range(max_len):
            if pos < len(ordered):
                idx = int(ordered[pos])
                if idx not in used:
                    mixed.append(idx)
                    used.add(idx)
            if pos < len(random_order):
                idx = int(random_order[pos])
                if idx not in used:
                    mixed.append(idx)
                    used.add(idx)
        return mixed
    if mode == "random_backfill":
        head = ordered[: max(1, int(math.ceil(len(ordered) * 0.6)))]
        used = set(head)
        tail = _weighted_random_order(utility, indices=[idx for idx in ordered if idx not in used])
        return head + [idx for idx in tail if idx not in used]
    return ordered


def _prepend_keepalive_indices(
    *,
    ordered: list[int],
    state: dict[str, torch.Tensor | None],
    runtime_state: dict[str, Any] | None,
    config: dict[str, Any],
) -> list[int]:
    keep_alive_topk = max(0, int(config.get("keep_alive_topk", 0) or 0))
    if keep_alive_topk <= 0 or runtime_state is None:
        return ordered
    memory = runtime_state.get("pure_keepalive")
    if not isinstance(memory, dict):
        return ordered
    prev_pixel = memory.get("pixel_xy")
    if prev_pixel is None or prev_pixel.numel() == 0:
        return ordered
    pixel_xy = state["pixel_xy"]
    utility = state["utility"]
    assert pixel_xy is not None and utility is not None
    radius = float(config.get("anchor_match_radius_px", config.get("dedupe_radius_px", 8.0)))
    bonus = float(config.get("survival_bias_scale", 0.0) or 0.0)
    dist = torch.cdist(prev_pixel.float(), pixel_xy.float())
    matched: list[int] = []
    used: set[int] = set()
    for row_idx in range(int(dist.shape[0])):
        candidate_order = torch.argsort(dist[row_idx], descending=False)
        for idx in candidate_order.tolist():
            if idx in used:
                continue
            if float(dist[row_idx, idx].item()) > radius:
                break
            matched.append(int(idx))
            used.add(int(idx))
            break
        if len(matched) >= keep_alive_topk:
            break
    if not matched:
        return ordered
    matched = sorted(
        matched,
        key=lambda idx: float(utility[idx].item()) + bonus,
        reverse=True,
    )[:keep_alive_topk]
    matched_set = set(matched)
    return matched + [idx for idx in ordered if idx not in matched_set]


def _update_keepalive_state(
    *,
    runtime_state: dict[str, Any] | None,
    pixels: torch.Tensor,
    utility: torch.Tensor,
    config: dict[str, Any],
) -> None:
    keep_alive_topk = max(0, int(config.get("keep_alive_topk", 0) or 0))
    if runtime_state is None or keep_alive_topk <= 0 or pixels.numel() == 0:
        return
    ranking = torch.argsort(utility.float(), descending=True)[:keep_alive_topk]
    runtime_state["pure_keepalive"] = {
        "pixel_xy": pixels[ranking].detach(),
        "utility": utility[ranking].detach(),
    }


def _assemble_patch_input(
    *,
    coords: torch.Tensor,
    pixels: torch.Tensor,
    qualities: torch.Tensor,
    sources: torch.Tensor,
    utility: torch.Tensor,
    image_height: int,
    image_width: int,
    grid_rows: int,
    grid_cols: int,
    descriptor_bias: torch.Tensor | None = None,
    gmap_descriptor_bias: torch.Tensor | None = None,
    repeated_flags: torch.Tensor | None = None,
    unique_semantic_count_before_repeat: torch.Tensor | None = None,
    dedupe_radius_used: torch.Tensor | None = None,
) -> dict[str, Any]:
    metadata = _build_metadata(
        pixel_xy=pixels,
        source_labels=sources,
        utilities=utility,
        image_height=image_height,
        image_width=image_width,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        repeated_patch_flags=repeated_flags,
        unique_semantic_count_before_repeat=unique_semantic_count_before_repeat,
        dedupe_radius_used=dedupe_radius_used,
    )
    patch_input: dict[str, Any] = {
        "external_coords": coords.unsqueeze(0).detach(),
        "external_quality": qualities.unsqueeze(0).detach(),
        "patch_metadata": metadata,
    }
    if descriptor_bias is not None:
        patch_input["external_descriptor_bias"] = descriptor_bias.unsqueeze(0).detach()
    if gmap_descriptor_bias is not None:
        patch_input["external_gmap_bias"] = gmap_descriptor_bias.unsqueeze(0).detach()
    return patch_input


def _semantic_full_cover_patch_input(
    frame_output: DinoDPVOFrameOutput,
    *,
    patch_budget: int,
    image_height: int,
    image_width: int,
    dpvo_res: int,
    static_score_weight: float,
    grid_rows: int,
    grid_cols: int,
    max_semantic_per_cell: int,
    dedupe_radius_px: float,
    dedupe_schedule_px: list[float] | None,
    semantic_backfill_source: str,
    config: dict[str, Any],
    runtime_state: dict[str, Any] | None,
) -> dict[str, Any]:
    device = frame_output.proposal.pixel_xy.device
    state = _semantic_candidate_state(
        frame_output,
        static_score_weight=static_score_weight,
        dpvo_res=dpvo_res,
        image_height=image_height,
        image_width=image_width,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
    )
    micro_count = max(1, int(config.get("micro_patch_count", 1)))
    region_target = max(1, int(math.ceil(float(patch_budget) / float(micro_count))))
    ordered = _ordered_semantic_candidates(
        state=state,
        image_height=image_height,
        image_width=image_width,
        max_per_cell=max_semantic_per_cell,
        config=config,
    )
    ordered = _sampler_ordered_candidates(state=state, ordered=ordered, config=config)
    ordered = _prepend_keepalive_indices(
        ordered=ordered,
        state=state,
        runtime_state=runtime_state,
        config=config,
    )
    radius_schedule = [float(v) for v in (dedupe_schedule_px or [float(dedupe_radius_px)])]
    keep, used_dedupe_radius = _adaptive_select_semantic_indices(
        ordered_indices=ordered,
        pixel_xy=state["pixel_xy"],  # type: ignore[arg-type]
        patch_target=int(region_target),
        dedupe_schedule_px=radius_schedule,
        native_pixel_xy=None,
    )
    unique_count_before_repeat = len(keep)

    if keep:
        keep_idx = torch.as_tensor(keep, device=device, dtype=torch.long)
        expanded = _expand_micro_regions(
            state=state,
            keep_idx=keep_idx,
            patch_target=int(patch_budget),
            image_height=image_height,
            image_width=image_width,
            config=config,
        )
        pixels = expanded["pixel_xy"]
        utility = expanded["utility"]
        descriptor_bias = expanded["descriptor_bias"]
        gmap_descriptor_bias = expanded["gmap_descriptor_bias"]
        assert pixels is not None and utility is not None
        coords = pixel_to_dpvo_coords(pixels, dpvo_res=dpvo_res)
        quality = _process_external_quality(expanded["quality"], config=config)
    else:
        coords = torch.zeros((0, 2), dtype=torch.float32, device=device)
        pixels = torch.zeros((0, 2), dtype=torch.float32, device=device)
        utility = torch.zeros((0,), dtype=torch.float32, device=device)
        quality = torch.zeros((0, 1), dtype=torch.float32, device=device)
        descriptor_bias = None
        gmap_descriptor_bias = None
    repeated_flags = torch.zeros((coords.shape[0],), dtype=torch.float32, device=device)

    source_labels = torch.full((coords.shape[0],), SOURCE_DINO, dtype=torch.long, device=device)
    if coords.shape[0] < int(patch_budget):
        missing = int(patch_budget) - int(coords.shape[0])
        backfill = str(semantic_backfill_source).lower()
        if backfill == "native":
            reduced_coords = [coords]
            pixel_coords = [pixels]
            sources = [source_labels]
            utilities = [utility]
            quality_list = [quality]
            descriptor_biases = [descriptor_bias] if descriptor_bias is not None else None
            gmap_biases = [gmap_descriptor_bias] if gmap_descriptor_bias is not None else None
            _fill_with_native(
                reduced_coords=reduced_coords,
                pixel_coords=pixel_coords,
                source_labels=sources,
                utilities=utilities,
                quality_list=quality_list,
                descriptor_biases=descriptor_biases,
                gmap_descriptor_biases=gmap_biases,
                missing=missing,
                image_height=image_height,
                image_width=image_width,
                dpvo_res=dpvo_res,
                imap_dim=None if descriptor_bias is None else int(descriptor_bias.shape[1]),
                gmap_dim=None if gmap_descriptor_bias is None else int(gmap_descriptor_bias.shape[1]),
                device=device,
            )
            coords = torch.cat(reduced_coords, dim=0)
            pixels = torch.cat(pixel_coords, dim=0)
            source_labels = torch.cat(sources, dim=0)
            utility = torch.cat(utilities, dim=0)
            quality = torch.cat(quality_list, dim=0)
            if descriptor_biases is not None:
                descriptor_bias = torch.cat(descriptor_biases, dim=0)
            if gmap_biases is not None:
                gmap_descriptor_bias = torch.cat(gmap_biases, dim=0)
            repeated_flags = torch.zeros((coords.shape[0],), dtype=torch.float32, device=device)
        else:
            fallback_coords = coords
            fallback_pixels = pixels
            fallback_utility = utility
            fallback_quality = quality
            fallback_descriptor_bias = descriptor_bias
            fallback_gmap_descriptor_bias = gmap_descriptor_bias
            if fallback_coords.shape[0] == 0 and state["reduced_xy"].shape[0] > 0:  # type: ignore[index]
                warnings.warn(
                    "Semantic full-cover selection exhausted unique candidates before selecting any patches; "
                    "falling back to top semantic proposals.",
                    stacklevel=2,
                )
                fallback_pixels = state["pixel_xy"]  # type: ignore[assignment]
                fallback_coords = state["reduced_xy"]  # type: ignore[assignment]
                fallback_utility = state["utility"]  # type: ignore[assignment]
                fallback_quality = _process_external_quality(state["quality"], config=config)  # type: ignore[arg-type]
                fallback_descriptor_bias = state["descriptor_bias"]  # type: ignore[assignment]
                fallback_gmap_descriptor_bias = state["gmap_descriptor_bias"]  # type: ignore[assignment]
            if fallback_coords.shape[0] == 0:
                warnings.warn(
                    "Semantic full-cover mode produced no semantic candidates; falling back to native padding.",
                    stacklevel=2,
                )
                reduced_coords = [coords]
                pixel_coords = [pixels]
                sources = [source_labels]
                utilities = [utility]
                quality_list = [quality]
                _fill_with_native(
                    reduced_coords=reduced_coords,
                    pixel_coords=pixel_coords,
                    source_labels=sources,
                    utilities=utilities,
                    quality_list=quality_list,
                    descriptor_biases=None,
                    gmap_descriptor_biases=None,
                    missing=missing,
                    image_height=image_height,
                    image_width=image_width,
                    dpvo_res=dpvo_res,
                    imap_dim=None,
                    gmap_dim=None,
                    device=device,
                )
                coords = torch.cat(reduced_coords, dim=0)
                pixels = torch.cat(pixel_coords, dim=0)
                source_labels = torch.cat(sources, dim=0)
                utility = torch.cat(utilities, dim=0)
                quality = torch.cat(quality_list, dim=0)
                repeated_flags = torch.zeros((coords.shape[0],), dtype=torch.float32, device=device)
            else:
                unique_count = int(fallback_coords.shape[0])
                if _patch_sampler_mode(config) == "random_backfill":
                    candidate_order = _weighted_random_order(state["utility"])  # type: ignore[arg-type]
                    sampled_idx = torch.as_tensor(
                        candidate_order[: max(int(patch_budget), unique_count)],
                        device=device,
                        dtype=torch.long,
                    )
                    all_pixels = state["pixel_xy"][sampled_idx]  # type: ignore[index]
                    all_coords = pixel_to_dpvo_coords(all_pixels, dpvo_res=dpvo_res)
                    all_utility = state["utility"][sampled_idx]  # type: ignore[index]
                    all_quality = _process_external_quality(state["quality"][sampled_idx], config=config)  # type: ignore[index]
                    coords = _repeat_to_budget(all_coords, int(patch_budget))
                    pixels = _repeat_to_budget(all_pixels, int(patch_budget))
                    utility = _repeat_to_budget(all_utility, int(patch_budget))
                    quality = _repeat_to_budget(all_quality, int(patch_budget))
                    source_labels = torch.full((coords.shape[0],), SOURCE_DINO, dtype=torch.long, device=device)
                    repeated_flags = torch.zeros((int(patch_budget),), dtype=torch.float32, device=device)
                    if unique_count < int(patch_budget):
                        repeated_flags[unique_count:] = 1.0
                    descriptor_source = state["descriptor_bias"]
                    if descriptor_source is not None:
                        descriptor_bias = _repeat_to_budget(descriptor_source[sampled_idx], int(patch_budget))
                    gmap_source = state["gmap_descriptor_bias"]
                    if gmap_source is not None:
                        gmap_descriptor_bias = _repeat_to_budget(gmap_source[sampled_idx], int(patch_budget))
                else:
                    warnings.warn(
                        f"Semantic full-cover mode found only {int(coords.shape[0])} unique semantic patches; "
                        "repeating semantic proposals to fill DPVO budget.",
                        stacklevel=2,
                    )
                    coords = _repeat_to_budget(fallback_coords, int(patch_budget))
                    pixels = _repeat_to_budget(fallback_pixels, int(patch_budget))
                    utility = _repeat_to_budget(fallback_utility, int(patch_budget))
                    quality = _repeat_to_budget(fallback_quality, int(patch_budget))
                    source_labels = torch.full((coords.shape[0],), SOURCE_DINO, dtype=torch.long, device=device)
                    repeated_flags = torch.zeros((int(patch_budget),), dtype=torch.float32, device=device)
                    if unique_count < int(patch_budget):
                        repeated_flags[unique_count:] = 1.0
                    if fallback_descriptor_bias is not None:
                        descriptor_bias = _repeat_to_budget(fallback_descriptor_bias, int(patch_budget))
                    if fallback_gmap_descriptor_bias is not None:
                        gmap_descriptor_bias = _repeat_to_budget(fallback_gmap_descriptor_bias, int(patch_budget))

    if coords.shape[0] > int(patch_budget):
        coords = coords[: int(patch_budget)]
        pixels = pixels[: int(patch_budget)]
        source_labels = source_labels[: int(patch_budget)]
        utility = utility[: int(patch_budget)]
        quality = quality[: int(patch_budget)]
        repeated_flags = repeated_flags[: int(patch_budget)]
        if descriptor_bias is not None:
            descriptor_bias = descriptor_bias[: int(patch_budget)]
        if gmap_descriptor_bias is not None:
            gmap_descriptor_bias = gmap_descriptor_bias[: int(patch_budget)]

    unique_count_tensor = torch.full(
        (coords.shape[0],),
        float(unique_count_before_repeat),
        dtype=torch.float32,
        device=device,
    )
    dedupe_radius_tensor = torch.full(
        (coords.shape[0],),
        float(used_dedupe_radius if math.isfinite(float(used_dedupe_radius)) else 0.0),
        dtype=torch.float32,
        device=device,
    )
    _update_keepalive_state(
        runtime_state=runtime_state,
        pixels=pixels,
        utility=utility,
        config=config,
    )
    return _assemble_patch_input(
        coords=coords,
        pixels=pixels,
        qualities=quality,
        sources=source_labels,
        utility=utility,
        image_height=image_height,
        image_width=image_width,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        descriptor_bias=descriptor_bias,
        gmap_descriptor_bias=gmap_descriptor_bias,
        repeated_flags=repeated_flags,
        unique_semantic_count_before_repeat=unique_count_tensor,
        dedupe_radius_used=dedupe_radius_tensor,
    )


def _update_anchor_state(
    *,
    runtime_state: dict[str, Any] | None,
    state: dict[str, torch.Tensor | None],
    ordered: list[int],
    anchor_budget: int,
    config: dict[str, Any],
) -> torch.Tensor:
    device = state["pixel_xy"].device  # type: ignore[union-attr]
    anchor_budget = max(0, int(anchor_budget))
    if anchor_budget <= 0 or runtime_state is None:
        return torch.zeros((0,), dtype=torch.long, device=device)

    descriptor = _normalize_rows(state["descriptor"])  # type: ignore[arg-type]
    pixel_xy = state["pixel_xy"]  # type: ignore[assignment]
    utility = state["utility"]  # type: ignore[assignment]
    anchor_utility = utility + _register_anchor_bonus(state=state, config=config)
    if descriptor.numel() == 0:
        runtime_state["semantic_anchors"] = None
        return torch.zeros((0,), dtype=torch.long, device=device)

    anchors = runtime_state.get("semantic_anchors")
    prev_desc = None if anchors is None else anchors.get("descriptor")
    prev_pixel = None if anchors is None else anchors.get("pixel_xy")
    match_cosine = float(config.get("anchor_match_cosine", 0.75))
    match_radius = float(config.get("anchor_match_radius_px", 24.0))
    refresh_fraction = max(0.0, min(1.0, float(config.get("anchor_refresh_fraction", 0.25))))
    preserve_budget = max(0, int(anchor_budget - math.ceil(anchor_budget * refresh_fraction)))

    matched: list[int] = []
    used_current: set[int] = set()
    if (
        prev_desc is not None
        and prev_pixel is not None
        and prev_desc.numel() > 0
        and prev_pixel.numel() > 0
    ):
        sim = prev_desc @ descriptor.t()
        dist = torch.cdist(prev_pixel.float(), pixel_xy.float())
        for ai in range(sim.shape[0]):
            candidate_order = torch.argsort(sim[ai], descending=True)
            for idx in candidate_order.tolist():
                if idx in used_current:
                    continue
                if float(sim[ai, idx].item()) < match_cosine:
                    break
                if float(dist[ai, idx].item()) > match_radius:
                    continue
                matched.append(int(idx))
                used_current.add(int(idx))
                break
        if len(matched) > preserve_budget:
            matched = sorted(matched, key=lambda idx: float(anchor_utility[idx].item()), reverse=True)[:preserve_budget]
            used_current = set(matched)

    selection_order = sorted(ordered, key=lambda idx: float(anchor_utility[idx].item()), reverse=True)
    selected = list(matched)
    for idx in selection_order:
        if len(selected) >= anchor_budget:
            break
        if idx in used_current:
            continue
        selected.append(int(idx))
        used_current.add(int(idx))

    keep_idx = torch.as_tensor(selected[:anchor_budget], device=device, dtype=torch.long)
    runtime_state["semantic_anchors"] = {
        "descriptor": descriptor[keep_idx].detach(),
        "pixel_xy": pixel_xy[keep_idx].detach(),
    }
    runtime_state["anchor_age"] = 0
    return keep_idx


def _hybrid_patch_input(
    frame_output: DinoDPVOFrameOutput,
    *,
    patch_budget: int,
    image_height: int,
    image_width: int,
    dpvo_res: int,
    native_fraction: float,
    dino_fraction: float,
    static_score_weight: float,
    grid_rows: int,
    grid_cols: int,
    max_dino_per_cell: int,
    dedupe_radius_px: float,
    config: dict[str, Any],
    runtime_state: dict[str, Any] | None,
) -> dict[str, Any]:
    device = frame_output.proposal.pixel_xy.device
    patch_budget = int(patch_budget)
    anchor_budget = int(config.get("anchor_budget", 0) or 0)
    if anchor_budget > 0:
        dino_target = max(0, min(patch_budget, anchor_budget))
        native_target = max(0, patch_budget - dino_target)
    else:
        frac_sum = max(float(native_fraction) + float(dino_fraction), 1e-6)
        native_target = int(round(float(patch_budget) * float(native_fraction) / frac_sum))
        native_target = max(0, min(patch_budget, native_target))
        dino_target = max(0, patch_budget - native_target)

    native_reduced = _sample_native_reduced_coords(
        count=native_target,
        image_height=image_height,
        image_width=image_width,
        dpvo_res=dpvo_res,
        device=device,
    )
    native_pixel = (native_reduced + 0.5) * float(dpvo_res)

    state = _semantic_candidate_state(
        frame_output,
        static_score_weight=static_score_weight,
        dpvo_res=dpvo_res,
        image_height=image_height,
        image_width=image_width,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
    )
    micro_count = max(1, int(config.get("micro_patch_count", 1)))
    region_target = max(1, int(math.ceil(float(dino_target) / float(micro_count)))) if dino_target > 0 else 0
    ordered = _ordered_semantic_candidates(
        state=state,
        image_height=image_height,
        image_width=image_width,
        max_per_cell=max_dino_per_cell,
        config=config,
    )

    if anchor_budget > 0:
        keep_idx = _update_anchor_state(
            runtime_state=runtime_state,
            state=state,
            ordered=ordered,
            anchor_budget=region_target,
            config=config,
        )
    else:
        keep = _select_semantic_indices(
            ordered_indices=ordered,
            pixel_xy=state["pixel_xy"],  # type: ignore[arg-type]
            patch_target=region_target,
            dedupe_radius_px=dedupe_radius_px,
            native_pixel_xy=native_pixel,
        )
        keep_idx = torch.as_tensor(keep, device=device, dtype=torch.long)

    if keep_idx.numel() > 0 and dino_target > 0:
        expanded = _expand_micro_regions(
            state=state,
            keep_idx=keep_idx,
            patch_target=int(dino_target),
            image_height=image_height,
            image_width=image_width,
            config=config,
        )
        dino_pixel = expanded["pixel_xy"]
        dino_reduced = pixel_to_dpvo_coords(dino_pixel, dpvo_res=dpvo_res) if dino_pixel is not None else torch.zeros((0, 2), dtype=torch.float32, device=device)
        dino_utility = expanded["utility"]
        dino_quality = _process_external_quality(expanded["quality"], config=config)
        dino_descriptor_bias = expanded["descriptor_bias"]
        dino_gmap_descriptor_bias = expanded["gmap_descriptor_bias"]
    else:
        dino_reduced = torch.zeros((0, 2), dtype=torch.float32, device=device)
        dino_pixel = torch.zeros((0, 2), dtype=torch.float32, device=device)
        dino_utility = torch.zeros((0,), dtype=torch.float32, device=device)
        dino_quality = torch.zeros((0, 1), dtype=torch.float32, device=device)
        dino_descriptor_bias = None
        dino_gmap_descriptor_bias = None

    reduced_coords = [native_reduced, dino_reduced]
    pixel_coords = [native_pixel, dino_pixel]
    source_labels = [
        torch.full((native_reduced.shape[0],), SOURCE_NATIVE, dtype=torch.long, device=device),
        torch.full((dino_reduced.shape[0],), SOURCE_DINO, dtype=torch.long, device=device),
    ]
    utilities = [
        torch.ones((native_reduced.shape[0],), dtype=torch.float32, device=device),
        dino_utility,
    ]
    quality_list = [
        torch.ones((native_reduced.shape[0], 1), dtype=torch.float32, device=device),
        dino_quality,
    ]
    descriptor_biases = None
    gmap_biases = None
    if dino_descriptor_bias is not None:
        descriptor_biases = [
            torch.zeros((native_reduced.shape[0], dino_descriptor_bias.shape[1]), dtype=torch.float32, device=device),
            dino_descriptor_bias,
        ]
    if dino_gmap_descriptor_bias is not None:
        gmap_biases = [
            torch.zeros((native_reduced.shape[0], dino_gmap_descriptor_bias.shape[1]), dtype=torch.float32, device=device),
            dino_gmap_descriptor_bias,
        ]

    current = native_reduced.shape[0] + dino_reduced.shape[0]
    if current < patch_budget:
        _fill_with_native(
            reduced_coords=reduced_coords,
            pixel_coords=pixel_coords,
            source_labels=source_labels,
            utilities=utilities,
            quality_list=quality_list,
            descriptor_biases=descriptor_biases,
            gmap_descriptor_biases=gmap_biases,
            missing=patch_budget - current,
            image_height=image_height,
            image_width=image_width,
            dpvo_res=dpvo_res,
            imap_dim=None if dino_descriptor_bias is None else int(dino_descriptor_bias.shape[1]),
            gmap_dim=None if dino_gmap_descriptor_bias is None else int(dino_gmap_descriptor_bias.shape[1]),
            device=device,
        )

    coords = torch.cat(reduced_coords, dim=0)
    pixels = torch.cat(pixel_coords, dim=0)
    sources = torch.cat(source_labels, dim=0)
    utility = torch.cat(utilities, dim=0)
    quality = torch.cat(quality_list, dim=0)
    descriptor_bias = None if descriptor_biases is None else torch.cat(descriptor_biases, dim=0)
    gmap_descriptor_bias = None if gmap_biases is None else torch.cat(gmap_biases, dim=0)
    if coords.shape[0] > patch_budget:
        coords = coords[:patch_budget]
        pixels = pixels[:patch_budget]
        sources = sources[:patch_budget]
        utility = utility[:patch_budget]
        quality = quality[:patch_budget]
        if descriptor_bias is not None:
            descriptor_bias = descriptor_bias[:patch_budget]
        if gmap_descriptor_bias is not None:
            gmap_descriptor_bias = gmap_descriptor_bias[:patch_budget]

    return _assemble_patch_input(
        coords=coords,
        pixels=pixels,
        qualities=quality,
        sources=sources,
        utility=utility,
        image_height=image_height,
        image_width=image_width,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        descriptor_bias=descriptor_bias,
        gmap_descriptor_bias=gmap_descriptor_bias,
    )


def build_dpvo_patch_input(
    frame_output: DinoDPVOFrameOutput,
    *,
    patch_budget: int,
    frontend_mode: str,
    dpvo_res: int = 4,
    image_height: int | None = None,
    image_width: int | None = None,
    config: dict[str, Any] | None = None,
    runtime_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if int(patch_budget) <= 0:
        raise ValueError(f"patch_budget must be > 0, got {patch_budget}")

    mode = str(frontend_mode).lower()
    config = dict(config or {})
    grid_rows = int(config.get("hybrid_grid_rows", 6))
    grid_cols = int(config.get("hybrid_grid_cols", 8))

    if runtime_state is not None:
        anchor_max_age = int(config.get("anchor_max_age", 0) or 0)
        anchors = runtime_state.get("semantic_anchors")
        if anchor_max_age > 0 and anchors is not None:
            age = int(runtime_state.get("anchor_age", 0)) + 1
            runtime_state["anchor_age"] = age
            if age > anchor_max_age:
                runtime_state["semantic_anchors"] = None
                runtime_state["anchor_age"] = 0

    if mode == "dino_hybrid":
        if image_height is None or image_width is None:
            raise ValueError("image_height and image_width are required for dino_hybrid mode")
        return _hybrid_patch_input(
            frame_output,
            patch_budget=int(patch_budget),
            image_height=int(image_height),
            image_width=int(image_width),
            dpvo_res=int(dpvo_res),
            native_fraction=float(config.get("native_fraction", 0.75)),
            dino_fraction=float(config.get("dino_fraction", 0.25)),
            static_score_weight=float(config.get("static_score_weight", 0.35)),
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            max_dino_per_cell=int(config.get("max_dino_per_cell", 1)),
            dedupe_radius_px=float(config.get("dedupe_radius_px", 8.0)),
            config=config,
            runtime_state=runtime_state,
        )

    if mode in {"dino_proposals", "dino_full"} and bool(config.get("enforce_unique_semantic", False)):
        if image_height is None or image_width is None:
            raise ValueError("image_height and image_width are required for enforce_unique_semantic mode")
        return _semantic_full_cover_patch_input(
            frame_output,
            patch_budget=int(patch_budget),
            image_height=int(image_height),
            image_width=int(image_width),
            dpvo_res=int(dpvo_res),
            static_score_weight=float(config.get("static_score_weight", 0.35)),
            grid_rows=int(config.get("semantic_grid_rows", config.get("hybrid_grid_rows", 6))),
            grid_cols=int(config.get("semantic_grid_cols", config.get("hybrid_grid_cols", 8))),
            max_semantic_per_cell=int(config.get("max_semantic_per_cell", config.get("max_dino_per_cell", 2))),
            dedupe_radius_px=float(config.get("dedupe_radius_px", 8.0)),
            dedupe_schedule_px=[float(v) for v in config.get("semantic_dedupe_schedule_px", [config.get("dedupe_radius_px", 8.0)])],
            semantic_backfill_source=str(config.get("semantic_backfill_source", "dino")),
            config=config,
            runtime_state=runtime_state,
        )

    coords = pixel_to_dpvo_coords(frame_output.proposal.pixel_xy, dpvo_res=dpvo_res)
    pixels = _repeat_to_budget(frame_output.proposal.pixel_xy, int(patch_budget))
    coords = _repeat_to_budget(coords, int(patch_budget))
    qualities = _repeat_to_budget(_process_external_quality(frame_output.qualities, config=config), int(patch_budget))
    sources = torch.full((int(patch_budget),), SOURCE_DINO, dtype=torch.long, device=pixels.device)
    utility = _repeat_to_budget(frame_output.qualities, int(patch_budget))
    descriptor_bias = None
    if frame_output.descriptor_bias is not None:
        descriptor_bias = _repeat_to_budget(frame_output.descriptor_bias, int(patch_budget))
    gmap_descriptor_bias = None
    if frame_output.gmap_descriptor_bias is not None:
        gmap_descriptor_bias = _repeat_to_budget(frame_output.gmap_descriptor_bias, int(patch_budget))
    return _assemble_patch_input(
        coords=coords,
        pixels=pixels,
        qualities=qualities,
        sources=sources,
        utility=utility,
        image_height=int(image_height or 1),
        image_width=int(image_width or 1),
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        descriptor_bias=descriptor_bias,
        gmap_descriptor_bias=gmap_descriptor_bias,
    )
