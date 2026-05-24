from __future__ import annotations

import types
import unittest
from pathlib import Path
import sys

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "refocus_vo" / "src"))

from refocus_vo.patchgraph.model import DinoPatchGraphVO, FrameObservation, PairPrediction
from refocus_vo.patchgraph.teacher import DinoPatchTeacher, PseudoObjectPatchProposal


def _make_proposal(num_patches: int = 1) -> PseudoObjectPatchProposal:
    patch_indices = torch.arange(num_patches, dtype=torch.long)
    patch_xy = torch.zeros((num_patches, 2), dtype=torch.float32)
    pixel_xy = torch.zeros((num_patches, 2), dtype=torch.float32)
    return PseudoObjectPatchProposal(
        patch_indices=patch_indices,
        patch_xy=patch_xy,
        coarse_pixel_xy=pixel_xy.clone(),
        pixel_xy=pixel_xy.clone(),
        offset_xy=torch.zeros_like(pixel_xy),
        scores=torch.ones((num_patches,), dtype=torch.float32),
        object_ids=torch.zeros((num_patches,), dtype=torch.long),
        descriptors=torch.zeros((num_patches, 4), dtype=torch.float32),
        local_features=torch.zeros((num_patches, 4), dtype=torch.float32),
    )


class RefocusPatchGraphStageTests(unittest.TestCase):
    def test_teacher_selection_can_override_selector_bias(self) -> None:
        teacher = DinoPatchTeacher(
            patch_size=16,
            num_patches=1,
            max_nodes_per_object_ratio=1.0,
        )
        fused = torch.zeros((1, 2, 1, 4), dtype=torch.float32)
        local = torch.zeros((1, 2, 1, 4), dtype=torch.float32)
        object_ids = torch.zeros((1, 1, 4), dtype=torch.long)
        teacher_scores = torch.tensor([[[0.51, 0.50, 0.49, 0.20]]], dtype=torch.float32)
        selector_logits = torch.tensor([[[[-10.0, -10.0, -10.0, 10.0]]]], dtype=torch.float32)

        blended = teacher.select_patches(
            fused=fused,
            local_features=local,
            patch_score=teacher_scores,
            object_ids=object_ids,
            selector_logits=selector_logits,
            num_patches=1,
        )[0]
        teacher_only = teacher.select_patches(
            fused=fused,
            local_features=local,
            patch_score=teacher_scores,
            object_ids=object_ids,
            selector_logits=None,
            num_patches=1,
        )[0]

        self.assertEqual(int(blended.patch_indices[0].item()), 3)
        self.assertEqual(int(teacher_only.patch_indices[0].item()), 0)

    def test_offset_refinement_stays_inside_patch_extent(self) -> None:
        model = DinoPatchGraphVO.__new__(DinoPatchGraphVO)
        nn.Module.__init__(model)
        model.patch_size = 16
        model.enable_offset_refinement = True
        model.offset_head = nn.Linear(8, 2)
        with torch.no_grad():
            model.offset_head.weight.zero_()
            model.offset_head.bias.copy_(torch.tensor([100.0, -100.0]))

        proposal = PseudoObjectPatchProposal(
            patch_indices=torch.tensor([0], dtype=torch.long),
            patch_xy=torch.tensor([[1.0, 2.0]], dtype=torch.float32),
            coarse_pixel_xy=torch.tensor([[100.0, 80.0]], dtype=torch.float32),
            pixel_xy=torch.tensor([[100.0, 80.0]], dtype=torch.float32),
            offset_xy=torch.zeros((1, 2), dtype=torch.float32),
            scores=torch.tensor([1.0], dtype=torch.float32),
            object_ids=torch.tensor([0], dtype=torch.long),
            descriptors=torch.ones((1, 4), dtype=torch.float32),
            local_features=torch.zeros((1, 4), dtype=torch.float32),
        )
        fine_local_map = torch.ones((1, 4, 30, 40), dtype=torch.float32)

        refined = model._refine_proposal(
            proposal,
            fine_local_map=fine_local_map,
            image_height=240,
            image_width=320,
        )

        self.assertLessEqual(float(refined.offset_xy.abs().max().item()), 8.0 + 1e-5)
        self.assertTrue(torch.all(refined.pixel_xy[:, 0] >= 0.0))
        self.assertTrue(torch.all(refined.pixel_xy[:, 0] <= 319.0))
        self.assertTrue(torch.all(refined.pixel_xy[:, 1] >= 0.0))
        self.assertTrue(torch.all(refined.pixel_xy[:, 1] <= 239.0))
        self.assertGreater(float(refined.offset_xy.abs().sum().item()), 0.0)

    def test_multiframe_history_uses_t2_and_t3_edges(self) -> None:
        model = DinoPatchGraphVO.__new__(DinoPatchGraphVO)
        nn.Module.__init__(model)
        model.use_multiframe_graph = True
        model.max_history = 3
        captured: dict[str, list[int]] = {}

        def fake_pair_prediction(self, src_obs, tgt_obs, *, src_frame_idx: int, tgt_frame_idx: int, lag: int):
            return PairPrediction(
                src_frame_idx=src_frame_idx,
                tgt_frame_idx=tgt_frame_idx,
                lag=lag,
                src_indices=torch.tensor([0], dtype=torch.long),
                tgt_indices=torch.tensor([0], dtype=torch.long),
                similarity=torch.tensor([1.0], dtype=torch.float32),
                confidence_logits=torch.tensor([0.0], dtype=torch.float32),
                edge_embeddings=torch.zeros((1, 4), dtype=torch.float32),
                pooled_embedding=torch.full((4,), float(lag), dtype=torch.float32),
                pose_vec=torch.zeros((6,), dtype=torch.float32),
            )

        def fake_predict_frame_pose(self, incoming_pairs, prev_hidden):
            captured["lags"] = [int(pair.lag) for pair in incoming_pairs]
            return torch.zeros((6,), dtype=torch.float32), torch.ones((3,), dtype=torch.float32)

        model._pair_prediction = types.MethodType(fake_pair_prediction, model)
        model._predict_frame_pose = types.MethodType(fake_predict_frame_pose, model)

        history = [
            FrameObservation(
                fused=torch.zeros((4, 2, 2), dtype=torch.float32),
                local_map=torch.zeros((2, 2, 2), dtype=torch.float32),
                selector_logits=torch.zeros((2, 2), dtype=torch.float32),
                patch_score=torch.zeros((2, 2), dtype=torch.float32),
                object_ids=torch.zeros((2, 2), dtype=torch.long),
                proposal=_make_proposal(),
            )
            for _ in range(4)
        ]
        cur_obs = history[0]

        incoming_pairs, pose_vec, hidden = model.predict_target_from_history(history, cur_obs, prev_hidden=None)

        self.assertEqual(len(incoming_pairs), 3)
        self.assertEqual(captured["lags"], [3, 2, 1])
        self.assertTrue(any(lag == 2 for lag in captured["lags"]))
        self.assertTrue(any(lag == 3 for lag in captured["lags"]))
        self.assertEqual(tuple(pose_vec.shape), (6,))
        self.assertEqual(tuple(hidden.shape), (3,))


if __name__ == "__main__":
    unittest.main()
