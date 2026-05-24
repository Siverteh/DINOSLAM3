from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "refocus_vo" / "src"))

from refocus_vo.train_dino_dpvo_frontend import (  # noqa: E402
    _compute_sampled_similarity_loss,
    _gram_anchor_teacher_fused,
)


class _TupleTeacher:
    def _encode_backbone(self, images: torch.Tensor):
        fused = torch.ones(images.shape[0], images.shape[1], 8, 2, 2, device=images.device)
        register = torch.zeros(images.shape[0], images.shape[1], 8, device=images.device)
        return fused, register


class _TensorTeacher:
    def _encode_backbone(self, images: torch.Tensor):
        return torch.ones(images.shape[0], images.shape[1], 8, 2, 2, device=images.device)


class GramAnchorLossTests(unittest.TestCase):
    def test_teacher_fused_helper_unpacks_tuple_output(self) -> None:
        images = torch.randn(1, 2, 3, 32, 32)
        fused = _gram_anchor_teacher_fused(_TupleTeacher(), images)
        self.assertEqual(tuple(fused.shape), (1, 2, 8, 2, 2))

    def test_teacher_fused_helper_accepts_tensor_output(self) -> None:
        images = torch.randn(1, 2, 3, 32, 32)
        fused = _gram_anchor_teacher_fused(_TensorTeacher(), images)
        self.assertEqual(tuple(fused.shape), (1, 2, 8, 2, 2))

    def test_similarity_loss_is_zero_for_identical_features(self) -> None:
        torch.manual_seed(7)
        fused = torch.randn(2, 3, 8, 4, 4)
        loss = _compute_sampled_similarity_loss(
            fused,
            fused.clone(),
            sample_tokens=16,
            downsample_stride=1,
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(float(loss.item()), 0.0, places=6)

    def test_similarity_loss_backward_is_finite(self) -> None:
        torch.manual_seed(11)
        student = torch.randn(2, 2, 8, 4, 4, requires_grad=True)
        teacher = student.detach().clone() + 0.1 * torch.randn(2, 2, 8, 4, 4)

        loss = _compute_sampled_similarity_loss(
            student,
            teacher,
            sample_tokens=12,
            downsample_stride=2,
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(student.grad)
        self.assertTrue(torch.isfinite(student.grad).all().item())


if __name__ == "__main__":
    unittest.main()
