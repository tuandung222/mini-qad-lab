"""
Unit tests for QADLoss and Distillation logic.
"""

import torch
import pytest
from mini_qad.distill import QADLoss


def test_qad_loss_pure_task():
    loss_fn = QADLoss(alpha=1.0)
    student_logits = torch.randn(4, 10, requires_grad=True)
    labels = torch.randint(0, 10, (4,))

    loss, metrics = loss_fn(student_logits, labels=labels)
    assert loss.item() > 0
    assert "task_loss" in metrics
    assert "distill_loss" not in metrics

    loss.backward()
    assert student_logits.grad is not None


def test_qad_loss_pure_kd():
    loss_fn = QADLoss(alpha=0.0, temperature=2.0)
    student_logits = torch.randn(4, 10, requires_grad=True)
    teacher_logits = torch.randn(4, 10)

    loss, metrics = loss_fn(student_logits, teacher_logits=teacher_logits)
    assert loss.item() >= 0
    assert "distill_loss" in metrics

    loss.backward()
    assert student_logits.grad is not None


def test_qad_loss_combined():
    loss_fn = QADLoss(alpha=0.3, temperature=2.0)
    student_logits = torch.randn(4, 10, requires_grad=True)
    teacher_logits = torch.randn(4, 10)
    labels = torch.randint(0, 10, (4,))

    loss, metrics = loss_fn(student_logits, teacher_logits=teacher_logits, labels=labels)
    assert loss.item() > 0
    assert "task_loss" in metrics
    assert "distill_loss" in metrics
