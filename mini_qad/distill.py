"""
Distillation loss functions for Quantization-Aware Distillation (QAD).
Formulates the combined task loss and logit-level KL divergence.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class QADLoss(nn.Module):
    """
    Quantization-Aware Distillation Loss module.

    Computes:
        L_total = alpha * L_task + (1 - alpha) * L_distill

    Where:
        L_task: Cross-Entropy loss between student logits and ground truth labels.
        L_distill: Temperature-scaled KL divergence between teacher and student logits:
                   L_distill = T^2 * KL( Softmax(z_teacher / T) || LogSoftmax(z_student / T) )

    Parameters:
        alpha: Weight for task loss in [0.0, 1.0].
               If alpha=0.0 (Pure QAD), only the teacher's distribution guides the student.
               If alpha=1.0 (Standard QAT), the teacher is ignored and only hard labels are used.
        temperature: Temperature factor T for smoothing logit distributions (default: 2.0).
        ignore_index: Index to ignore in CrossEntropyLoss (e.g., -100 for padding).
    """

    def __init__(
        self,
        alpha: float = 0.5,
        temperature: float = 2.0,
        ignore_index: int = -100,
    ):
        super().__init__()
        assert 0.0 <= alpha <= 1.0, f"alpha must be between 0.0 and 1.0, got {alpha}"
        assert temperature > 0.0, f"temperature must be positive, got {temperature}"

        self.alpha = alpha
        self.temperature = temperature
        self.ignore_index = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Calculate the combined QAD loss.

        Args:
            student_logits: Logits from quantized student (B, ..., V)
            teacher_logits: Logits from frozen full-precision teacher (B, ..., V)
            labels: Ground truth token indices (B, ...)

        Returns:
            total_loss: Scalar loss tensor for backpropagation
            metrics: Dictionary of individual loss components for logging
        """
        metrics = {}
        total_loss = torch.tensor(0.0, device=student_logits.device)

        # Flatten shapes if needed (B, S, V) -> (B*S, V)
        vocab_size = student_logits.size(-1)
        flat_student_logits = student_logits.view(-1, vocab_size)

        # 1. Task Loss (Cross Entropy)
        task_loss = torch.tensor(0.0, device=student_logits.device)
        if labels is not None and self.alpha > 0.0:
            flat_labels = labels.view(-1)
            task_loss = self.ce_loss(flat_student_logits, flat_labels)
            metrics["task_loss"] = task_loss.item()
            total_loss = total_loss + self.alpha * task_loss
        elif labels is not None:
            # Still compute task loss for tracking metrics even if alpha=0
            with torch.no_grad():
                flat_labels = labels.view(-1)
                task_loss = self.ce_loss(flat_student_logits, flat_labels)
                metrics["task_loss"] = task_loss.item()

        # 2. Distillation Loss (KL Divergence with Temperature)
        distill_loss = torch.tensor(0.0, device=student_logits.device)
        if teacher_logits is not None and self.alpha < 1.0:
            flat_teacher_logits = teacher_logits.view(-1, vocab_size)

            # Soft targets from teacher
            p_teacher = F.softmax(flat_teacher_logits / self.temperature, dim=-1)
            # Log probabilities from student
            log_p_student = F.log_softmax(flat_student_logits / self.temperature, dim=-1)

            # KL divergence: KL(teacher || student)
            # F.kl_div expects log_p as input and target probabilities
            # reduction='batchmean' mathematically computes the true KL divergence
            kl_div = F.kl_div(log_p_student, p_teacher, reduction="batchmean")

            # Scale by T^2 as per Hinton et al. to balance gradients
            distill_loss = kl_div * (self.temperature**2)
            metrics["distill_loss"] = distill_loss.item()
            metrics["raw_kl"] = kl_div.item()

            total_loss = total_loss + (1.0 - self.alpha) * distill_loss

        metrics["total_loss"] = total_loss.item()
        return total_loss, metrics
