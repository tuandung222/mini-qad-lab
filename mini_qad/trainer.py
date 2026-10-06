"""
Training engine supporting Apple Silicon (MPS), CUDA, and CPU.
Handles training of Teacher baselines, QAT students, and QAD (Distillation) students.
"""

from typing import Dict, List, Optional, Tuple
import math
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn

from .distill import QADLoss

console = Console()


def get_default_device() -> torch.device:
    """Detect the fastest available hardware accelerator on Mac, Linux, or Windows."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def evaluate_model(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate validation loss, perplexity (PPL), and next-token accuracy.
    """
    model.eval()
    ce_loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_tokens = 0
    correct_tokens = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)
            loss = ce_loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))

            preds = logits.argmax(dim=-1)
            correct_tokens += (preds == targets).sum().item()
            num_tokens = targets.numel()
            total_tokens += num_tokens

            total_loss += loss.item() * num_tokens

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    accuracy = (correct_tokens / total_tokens * 100.0) if total_tokens > 0 else 0.0
    ppl = math.exp(min(avg_loss, 20.0))  # Cap to prevent overflow

    return {
        "loss": avg_loss,
        "ppl": ppl,
        "accuracy": accuracy,
    }


class QADTrainer:
    """
    Trainer for Knowledge Distillation and Quantization-Aware Distillation.
    """

    def __init__(
        self,
        student: nn.Module,
        teacher: Optional[nn.Module] = None,
        loss_fn: Optional[QADLoss] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: Optional[torch.device] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        self.device = device or get_default_device()
        self.student = student.to(self.device)
        self.teacher = teacher.to(self.device) if teacher is not None else None

        if self.teacher is not None:
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad = False

        self.loss_fn = loss_fn or QADLoss(alpha=0.5, temperature=2.0)
        self.optimizer = optimizer or torch.optim.AdamW(
            self.student.parameters(), lr=lr, weight_decay=weight_decay
        )

        self.history: List[Dict[str, float]] = []

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        self.student.train()
        total_loss = 0.0
        total_task_loss = 0.0
        total_distill_loss = 0.0
        num_batches = len(train_loader)

        for inputs, targets in train_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            student_logits = self.student(inputs)

            teacher_logits = None
            if self.teacher is not None and self.loss_fn.alpha < 1.0:
                with torch.no_grad():
                    teacher_logits = self.teacher(inputs)

            loss, metrics = self.loss_fn(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                labels=targets,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += metrics.get("total_loss", 0.0)
            total_task_loss += metrics.get("task_loss", 0.0)
            total_distill_loss += metrics.get("distill_loss", 0.0)

        return {
            "train_loss": total_loss / max(1, num_batches),
            "train_task_loss": total_task_loss / max(1, num_batches),
            "train_distill_loss": total_distill_loss / max(1, num_batches),
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 5,
        verbose: bool = True,
    ) -> List[Dict[str, float]]:
        """Run training loop across epochs and track validation metrics."""
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            train_metrics = self.train_epoch(train_loader)
            val_metrics = evaluate_model(self.student, val_loader, self.device)
            elapsed = time.time() - start_time

            epoch_record = {
                "epoch": epoch,
                "train_loss": train_metrics["train_loss"],
                "val_loss": val_metrics["loss"],
                "val_ppl": val_metrics["ppl"],
                "val_accuracy": val_metrics["accuracy"],
                "time_sec": elapsed,
            }
            self.history.append(epoch_record)

            if verbose:
                console.print(
                    f"[bold cyan]Epoch {epoch:02d}/{epochs:02d}[/bold cyan] | "
                    f"Train Loss: [yellow]{train_metrics['train_loss']:.4f}[/yellow] | "
                    f"Val Loss: [green]{val_metrics['loss']:.4f}[/green] | "
                    f"Val PPL: [magenta]{val_metrics['ppl']:.2f}[/magenta] | "
                    f"Val Acc: [bold blue]{val_metrics['accuracy']:.2f}%[/bold blue] | "
                    f"({elapsed:.1f}s)"
                )

        return self.history
