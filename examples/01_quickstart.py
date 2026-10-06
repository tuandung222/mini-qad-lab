"""
Quickstart: Quantization-Aware Distillation (QAD) on Mac (MPS/CPU).
Runs in ~30 seconds, demonstrating immediate accuracy recovery.
"""

import sys
import os
import copy
import torch

# Ensure package is found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mini_qad import (
    MiniTransformerLM,
    QuantFormat,
    convert_to_quantized_model,
    calibrate_model,
    freeze_quantizer_scales,
    QADLoss,
    QADTrainer,
    evaluate_model,
    get_default_device,
    get_dataloaders,
)


def main():
    device = get_default_device()
    print(f"==================================================")
    print(f"🚀 Running Mini QAD Quickstart on device: {device}")
    print(f"==================================================")

    # 1. Dataset
    train_loader, val_loader = get_dataloaders(
        num_train=800, num_val=200, seq_len=32, vocab_size=300, batch_size=32
    )

    # 2. Train baseline Teacher model (FP32)
    print("\n[Step 1] Training baseline Teacher model (FP32)...")
    teacher = MiniTransformerLM(vocab_size=300, d_model=64, n_heads=2, n_layers=2, max_seq_len=32)
    teacher_trainer = QADTrainer(
        student=teacher,
        loss_fn=QADLoss(alpha=1.0),  # Pure CrossEntropy for teacher
        lr=3e-3,
        device=device,
    )
    teacher_trainer.fit(train_loader, val_loader, epochs=4, verbose=True)

    teacher_eval = evaluate_model(teacher, val_loader, device)
    print(f"✅ Teacher (FP32) final accuracy: {teacher_eval['accuracy']:.2f}% (PPL: {teacher_eval['ppl']:.2f})")

    # 3. Create Student via PTQ (Post-Training Quantization - INT4)
    print("\n[Step 2] Applying PTQ Quantization (INT4 weights)...")
    student = copy.deepcopy(teacher)
    convert_to_quantized_model(student, weight_format=QuantFormat.INT4, per_channel_weight=True)

    # Calibrate on train data
    calibrate_model(student, train_loader, device, num_batches=10)
    freeze_quantizer_scales(student)

    ptq_eval = evaluate_model(student, val_loader, device)
    acc_drop = teacher_eval['accuracy'] - ptq_eval['accuracy']
    print(f"⚠️  PTQ Student (INT4) accuracy: {ptq_eval['accuracy']:.2f}% (Drop: -{acc_drop:.2f} pp)")

    # 4. Recover accuracy using Quantization-Aware Distillation (QAD)
    print("\n[Step 3] Running QAD (Student fine-tuning guided by Teacher)...")
    qad_loss = QADLoss(alpha=0.2, temperature=2.0)
    qad_trainer = QADTrainer(
        student=student,
        teacher=teacher,
        loss_fn=qad_loss,
        lr=1e-3,
        device=device,
    )
    qad_trainer.fit(train_loader, val_loader, epochs=3, verbose=True)

    qad_eval = evaluate_model(student, val_loader, device)
    acc_recovered = qad_eval['accuracy'] - ptq_eval['accuracy']

    print(f"\n==================================================")
    print(f"📊 SUMMARY OF RECOVERY RESULTS:")
    print(f"   - Teacher (FP32):  {teacher_eval['accuracy']:.2f}%")
    print(f"   - PTQ (INT4):       {ptq_eval['accuracy']:.2f}%  (Gap: -{acc_drop:.2f}%)")
    print(f"   - QAD (Recovered):  {qad_eval['accuracy']:.2f}%  (+{acc_recovered:.2f}% gained)")
    print(f"==================================================")


if __name__ == "__main__":
    main()
