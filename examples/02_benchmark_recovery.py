"""
Comprehensive Benchmark: Quantization-Aware Distillation (QAD) vs QAT vs PTQ.
Generates learning curves and accuracy comparison charts saved to `figures/`.
"""

import sys
import os
import copy
import json
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
    plot_benchmark_comparison,
    plot_learning_curves,
)


def run_benchmark():
    device = get_default_device()
    print("=" * 60)
    print(f"📊 Running QAD Benchmark on Device: {device}")
    print("=" * 60)

    # 1. Dataset setup
    train_loader, val_loader = get_dataloaders(
        num_train=1200, num_val=300, seq_len=32, vocab_size=400, batch_size=32, seed=42
    )

    # 2. Train Teacher Model (FP32 Baseline)
    print("\n[Phase 1] Training FP32 Teacher Baseline...")
    teacher = MiniTransformerLM(vocab_size=400, d_model=128, n_heads=4, n_layers=2, max_seq_len=32)
    teacher_trainer = QADTrainer(
        student=teacher,
        loss_fn=QADLoss(alpha=1.0),
        lr=2e-3,
        device=device,
    )
    teacher_trainer.fit(train_loader, val_loader, epochs=5, verbose=True)
    teacher_res = evaluate_model(teacher, val_loader, device)
    print(f"✅ Teacher (FP32) -> Accuracy: {teacher_res['accuracy']:.2f}% | PPL: {teacher_res['ppl']:.2f}")

    # 3. Post-Training Quantization (PTQ Baseline)
    print("\n[Phase 2] Evaluating Post-Training Quantization (INT4)...")
    ptq_student = copy.deepcopy(teacher)
    convert_to_quantized_model(ptq_student, weight_format=QuantFormat.INT4, per_channel_weight=True)
    calibrate_model(ptq_student, train_loader, device, num_batches=15)
    freeze_quantizer_scales(ptq_student)
    ptq_res = evaluate_model(ptq_student, val_loader, device)
    print(f"⚠️  PTQ (INT4)     -> Accuracy: {ptq_res['accuracy']:.2f}% | PPL: {ptq_res['ppl']:.2f}")

    # 4. Standard QAT (Task Loss Only: alpha = 1.0)
    print("\n[Phase 3] Running Standard QAT (Task loss only, alpha=1.0)...")
    qat_student = copy.deepcopy(ptq_student)
    qat_loss = QADLoss(alpha=1.0)
    qat_trainer = QADTrainer(
        student=qat_student,
        loss_fn=qat_loss,
        lr=1e-3,
        device=device,
    )
    qat_history = qat_trainer.fit(train_loader, val_loader, epochs=4, verbose=True)
    qat_res = evaluate_model(qat_student, val_loader, device)
    print(f"📈 QAT (Task Loss)-> Accuracy: {qat_res['accuracy']:.2f}% | PPL: {qat_res['ppl']:.2f}")

    # 5. Quantization-Aware Distillation (QAD: Teacher Soft Targets + Task Loss)
    print("\n[Phase 4] Running QAD (Teacher Distillation, alpha=0.2, T=2.0)...")
    qad_student = copy.deepcopy(ptq_student)
    qad_loss = QADLoss(alpha=0.2, temperature=2.0)
    qad_trainer = QADTrainer(
        student=qad_student,
        teacher=teacher,
        loss_fn=qad_loss,
        lr=1e-3,
        device=device,
    )
    qad_history = qad_trainer.fit(train_loader, val_loader, epochs=4, verbose=True)
    qad_res = evaluate_model(qad_student, val_loader, device)
    print(f"🌟 QAD (Distilled)-> Accuracy: {qad_res['accuracy']:.2f}% | PPL: {qad_res['ppl']:.2f}")

    # Summary results
    benchmark_results = {
        "Teacher (FP32)": teacher_res,
        "PTQ (INT4)": ptq_res,
        "QAT (Hard Labels)": qat_res,
        "QAD (Distillation)": qad_res,
    }

    # 6. Save figures and plots
    os.makedirs("figures", exist_ok=True)
    plot_benchmark_comparison(
        benchmark_results,
        save_path="figures/benchmark_accuracy.png",
        metric="accuracy",
    )
    plot_benchmark_comparison(
        benchmark_results,
        save_path="figures/benchmark_ppl.png",
        metric="ppl",
    )
    plot_learning_curves(
        {"QAT": qat_history, "QAD": qad_history},
        save_path="figures/learning_curves.png",
    )

    # Save metrics JSON
    with open("figures/benchmark_metrics.json", "w") as f:
        json.dump(benchmark_results, f, indent=2)

    print("\n" + "=" * 60)
    print("🏆 FINAL BENCHMARK SUMMARY:")
    for k, v in benchmark_results.items():
        print(f"   * {k:<20}: Accuracy = {v['accuracy']:6.2f}% | PPL = {v['ppl']:6.2f}")
    print("=" * 60)
    print("Plots saved in ./figures/")


if __name__ == "__main__":
    run_benchmark()
