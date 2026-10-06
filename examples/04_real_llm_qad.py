"""
Real LLM QAD Experiment:
Methodology:
1. SFT Train Base FP32 Model on Alpaca dataset -> Specialized Teacher (FP32).
2. Apply Post-Training Quantization (INT4) on Teacher -> Degraded PTQ Student.
3. Apply Quantization-Aware Distillation (QAD) -> Recover Quantization Gap!
Runnable on Apple Silicon (MPS) or CPU.
"""

import sys
import os
import copy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure package is found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mini_qad import (
    QuantFormat,
    convert_to_quantized_model,
    calibrate_model,
    freeze_quantizer_scales,
    QADLoss,
    QADTrainer,
    evaluate_model,
    get_default_device,
    get_real_dataloaders,
)


def main():
    device = get_default_device()
    model_name = os.getenv("QAD_MODEL", "openai-community/gpt2")

    print("=" * 65)
    print(f"🚀 Rigorous 3-Stage Real LLM QAD Experiment")
    print(f"📦 Model: {model_name}")
    print(f"📚 Dataset: tatsu-lab/alpaca (Real SFT Instruction Dataset)")
    print(f"📱 Target Device: {device}")
    print("=" * 65)

    print(f"\n[1/4] Loading tokenizer and base model ({model_name})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load Base FP32 Model
    model_fp32 = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
    ).to(device)

    # Load real SFT dataset (Alpaca)
    print("\n[2/4] Preparing real SFT dataset (tatsu-lab/alpaca)...")
    train_loader, val_loader = get_real_dataloaders(
        tokenizer=tokenizer,
        dataset_name="tatsu-lab/alpaca",
        max_samples=250,
        max_length=64,
        batch_size=4,
    )
    print(f"Loaded {len(train_loader.dataset)} training samples, {len(val_loader.dataset)} validation samples.")

    # Measure zero-shot base model before training
    raw_eval = evaluate_model(model_fp32, val_loader, device)
    print(f"Pretrained Zero-Shot Base -> Val Loss: {raw_eval['loss']:.4f} | PPL: {raw_eval['ppl']:.2f}")

    # =========================================================================
    # STAGE 1: Train Base FP32 on Dataset to create a true Specialized Teacher!
    # =========================================================================
    print("\n" + "-" * 65)
    print("🔥 [STAGE 1] SFT Training FP32 Model on Alpaca -> Producing True Teacher...")
    print("-" * 65)
    teacher_opt = torch.optim.AdamW(model_fp32.parameters(), lr=1e-4, weight_decay=1e-4)
    teacher_trainer = QADTrainer(
        student=model_fp32,
        teacher=None,
        loss_fn=QADLoss(alpha=1.0, ignore_index=-100),
        optimizer=teacher_opt,
        device=device,
    )
    teacher_trainer.fit(train_loader, val_loader, epochs=2, verbose=True)

    teacher_eval = evaluate_model(model_fp32, val_loader, device)
    print(f"✅ SFT Teacher (FP32) Converged -> Val Loss: {teacher_eval['loss']:.4f} | PPL: {teacher_eval['ppl']:.2f} | Acc: {teacher_eval['accuracy']:.2f}%")

    # Freeze Teacher
    teacher = model_fp32
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # =========================================================================
    # STAGE 2: Post-Training Quantization (INT4) on the Converged Teacher
    # =========================================================================
    print("\n" + "-" * 65)
    print("⚠️ [STAGE 2] Quantizing Converged Teacher to INT4 (PTQ)...")
    print("-" * 65)
    student = copy.deepcopy(teacher)
    convert_to_quantized_model(
        student,
        weight_format=QuantFormat.INT4,
        per_channel_weight=True,
        exclude_modules={"lm_head"},
    )
    calibrate_model(student, train_loader, device, num_batches=15)
    freeze_quantizer_scales(student)

    ptq_eval = evaluate_model(student, val_loader, device)
    quant_gap = ptq_eval['loss'] - teacher_eval['loss']
    print(f"📉 PTQ Student (INT4) -> Val Loss: {ptq_eval['loss']:.4f} | PPL: {ptq_eval['ppl']:.2f} | Acc: {ptq_eval['accuracy']:.2f}%")
    print(f"🚨 Quantization Accuracy Deficit (Gap to Teacher): +{quant_gap:.4f} loss ({ptq_eval['ppl'] - teacher_eval['ppl']:.2f} PPL drop)")

    # =========================================================================
    # STAGE 3: Quantization-Aware Distillation (QAD) to Recover Accuracy
    # =========================================================================
    print("\n" + "-" * 65)
    print("🌟 [STAGE 3] Running QAD (Quantization-Aware Distillation from Teacher)...")
    print("-" * 65)
    qad_loss = QADLoss(alpha=0.3, temperature=2.0, ignore_index=-100)
    qad_opt = torch.optim.AdamW(student.parameters(), lr=5e-5, weight_decay=1e-4)

    qad_trainer = QADTrainer(
        student=student,
        teacher=teacher,
        loss_fn=qad_loss,
        optimizer=qad_opt,
        device=device,
    )
    qad_trainer.fit(train_loader, val_loader, epochs=2, verbose=True)

    qad_eval = evaluate_model(student, val_loader, device)
    recovered = ptq_eval['loss'] - qad_eval['loss']

    print("\n" + "=" * 65)
    print("🏆 RIGOROUS QAD EXPERIMENT FINAL SUMMARY:")
    print(f"   * 1. SFT Teacher (FP32)    : Val Loss = {teacher_eval['loss']:.4f} | PPL = {teacher_eval['ppl']:.2f} | Acc = {teacher_eval['accuracy']:.2f}%")
    print(f"   * 2. PTQ Student (INT4)    : Val Loss = {ptq_eval['loss']:.4f} | PPL = {ptq_eval['ppl']:.2f} | Acc = {ptq_eval['accuracy']:.2f}% (Quantization Gap: +{quant_gap:.4f})")
    print(f"   * 3. QAD Student (Recovered): Val Loss = {qad_eval['loss']:.4f} | PPL = {qad_eval['ppl']:.2f} | Acc = {qad_eval['accuracy']:.2f}% (Recovered: -{recovered:.4f})")
    print("=" * 65)


if __name__ == "__main__":
    main()
