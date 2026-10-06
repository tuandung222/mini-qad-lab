"""
Real LLM QAD Experiment:
Applies Quantization-Aware Distillation to a real Hugging Face model
(e.g., SmolLM-135M or GPT-2) using the real-world Alpaca SFT instruction dataset.
Runnable on Apple Silicon (MPS) or CPU!
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
    print(f"🚀 Real LLM QAD Experiment")
    print(f"📦 Model: {model_name}")
    print(f"📚 Dataset: tatsu-lab/alpaca (Real SFT Instruction Dataset)")
    print(f"📱 Target Device: {device}")
    print("=" * 65)

    print(f"\n[1/5] Loading tokenizer and model ({model_name})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load Teacher (FP32)
    teacher = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
    ).to(device)
    teacher.eval()

    # Load real SFT dataset (Alpaca)
    print("\n[2/5] Loading real SFT dataset (tatsu-lab/alpaca)...")
    train_loader, val_loader = get_real_dataloaders(
        tokenizer=tokenizer,
        dataset_name="tatsu-lab/alpaca",
        max_samples=200,
        max_length=64,
        batch_size=4,
    )
    print(f"Successfully loaded {len(train_loader.dataset)} training samples, {len(val_loader.dataset)} validation samples.")

    # Evaluate Teacher
    teacher_eval = evaluate_model(teacher, val_loader, device)
    print(f"✅ Teacher Baseline -> Val Loss: {teacher_eval['loss']:.4f} | PPL: {teacher_eval['ppl']:.2f}")

    # [3/5] Create Student via Fake Quantization (INT4 weights)
    print("\n[3/5] Quantizing Student to INT4 (using Straight-Through Estimator)...")
    student = copy.deepcopy(teacher)
    # Exclude lm_head to avoid extreme degradation on classification head
    convert_to_quantized_model(
        student,
        weight_format=QuantFormat.INT4,
        per_channel_weight=True,
        exclude_modules={"lm_head"},
    )
    calibrate_model(student, train_loader, device, num_batches=10)
    freeze_quantizer_scales(student)

    ptq_eval = evaluate_model(student, val_loader, device)
    loss_increase = ptq_eval['loss'] - teacher_eval['loss']
    print(f"⚠️  PTQ Student (INT4) -> Val Loss: {ptq_eval['loss']:.4f} | PPL: {ptq_eval['ppl']:.2f} (Loss bump: +{loss_increase:.4f})")

    # [4/5] Run QAD Recovery Training
    print("\n[4/5] Running Quantization-Aware Distillation (QAD)...")
    qad_loss = QADLoss(alpha=0.2, temperature=2.0, ignore_index=-100)
    optimizer = torch.optim.AdamW(student.parameters(), lr=5e-5, weight_decay=1e-4)

    trainer = QADTrainer(
        student=student,
        teacher=teacher,
        loss_fn=qad_loss,
        optimizer=optimizer,
        device=device,
    )
    trainer.fit(train_loader, val_loader, epochs=2, verbose=True)

    # [5/5] Final evaluation
    qad_eval = evaluate_model(student, val_loader, device)
    recovered_loss = ptq_eval['loss'] - qad_eval['loss']

    print("\n" + "=" * 65)
    print("📊 REAL LLM EXPERIMENT SUMMARY:")
    print(f"   * Teacher (FP32):   Val Loss = {teacher_eval['loss']:.4f} | PPL = {teacher_eval['ppl']:.2f}")
    print(f"   * PTQ Student:      Val Loss = {ptq_eval['loss']:.4f} | PPL = {ptq_eval['ppl']:.2f}")
    print(f"   * QAD Student:      Val Loss = {qad_eval['loss']:.4f} | PPL = {qad_eval['ppl']:.2f} (Recovered: -{recovered_loss:.4f} loss)")
    print("=" * 65)


if __name__ == "__main__":
    main()
