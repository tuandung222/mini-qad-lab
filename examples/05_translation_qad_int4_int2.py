"""
Real Translation QAD Experiment (English -> Vietnamese):
Model: Helsinki-NLP/opus-mt-en-vi (72M Seq2Seq MarianMT)
Dataset: Helsinki-NLP/opus-100 (en-vi)
Evaluates INT4 vs INT2 (extreme 2-bit quantization) and QAD recovery!
Runnable on Apple Silicon (MPS) or CPU.
"""

import sys
import os
import copy
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure package is found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mini_qad import (
    QuantFormat,
    convert_to_quantized_model,
    calibrate_model,
    freeze_quantizer_scales,
    QADLoss,
    get_default_device,
)


class TranslationDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_src_len=48, max_tgt_len=48):
        self.examples = []
        for pair in pairs:
            src = pair.get("en", "").strip()
            tgt = pair.get("vi", "").strip()
            if not src or not tgt:
                continue
            src_enc = tokenizer(src, truncation=True, max_length=max_src_len, return_tensors="pt")
            tgt_enc = tokenizer(tgt, truncation=True, max_length=max_tgt_len, return_tensors="pt")
            self.examples.append({
                "src_ids": src_enc["input_ids"].squeeze(0),
                "src_mask": src_enc["attention_mask"].squeeze(0),
                "tgt_ids": tgt_enc["input_ids"].squeeze(0),
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def pad_translation_collate(batch):
    max_src_len = max(len(x["src_ids"]) for x in batch)
    max_tgt_len = max(len(x["tgt_ids"]) for x in batch)

    batch_src_ids, batch_src_mask, batch_tgt_ids = [], [], []
    for item in batch:
        pad_src = max_src_len - len(item["src_ids"])
        pad_tgt = max_tgt_len - len(item["tgt_ids"])

        batch_src_ids.append(torch.cat([item["src_ids"], torch.zeros(pad_src, dtype=torch.long)]))
        batch_src_mask.append(torch.cat([item["src_mask"], torch.zeros(pad_src, dtype=torch.long)]))
        batch_tgt_ids.append(torch.cat([item["tgt_ids"], torch.full((pad_tgt,), -100, dtype=torch.long)]))

    return {
        "input_ids": torch.stack(batch_src_ids),
        "attention_mask": torch.stack(batch_src_mask),
        "labels": torch.stack(batch_tgt_ids),
    }


def evaluate_seq2seq(model, dataloader, device):
    model.eval()
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    total_loss, total_tokens, correct_tokens = 0.0, 0, 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = out.logits

            flat_logits = logits.view(-1, logits.size(-1))
            flat_labels = labels.view(-1)
            loss = ce_loss_fn(flat_logits, flat_labels)

            preds = flat_logits.argmax(dim=-1)
            valid_mask = flat_labels != -100
            valid_count = valid_mask.sum().item()

            if valid_count > 0:
                correct_tokens += ((preds == flat_labels) & valid_mask).sum().item()
                total_tokens += valid_count
                total_loss += loss.item() * valid_count

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    accuracy = (correct_tokens / total_tokens * 100.0) if total_tokens > 0 else 0.0
    ppl = math.exp(min(avg_loss, 20.0))
    return {"loss": avg_loss, "ppl": ppl, "accuracy": accuracy}


def train_translation_qad(student, teacher, train_loader, val_loader, device, epochs=2, lr=5e-5, alpha=0.3):
    loss_fn = QADLoss(alpha=alpha, temperature=2.0, ignore_index=-100)
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(1, epochs + 1):
        student.train()
        total_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            student_out = student(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            with torch.no_grad():
                teacher_out = teacher(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            loss, _ = loss_fn(student_out.logits, teacher_out.logits, labels=labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        val_metrics = evaluate_seq2seq(student, val_loader, device)
        print(f"   Epoch {epoch:02d}/{epochs:02d} | Train Loss: {total_loss/len(train_loader):.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | Val PPL: {val_metrics['ppl']:.2f} | Acc: {val_metrics['accuracy']:.2f}%")

    return evaluate_seq2seq(student, val_loader, device)


def generate_translation(model, tokenizer, test_sentences, device):
    model.eval()
    results = []
    for text in test_sentences:
        inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_length=40,
                num_beams=2,
                early_stopping=True,
            )
        vi_trans = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results.append((text, vi_trans))
    return results


def main():
    device = get_default_device()
    model_name = "Helsinki-NLP/opus-mt-en-vi"

    print("=" * 70)
    print(f"🌐 Neural Machine Translation (EN -> VI) Quantization Experiment")
    print(f"📦 Model: {model_name} (72M parameters)")
    print(f"🎯 Formats evaluated: INT4 (4-bit) & INT2 (Extreme 2-bit)")
    print(f"📱 Target Device: {device}")
    print("=" * 70)

    # 1. Load Tokenizer & Teacher Model (FP32)
    print("\n[Step 1] Loading MarianMT model & tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    teacher = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # 2. Load Real Translation Dataset (Helsinki-NLP/opus-100)
    print("\n[Step 2] Loading real bilingual corpus (Helsinki-NLP/opus-100 en-vi)...")
    ds = load_dataset("Helsinki-NLP/opus-100", "en-vi", split="train[:250]")
    pairs = [item["translation"] for item in ds]

    train_pairs = pairs[:200]
    val_pairs = pairs[200:250]

    train_dataset = TranslationDataset(train_pairs, tokenizer)
    val_dataset = TranslationDataset(val_pairs, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=pad_translation_collate)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=pad_translation_collate)
    print(f"Dataset ready: {len(train_dataset)} training pairs, {len(val_dataset)} validation pairs.")

    # 3. Evaluate FP32 Teacher Baseline
    teacher_eval = evaluate_seq2seq(teacher, val_loader, device)
    print(f"✅ Teacher (FP32) Baseline -> Loss: {teacher_eval['loss']:.4f} | PPL: {teacher_eval['ppl']:.2f} | Acc: {teacher_eval['accuracy']:.2f}%")

    # =========================================================================
    # EXPERIMENT A: INT4 (4-bit Quantization)
    # =========================================================================
    print("\n" + "=" * 70)
    print("🔹 EXPERIMENT A: INT4 QUANTIZATION & RECOVERY")
    print("=" * 70)
    student_int4 = copy.deepcopy(teacher)
    convert_to_quantized_model(student_int4, weight_format=QuantFormat.INT4, per_channel_weight=True, exclude_modules={"lm_head"})
    student_int4.eval()
    freeze_quantizer_scales(student_int4)

    ptq_int4_eval = evaluate_seq2seq(student_int4, val_loader, device)
    int4_gap = ptq_int4_eval['loss'] - teacher_eval['loss']
    print(f"⚠️  PTQ (INT4) -> Loss: {ptq_int4_eval['loss']:.4f} | PPL: {ptq_int4_eval['ppl']:.2f} | Acc: {ptq_int4_eval['accuracy']:.2f}% (Loss bump: +{int4_gap:.4f})")

    print("\n🔥 Running QAD for INT4 Student...")
    qad_int4_eval = train_translation_qad(student_int4, teacher, train_loader, val_loader, device, epochs=2, lr=8e-5, alpha=0.3)
    int4_recovered = ptq_int4_eval['loss'] - qad_int4_eval['loss']
    print(f"🌟 QAD (INT4) -> Loss: {qad_int4_eval['loss']:.4f} | PPL: {qad_int4_eval['ppl']:.2f} | Acc: {qad_int4_eval['accuracy']:.2f}% (Recovered: -{int4_recovered:.4f})")

    # =========================================================================
    # EXPERIMENT B: INT2 (Extreme 2-bit Quantization - 4 discrete levels!)
    # =========================================================================
    print("\n" + "=" * 70)
    print("🔸 EXPERIMENT B: INT2 QUANTIZATION (EXTREME 2-BIT) & RECOVERY")
    print("=" * 70)
    student_int2 = copy.deepcopy(teacher)
    convert_to_quantized_model(student_int2, weight_format=QuantFormat.INT2, per_channel_weight=True, exclude_modules={"lm_head"})
    student_int2.eval()
    freeze_quantizer_scales(student_int2)

    ptq_int2_eval = evaluate_seq2seq(student_int2, val_loader, device)
    int2_gap = ptq_int2_eval['loss'] - teacher_eval['loss']
    print(f"🚨 PTQ (INT2) -> Loss: {ptq_int2_eval['loss']:.4f} | PPL: {ptq_int2_eval['ppl']:.2f} | Acc: {ptq_int2_eval['accuracy']:.2f}% (Loss explosion: +{int2_gap:.4f})")

    print("\n🔥 Running QAD for INT2 Student (Rescuing Catastrophic Collapse)...")
    qad_int2_eval = train_translation_qad(student_int2, teacher, train_loader, val_loader, device, epochs=2, lr=1e-4, alpha=0.1)
    int2_recovered = ptq_int2_eval['loss'] - qad_int2_eval['loss']
    print(f"🌟 QAD (INT2) -> Loss: {qad_int2_eval['loss']:.4f} | PPL: {qad_int2_eval['ppl']:.2f} | Acc: {qad_int2_eval['accuracy']:.2f}% (Recovered: -{int2_recovered:.4f})")

    # =========================================================================
    # QUALITATIVE COMPARISON: Generate actual Vietnamese translations!
    # =========================================================================
    print("\n" + "=" * 70)
    print("🗣️ QUALITATIVE TRANSLATION SAMPLE COMPARISON:")
    print("=" * 70)
    test_prompts = [
        "What is it?",
        "We need to protect the environment.",
        "Artificial intelligence is changing the future of humanity.",
    ]

    teacher_gen = generate_translation(teacher, tokenizer, test_prompts, device)
    ptq_int4_gen = generate_translation(student_int4, tokenizer, test_prompts, device)
    ptq_int2_gen = generate_translation(student_int2, tokenizer, test_prompts, device)

    for i, p in enumerate(test_prompts):
        print(f"\n🇬🇧 English: \"{p}\"")
        print(f"   🇻🇳 Teacher (FP32) : {teacher_gen[i][1]}")
        print(f"   🇻🇳 INT4 (QAD)     : {ptq_int4_gen[i][1]}")
        print(f"   🇻🇳 INT2 (QAD)     : {ptq_int2_gen[i][1]}")

    # Summary table
    print("\n" + "=" * 70)
    print("📊 FINAL SUMMARY TABLE (INT4 vs INT2 TRANSLATION BENCHMARK):")
    print(f"   * 1. Teacher (FP32)    : Loss = {teacher_eval['loss']:.4f} | PPL = {teacher_eval['ppl']:.2f} | Acc = {teacher_eval['accuracy']:.2f}%")
    print(f"   * 2. PTQ (INT4)        : Loss = {ptq_int4_eval['loss']:.4f} | PPL = {ptq_int4_eval['ppl']:.2f} | Acc = {ptq_int4_eval['accuracy']:.2f}%")
    print(f"   * 3. QAD (INT4)        : Loss = {qad_int4_eval['loss']:.4f} | PPL = {qad_int4_eval['ppl']:.2f} | Acc = {qad_int4_eval['accuracy']:.2f}%")
    print(f"   * 4. PTQ (INT2)        : Loss = {ptq_int2_eval['loss']:.4f} | PPL = {ptq_int2_eval['ppl']:.2f} | Acc = {ptq_int2_eval['accuracy']:.2f}%")
    print(f"   * 5. QAD (INT2)        : Loss = {qad_int2_eval['loss']:.4f} | PPL = {qad_int2_eval['ppl']:.2f} | Acc = {qad_int2_eval['accuracy']:.2f}%")
    print("=" * 70)

    # Plot Comparison Chart
    os.makedirs("figures", exist_ok=True)
    models = ["Teacher\n(FP32)", "PTQ\n(INT4)", "QAD\n(INT4)", "PTQ\n(INT2)", "QAD\n(INT2)"]
    accs = [teacher_eval['accuracy'], ptq_int4_eval['accuracy'], qad_int4_eval['accuracy'], ptq_int2_eval['accuracy'], qad_int2_eval['accuracy']]
    colors = ["#2b5c8f", "#d95f02", "#1b9e77", "#e7298a", "#7570b3"]

    plt.figure(figsize=(9, 5), dpi=150)
    bars = plt.bar(models, accs, color=colors, width=0.55, edgecolor="black", linewidth=1.2)
    plt.ylabel("Token Prediction Accuracy (%)", fontsize=12, fontweight="bold")
    plt.title("Machine Translation (EN -> VI) Quantization: INT4 vs INT2", fontsize=13, fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        plt.annotate(f"{h:.2f}%", (bar.get_x() + bar.get_width() / 2, h),
                     textcoords="offset points", xytext=(0, 4), ha="center", va="bottom", fontweight="bold")

    chart_path = "figures/translation_int4_int2_benchmark.png"
    plt.tight_layout()
    plt.savefig(chart_path)
    plt.close()
    print(f"Chart saved to: {chart_path}")


if __name__ == "__main__":
    main()
