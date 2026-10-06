"""
Unit tests for model quantization transformation and end-to-end forward/backward.
"""

import torch
from mini_qad.models import MiniTransformerLM
from mini_qad.modules import convert_to_quantized_model, QuantizedLinear
from mini_qad.quantizer import QuantFormat


def test_model_quantization_conversion():
    model = MiniTransformerLM(vocab_size=100, d_model=32, n_heads=2, n_layers=2, max_seq_len=16)

    # Check initially standard nn.Linear
    linear_count = sum(1 for m in model.modules() if isinstance(m, torch.nn.Linear))
    assert linear_count > 0

    # Convert
    convert_to_quantized_model(model, weight_format=QuantFormat.INT4)

    # Check that linear layers became QuantizedLinear
    qlinear_count = sum(1 for m in model.modules() if isinstance(m, QuantizedLinear))
    assert qlinear_count == linear_count

    # Test forward and backward pass
    inputs = torch.randint(0, 100, (2, 8))
    logits = model(inputs)
    assert logits.shape == (2, 8, 100)

    loss = logits.sum()
    loss.backward()

    # Verify gradients flow back to weights
    for m in model.modules():
        if isinstance(m, QuantizedLinear):
            assert m.weight.grad is not None
