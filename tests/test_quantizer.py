"""
Unit tests for FakeQuantizer and Straight-Through Estimator (STE).
"""

import torch
import pytest
from mini_qad.quantizer import FakeQuantizer, QuantFormat, quantize_ste


def test_ste_gradient_flow():
    """Verify that gradients pass through the round function unhindered."""
    x = torch.tensor([1.2, 2.7, -3.4], requires_grad=True)
    y = quantize_ste(x)

    loss = (y**2).sum()
    loss.backward()

    # STE gradient: d(round(x))/dx == 1, so d(y^2)/dx == 2*y == 2*round(x)
    expected_grad = 2.0 * torch.round(torch.tensor([1.2, 2.7, -3.4]))
    assert torch.allclose(x.grad, expected_grad)


def test_fake_quantizer_int8():
    quantizer = FakeQuantizer(format=QuantFormat.INT8, per_channel=False)
    x = torch.randn(10, 20) * 5.0
    x_q = quantizer(x)

    assert x_q.shape == x.shape
    # Check that reconstructed values are discrete multiples of scale
    scale = quantizer.scale.item()
    steps = torch.round(x_q / scale)
    assert torch.allclose(x_q, steps * scale, atol=1e-5)
    assert steps.min().item() >= quantizer.qmin
    assert steps.max().item() <= quantizer.qmax


def test_fake_quantizer_int4():
    quantizer = FakeQuantizer(format=QuantFormat.INT4, per_channel=True)
    x = torch.randn(8, 16) * 10.0
    x_q = quantizer(x)

    assert x_q.shape == x.shape
    assert quantizer.scale.shape[0] == 8  # per-channel along dim 0


def test_fake_quantizer_int2():
    quantizer = FakeQuantizer(format=QuantFormat.INT2, per_channel=False, symmetric=False)
    x = torch.randn(10, 10) * 4.0
    x_q = quantizer(x)

    assert x_q.shape == x.shape
    scale = quantizer.scale.item()
    steps = torch.round(x_q / scale)
    assert steps.min().item() >= quantizer.qmin
    assert steps.max().item() <= quantizer.qmax

