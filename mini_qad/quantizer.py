"""
Quantization engine with Straight-Through Estimator (STE) for PyTorch.
Supports Apple Silicon (MPS), CUDA, and CPU.
"""

from enum import Enum
from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch.autograd import Function


class QuantFormat(str, Enum):
    INT8 = "int8"
    INT4 = "int4"
    INT2 = "int2"
    FP8_E4M3 = "fp8_e4m3"


class RoundSTE(Function):
    """
    Straight-Through Estimator (STE) for the round() operation.
    Forward: returns round(x)
    Backward: passes through the gradient dL/dy directly (identity gradient).
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


def quantize_ste(x: torch.Tensor) -> torch.Tensor:
    return RoundSTE.apply(x)


class FakeQuantizer(nn.Module):
    """
    Fake Quantizer module that applies simulated quantization using Straight-Through Estimator.

    During forward pass:
        x_q = round(clamp(x / scale, qmin, qmax)) * scale
    During backward pass:
        Gradients pass straight through to input weights.

    Parameters:
        format: QuantFormat or str ('int8', 'int4', 'fp8_e4m3')
        per_channel: Whether to use per-channel (dim=0) or per-tensor quantization
        symmetric: Symmetric quantization centered around 0
        ch_axis: Channel axis for per-channel quantization (default: 0 for Linear weights)
    """

    def __init__(
        self,
        format: QuantFormat = QuantFormat.INT8,
        per_channel: bool = False,
        symmetric: bool = True,
        ch_axis: int = 0,
    ):
        super().__init__()
        self.format = QuantFormat(format)
        self.per_channel = per_channel
        self.symmetric = symmetric
        self.ch_axis = ch_axis

        # Define bounds according to format
        if self.format == QuantFormat.INT8:
            self.qmin = -128.0 if not symmetric else -127.0
            self.qmax = 127.0
        elif self.format == QuantFormat.INT4:
            self.qmin = -8.0 if not symmetric else -7.0
            self.qmax = 7.0
        elif self.format == QuantFormat.INT2:
            # 2-bit quantization (4 discrete levels)
            self.qmin = -2.0 if not symmetric else -1.0
            self.qmax = 1.0
        elif self.format == QuantFormat.FP8_E4M3:
            # Simulated FP8 E4M3 range: [-448, 448]
            self.qmin = -448.0
            self.qmax = 448.0
        else:
            raise ValueError(f"Unsupported format: {format}")

        self.register_buffer("scale", torch.tensor(1.0))
        self.register_buffer("is_calibrated", torch.tensor(False, dtype=torch.bool))
        self.calibrating = False
        self.enabled = True

    def calculate_scale(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate quantization scale from tensor statistics."""
        with torch.no_grad():
            if self.per_channel:
                # Reduce over all dims except ch_axis
                dims = [i for i in range(x.dim()) if i != self.ch_axis]
                max_val = x.abs().amax(dim=dims, keepdim=True)
            else:
                max_val = x.abs().max()

            # Prevent division by zero
            max_val = torch.clamp(max_val, min=1e-8)
            scale = max_val / self.qmax
            return scale

    def calibrate(self, x: torch.Tensor):
        """Update scale factor based on observed tensor."""
        new_scale = self.calculate_scale(x)
        self.register_buffer("scale", new_scale)
        self.is_calibrated.copy_(torch.tensor(True, dtype=torch.bool))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x

        # If calibrating or uncalibrated, compute scale
        if self.calibrating or not self.is_calibrated.item():
            self.calibrate(x)

        # Apply simulated quantization with STE
        scale = self.scale.to(device=x.device, dtype=x.dtype)
        x_scaled = x / scale
        x_clamped = torch.clamp(x_scaled, min=self.qmin, max=self.qmax)
        x_quant = quantize_ste(x_clamped)
        x_dequant = x_quant * scale
        return x_dequant

    def extra_repr(self) -> str:
        return (
            f"format={self.format.value}, per_channel={self.per_channel}, "
            f"symmetric={self.symmetric}, qmin={self.qmin}, qmax={self.qmax}, "
            f"calibrated={self.is_calibrated.item()}"
        )
