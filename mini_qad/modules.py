"""
Quantized PyTorch layers and model transformation utilities.
Enables transparent conversion of nn.Linear layers to Fake-Quantized layers.
"""

from typing import Dict, List, Optional, Set, Type
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantizer import FakeQuantizer, QuantFormat


class QuantizedLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with simulated weight and activation quantization.

    Args:
        in_features: Size of each input sample
        out_features: Size of each output sample
        bias: If set to False, the layer will not learn an additive bias
        weight_format: Quantization format for weights (INT8, INT4, FP8)
        act_format: Quantization format for activations (optional, None for weight-only)
        per_channel_weight: If True, uses per-output-channel quantization for weights
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        weight_format: QuantFormat = QuantFormat.INT8,
        act_format: Optional[QuantFormat] = None,
        per_channel_weight: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.weight_quantizer = FakeQuantizer(
            format=weight_format,
            per_channel=per_channel_weight,
            symmetric=True,
            ch_axis=0,
        )

        if act_format is not None:
            self.act_quantizer: Optional[FakeQuantizer] = FakeQuantizer(
                format=act_format,
                per_channel=False,
                symmetric=True,
            )
        else:
            self.act_quantizer = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in**0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    @classmethod
    def from_float(
        cls,
        linear: nn.Linear,
        weight_format: QuantFormat = QuantFormat.INT8,
        act_format: Optional[QuantFormat] = None,
        per_channel_weight: bool = True,
    ) -> "QuantizedLinear":
        """Instantiate a QuantizedLinear from an existing nn.Linear module."""
        device = linear.weight.device
        dtype = linear.weight.dtype
        qlinear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            weight_format=weight_format,
            act_format=act_format,
            per_channel_weight=per_channel_weight,
        ).to(device=device, dtype=dtype)
        with torch.no_grad():
            qlinear.weight.copy_(linear.weight)
            if linear.bias is not None:
                qlinear.bias.copy_(linear.bias)
            # Calibrate weight quantizer immediately with original weights
            qlinear.weight_quantizer.calibrate(qlinear.weight)
        return qlinear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantize activations if configured
        if self.act_quantizer is not None:
            x = self.act_quantizer(x)

        # Quantize weights using STE
        w_q = self.weight_quantizer(self.weight)

        return F.linear(x, w_q, self.bias)

    def extra_repr(self) -> str:
        s = f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"
        s += f", weight_quant={self.weight_quantizer.format.value}"
        if self.act_quantizer is not None:
            s += f", act_quant={self.act_quantizer.format.value}"
        return s


def convert_to_quantized_model(
    model: nn.Module,
    weight_format: QuantFormat = QuantFormat.INT8,
    act_format: Optional[QuantFormat] = None,
    per_channel_weight: bool = True,
    exclude_modules: Optional[Set[str]] = None,
) -> nn.Module:
    """
    Recursively replaces all nn.Linear layers in model with QuantizedLinear.
    Returns a new or modified model.
    """
    if exclude_modules is None:
        exclude_modules = set()

    for name, child in model.named_children():
        if name in exclude_modules:
            continue

        if isinstance(child, nn.Linear) and not isinstance(child, QuantizedLinear):
            q_child = QuantizedLinear.from_float(
                child,
                weight_format=weight_format,
                act_format=act_format,
                per_channel_weight=per_channel_weight,
            )
            setattr(model, name, q_child)
        else:
            convert_to_quantized_model(
                child,
                weight_format=weight_format,
                act_format=act_format,
                per_channel_weight=per_channel_weight,
                exclude_modules=exclude_modules,
            )
    return model


def calibrate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_batches: int = 16,
):
    """
    Perform PTQ calibration pass on model activations.
    """
    model.eval()
    # Enable calibration mode on all FakeQuantizer modules
    for m in model.modules():
        if isinstance(m, FakeQuantizer):
            m.calibrating = True

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            if isinstance(batch, (tuple, list)):
                inputs = batch[0].to(device)
            elif isinstance(batch, dict):
                inputs = {k: v.to(device) for k, v in batch.items()}
            else:
                inputs = batch.to(device)

            if isinstance(inputs, dict):
                _ = model(**inputs)
            else:
                _ = model(inputs)

    # Disable calibration mode
    for m in model.modules():
        if isinstance(m, FakeQuantizer):
            m.calibrating = False


def freeze_quantizer_scales(model: nn.Module):
    """Freeze all quantizer scales so only weights adapt during QAT/QAD."""
    for m in model.modules():
        if isinstance(m, FakeQuantizer):
            m.calibrating = False
            # Ensure buffer scale is frozen
            m.is_calibrated.copy_(torch.tensor(True, dtype=torch.bool))
