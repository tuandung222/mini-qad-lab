"""
Mini QAD Lab: Quantization-Aware Distillation (QAD) on Apple Silicon (MPS) & PyTorch.
"""

from .quantizer import FakeQuantizer, QuantFormat, quantize_ste
from .modules import (
    QuantizedLinear,
    QuantizedConv1D,
    convert_to_quantized_model,
    calibrate_model,
    freeze_quantizer_scales,
)
from .distill import QADLoss
from .models import MiniTransformerLM
from .trainer import QADTrainer, evaluate_model, get_default_device
from .dataset import SyntheticLanguageDataset, get_dataloaders, RealInstructionDataset, get_real_dataloaders
from .visualizer import plot_benchmark_comparison, plot_learning_curves

__version__ = "0.1.0"

__all__ = [
    "FakeQuantizer",
    "QuantFormat",
    "quantize_ste",
    "QuantizedLinear",
    "QuantizedConv1D",
    "convert_to_quantized_model",
    "calibrate_model",
    "freeze_quantizer_scales",
    "QADLoss",
    "MiniTransformerLM",
    "QADTrainer",
    "evaluate_model",
    "get_default_device",
    "SyntheticLanguageDataset",
    "get_dataloaders",
    "RealInstructionDataset",
    "get_real_dataloaders",
    "plot_benchmark_comparison",
    "plot_learning_curves",
]
