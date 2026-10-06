"""
Bridge utility: Export Mini QAD checkpoints and configuration metadata
in standard NVIDIA ModelOpt format (`hf_quant_config.json`).
"""

import sys
import os
import json
import torch

# Ensure package is found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mini_qad import MiniTransformerLM, QuantFormat, convert_to_quantized_model, FakeQuantizer


def export_modelopt_config(model: torch.nn.Module, export_dir: str = "exported_model"):
    """
    Exports quantization metadata compatible with NVIDIA ModelOpt / Hugging Face.
    """
    os.makedirs(export_dir, exist_ok=True)

    quant_layers = {}
    for name, module in model.named_modules():
        if isinstance(module, FakeQuantizer):
            quant_layers[name] = {
                "format": module.format.value,
                "per_channel": module.per_channel,
                "symmetric": module.symmetric,
                "qmax": module.qmax,
                "scale_shape": list(module.scale.shape),
            }

    # ModelOpt-compatible format
    hf_quant_config = {
        "producer": {
            "name": "mini-qad-lab",
            "version": "0.1.0",
        },
        "quantization": {
            "quant_algo": "NVFP4_SIMULATED" if "fp8" in str(quant_layers) else "INT4_QAD",
            "kv_cache_quant_algo": None,
            "exclude_modules": [],
            "quantized_layers_count": len(quant_layers),
            "layers": quant_layers,
        },
    }

    config_path = os.path.join(export_dir, "hf_quant_config.json")
    with open(config_path, "w") as f:
        json.dump(hf_quant_config, f, indent=2)

    # Save weights dictionary
    weights_path = os.path.join(export_dir, "model_weights.pt")
    torch.save(model.state_dict(), weights_path)

    print(f"✅ Exported ModelOpt-compatible metadata to: {config_path}")
    print(f"✅ Exported model weights to: {weights_path}")


def main():
    print("Exporting sample quantized model to ModelOpt format...")
    model = MiniTransformerLM(vocab_size=200, d_model=64, n_heads=2, n_layers=2)
    convert_to_quantized_model(model, weight_format=QuantFormat.INT4)
    export_modelopt_config(model, "exported_modelopt_sample")


if __name__ == "__main__":
    main()
