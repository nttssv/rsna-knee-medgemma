"""Explicit FP32 residual-normalization policy and first-nonfinite tracing.

No weight replacement, clipping, score repair, or retry is performed here.
"""
from contextlib import contextmanager

import torch

from inference_core import ContractError


class FP32PostNorm(torch.nn.Module):
    """Preserve the original Gemma3 RMSNorm result instead of narrowing to FP16."""
    def __init__(self, original):
        super().__init__()
        self.original = original

    def forward(self, hidden_states):
        if self.training or self.original.training or torch.is_grad_enabled():
            raise ContractError("FP32 post-normalization is inference-only")
        # Original RMSNorm already computes in FP32 and returns its input dtype.
        # Passing FP32 preserves its result and promotes the following residual
        # addition. Original parameters and dispatch hooks remain intact.
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            return self.original(hidden_states.float())


class FP16PreNorm(FP32PostNorm):
    """Narrow only normalized branch inputs, never the wide residual stream."""
    def forward(self, hidden_states):
        normalized = super().forward(hidden_states)
        if not bool(torch.isfinite(normalized).all()):
            raise ContractError("Nonfinite normalized branch input before FP16 conversion")
        if float(normalized.abs().max()) > torch.finfo(torch.float16).max:
            raise ContractError("Normalized branch input exceeds FP16 range; refusing conversion")
        # Preserve the existing FP16 branch inputs and NF4 compute recipe,
        # while the independent residual remains FP32. Validate the actual
        # quantized compute dtype separately after the diagnostic.
        return normalized.to(torch.float16)


def install_fp32_post_norms(model):
    from transformers.models.gemma3.modeling_gemma3 import Gemma3RMSNorm

    base = model.get_base_model()
    if base.config.model_type != "gemma3":
        raise ContractError("FP32 residual policy requires the pinned Gemma3 architecture")
    layers = base.model.language_model.layers
    if len(layers) != 34:
        raise ContractError("FP32 residual policy requires exactly 34 decoder layers")
    targets = []
    for index, layer in enumerate(layers):
        for name in ("post_attention_layernorm", "post_feedforward_layernorm",
                     "input_layernorm", "pre_feedforward_layernorm"):
            original = getattr(layer, name)
            if type(original) is not Gemma3RMSNorm or original.training:
                raise ContractError(f"Unexpected normalization at decoder {index}/{name}")
            targets.append((layer, name, original, f"model.language_model.layers.{index}.{name}"))
    for layer, name, original, _ in targets:
        wrapper = FP32PostNorm if name.startswith("post_") else FP16PreNorm
        setattr(layer, name, wrapper(original).eval())
    return {
        "policy": "gemma3_post_norm_and_residual_fp32_v1",
        "modules": [path for _, _, _, path in targets],
        "module_count": len(targets), "post_norm_count": 68, "pre_norm_count": 68,
        "post_norm_output_dtype": "torch.float32",
        "residual_addition_dtype": "torch.float32",
        "quantized_linear_compute_dtype": "torch.float16",
        "normalized_branch_input_dtype": "torch.float16",
        "weights_changed": False,
        "score_clipping_or_repair": False,
    }


def verify_quantized_compute_dtype(model):
    import bitsandbytes as bnb
    inventory = {}
    for name, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit):
            dtype = str(module.compute_dtype)
            inventory[dtype] = inventory.get(dtype, 0) + 1
            if module.compute_dtype != torch.float16:
                raise ContractError(f"NF4 compute dtype changed at {name}: {dtype}")
    if not inventory:
        raise ContractError("No NF4 modules found for compute dtype verification")
    return inventory


def _activation(output):
    """Inspect the activation, not masks/caches/optional attention histories."""
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, (tuple, list)) and output:
        return _activation(output[0])
    return None


class ActivationTrace:
    """One diagnostic forward; stop at the first observed nonfinite module output."""
    def __init__(self, model):
        self.model = model
        self.events = []
        self.first_nonfinite = None
        self.handles = []
        self.used = False

    def _hook(self, name):
        def observe(module, _args, output):
            value = _activation(output)
            if value is None or not value.is_floating_point() or value.numel() == 0:
                return
            finite = bool(torch.isfinite(value).all())
            row = {
                "sequence": len(self.events), "module": name,
                "module_type": type(module).__name__, "dtype": str(value.dtype),
                "device": str(value.device), "shape": list(value.shape),
                "finite": finite,
                "absmax": float(value.detach().abs().max()) if finite else None,
            }
            self.events.append(row)
            if not finite:
                row.update(nan_count=int(torch.isnan(value).sum()),
                           inf_count=int(torch.isinf(value).sum()))
                self.first_nonfinite = row
                raise ContractError(f"First nonfinite activation: {name} ({type(module).__name__}, {value.dtype})")
        return observe

    @contextmanager
    def capture(self):
        if self.used:
            raise ContractError("Activation trace cannot be reused for a retry")
        self.used = True
        # Inspect leaf operations and computation blocks, including residual
        # decoder outputs. Hooks record metadata only, never retained tensors.
        for name, module in self.model.named_modules():
            self.handles.append(module.register_forward_hook(self._hook(name or "<model>")))
        try:
            yield self
        finally:
            for handle in self.handles:
                handle.remove()
            self.handles.clear()

    def summary(self):
        return {"observed_module_calls": len(self.events),
                "first_nonfinite": self.first_nonfinite,
                "scope": "first diagnostic forward module outputs; not every functional operation"}

    def record(self):
        return {**self.summary(), "events": self.events}
