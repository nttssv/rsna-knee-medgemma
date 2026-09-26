"""Synthetic CPU diagnostics; these do not establish BF16 or report parity."""

import json
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from transformers.models.gemma3.modeling_gemma3 import Gemma3RMSNorm  # noqa: E402

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from inference_core import ContractError  # noqa: E402
from numerical_runtime import (  # noqa: E402
    ActivationTrace, FP16PreNorm, FP32PostNorm, install_fp32_post_norms,
    verify_quantized_compute_dtype,
)


def _norm(size=8):
    return Gemma3RMSNorm(size, eps=1e-6).eval()


@pytest.fixture
def model_case():
    owner = torch.nn.Module()
    owner.language_model = torch.nn.Module()
    owner.language_model.layers = torch.nn.ModuleList()
    for _ in range(34):
        layer = torch.nn.Module()
        for name in ("input_layernorm", "pre_feedforward_layernorm",
                     "post_attention_layernorm", "post_feedforward_layernorm"):
            setattr(layer, name, _norm())
        layer.mlp = torch.nn.Linear(8, 8).eval()
        owner.language_model.layers.append(layer)
    owner.vision_tower = torch.nn.Linear(8, 8).eval()
    owner.multi_modal_projector = torch.nn.Linear(8, 8).eval()
    owner.eval()
    base = SimpleNamespace(config=SimpleNamespace(model_type="gemma3"), model=owner)
    model = SimpleNamespace(get_base_model=lambda: base)
    return model, owner, base


def test_exact_68_post_norms_preserve_every_parameter_and_other_module(model_case):
    model, owner, _ = model_case
    modules = dict(owner.named_modules())
    parameters = tuple(owner.parameters())
    values = tuple(p.detach().clone() for p in parameters)
    receipt = install_fp32_post_norms(model)
    assert receipt["module_count"] == 136
    assert receipt["post_norm_count"] == receipt["pre_norm_count"] == 68
    assert len(set(receipt["modules"])) == 136
    assert receipt["quantized_linear_compute_dtype"] == "torch.float16"
    assert receipt["weights_changed"] is False
    assert receipt["score_clipping_or_repair"] is False
    for name, original in modules.items():
        current = owner.get_submodule(name) if name else owner
        if name.endswith(("post_attention_layernorm", "post_feedforward_layernorm")):
            assert type(current) is FP32PostNorm
            assert current.original is original
            assert current.original.weight is original.weight
        elif name.endswith(("input_layernorm", "pre_feedforward_layernorm")):
            assert type(current) is FP16PreNorm
            assert current.original is original
            assert current.original.weight is original.weight
        else:
            assert current is original
    assert tuple(map(id, owner.parameters())) == tuple(map(id, parameters))
    assert all(torch.equal(before, after) for before, after in zip(values, parameters))


@pytest.mark.parametrize("corruption", ["architecture", "count", "last_type", "last_training"])
def test_installer_rejects_invalid_architecture_atomically(model_case, corruption):
    model, owner, base = model_case
    layers = owner.language_model.layers
    if corruption == "architecture":
        base.config.model_type = "other"
    elif corruption == "count":
        del layers[-1]
    elif corruption == "last_type":
        layers[-1].post_feedforward_layernorm = torch.nn.LayerNorm(8).eval()
    else:
        layers[-1].post_feedforward_layernorm.train()
    original = dict(owner.named_modules())
    with pytest.raises(ContractError):
        install_fp32_post_norms(model)
    assert all(owner.get_submodule(name) is module for name, module in original.items() if name)
    assert not any(isinstance(m, FP32PostNorm) for m in owner.modules())


def test_installer_cannot_silently_wrap_a_second_time(model_case):
    model, owner, _ = model_case
    install_fp32_post_norms(model)
    original = dict(owner.named_modules())
    with pytest.raises(ContractError, match="Unexpected normalization"):
        install_fp32_post_norms(model)
    assert all(owner.get_submodule(name) is module for name, module in original.items() if name)


def _actual_norm_weight_or_synthetic():
    """Read one small BF16 tensor, never load a model or disclose report data."""
    root = Path(__file__).resolve().parents[3]
    paths = sorted((root / "state/kaggle_image_baseline_v1/assets/base_model").glob("*.safetensors"))
    target = "language_model.model.layers.33.post_feedforward_layernorm.weight"
    for path in paths:
        with path.open("rb") as stream:
            header_size = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(header_size))
            if target not in header:
                continue
            info = header[target]
            assert info["dtype"] == "BF16" and info["shape"] == [2560]
            begin, end = info["data_offsets"]
            stream.seek(8 + header_size + begin)
            return torch.frombuffer(bytearray(stream.read(end - begin)), dtype=torch.bfloat16).float().clone()
    weight = torch.zeros(2560)
    weight[0] = 1376.0
    return weight


def test_synthetic_input_post_norm_overflow_is_prevented_without_clipping():
    # Uses actual checkpoint gamma if private assets exist, synthetic gamma in CI.
    # The activation remains synthetic in both cases, never an observed MRI result.
    weight = _actual_norm_weight_or_synthetic()
    original = _norm(len(weight))
    with torch.no_grad():
        original.weight.copy_(weight)
    wrapper = FP32PostNorm(original).eval()
    x = torch.zeros(1, len(weight), dtype=torch.float16)
    x[0, int(weight.argmax())] = 100
    with torch.inference_mode():
        historical = original(x)
        expected = original(x.float())
        actual = wrapper(x)
    assert not torch.isfinite(historical).all()
    assert torch.isfinite(actual).all()
    assert actual.dtype == torch.float32
    assert actual.max().item() > torch.finfo(torch.float16).max
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(original.weight, weight)
    assert wrapper.original.weight is original.weight


def test_fp32_post_norm_output_promotes_the_unchanged_residual_addition():
    original = _norm(1)
    with torch.no_grad():
        original.weight.fill_(39999)
    wrapper = FP32PostNorm(original).eval()
    residual = torch.tensor([[40000]], dtype=torch.float16)
    with torch.inference_mode():
        old = residual + original(residual)
        actual = residual + wrapper(residual)
    assert not torch.isfinite(old).all()
    assert actual.dtype == torch.float32
    assert actual.item() == 80000


def test_wrapper_preserves_original_dispatch_hooks():
    original = _norm()
    seen = []
    handle = original.register_forward_pre_hook(lambda _m, args: seen.append(args[0].dtype))
    wrapper = FP32PostNorm(original).eval()
    with torch.inference_mode():
        wrapper(torch.ones(1, 8, dtype=torch.float16))
    handle.remove()
    assert seen == [torch.float32]


def test_pre_norm_safely_narrows_only_normalized_branch_not_residual():
    original = _norm()
    wrapper = FP16PreNorm(original).eval()
    residual = torch.full((1, 8), 80000.0)
    saved = residual.clone()
    with torch.inference_mode():
        expected = original(residual).to(torch.float16)
        actual = wrapper(residual)
    assert actual.dtype == torch.float16
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(residual, saved)
    assert residual.dtype == torch.float32 and residual.max().item() == 80000


@pytest.mark.parametrize("failure", ["nan", "inf", "out_of_range"])
def test_pre_norm_refuses_nonfinite_or_unrepresentable_values(failure):
    original = _norm(1)
    with torch.no_grad():
        original.weight.fill_(70000 if failure == "out_of_range" else 1)
    wrapper = FP16PreNorm(original).eval()
    value = 100 if failure == "out_of_range" else float(failure)
    with torch.inference_mode(), pytest.raises(ContractError, match="Nonfinite|exceeds FP16"):
        wrapper(torch.tensor([[value]]))


def test_actual_compute_dtype_inventory_rejects_drift_and_no_nf4(monkeypatch):
    class FakeLinear4bit(torch.nn.Module):
        def __init__(self, dtype):
            super().__init__()
            self.compute_dtype = dtype
    monkeypatch.setitem(sys.modules, "bitsandbytes", SimpleNamespace(nn=SimpleNamespace(Linear4bit=FakeLinear4bit)))
    model = torch.nn.Sequential(FakeLinear4bit(torch.float16), FakeLinear4bit(torch.float16))
    assert verify_quantized_compute_dtype(model) == {"torch.float16": 2}
    model[1].compute_dtype = torch.float32
    with pytest.raises(ContractError, match="NF4 compute dtype changed at 1"):
        verify_quantized_compute_dtype(model)
    with pytest.raises(ContractError, match="No NF4 modules"):
        verify_quantized_compute_dtype(torch.nn.Identity())


@pytest.mark.parametrize("violation", ["wrapper_training", "original_training", "grad_enabled"])
def test_wrapper_is_inference_only(violation):
    original = _norm()
    wrapper = FP32PostNorm(original).eval()
    if violation == "wrapper_training":
        wrapper.train()
    elif violation == "original_training":
        original.train()
    context = torch.enable_grad() if violation == "grad_enabled" else torch.inference_mode()
    with context, pytest.raises(ContractError, match="inference-only"):
        wrapper(torch.ones(1, 8))


def test_trace_stops_at_first_nonfinite_and_removes_hooks():
    class Bad(torch.nn.Module):
        def forward(self, x):
            return x * float("inf")
    class Never(torch.nn.Module):
        def forward(self, x):
            raise AssertionError("must not run after first nonfinite")
    model = torch.nn.Sequential(torch.nn.Identity(), Bad(), Never())
    trace = ActivationTrace(model)
    with pytest.raises(ContractError, match="First nonfinite activation: 1"):
        with trace.capture(), torch.inference_mode():
            model(torch.tensor([0.0, 1.0]))
    assert trace.first_nonfinite["module"] == "1"
    assert trace.first_nonfinite["nan_count"] == 1
    assert trace.first_nonfinite["inf_count"] == 1
    assert len(trace.events) == 2
    assert not trace.handles
    assert all(not module._forward_hooks for module in model.modules())
    # JSON serialization rejects tensor retention, NaN/Inf numbers, or opaque objects.
    json.dumps(trace.record(), allow_nan=False)


def test_trace_checks_residual_block_output_not_only_leaf_operations():
    class Residual(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = torch.nn.Identity()
        def forward(self, x):
            return x + self.norm(x)
    model = Residual()
    trace = ActivationTrace(model)
    with pytest.raises(ContractError, match="<model>"):
        with trace.capture(), torch.inference_mode():
            model(torch.tensor([40000], dtype=torch.float16))
    assert trace.events[0]["finite"] is True
    assert trace.first_nonfinite["module"] == "<model>"


def test_trace_ignores_masks_and_optional_attention_histories():
    class WithMask(torch.nn.Module):
        def forward(self, x, attention_mask):
            return x, attention_mask
    trace = ActivationTrace(WithMask())
    with trace.capture(), torch.inference_mode():
        trace.model(torch.ones(2), attention_mask=torch.full((2, 2), float("-inf")))
    assert len(trace.events) == 1
    assert trace.first_nonfinite is None
    assert trace.events[0]["shape"] == [2]
    json.dumps(trace.record(), allow_nan=False)


@pytest.mark.parametrize("output_kind", ["last_hidden_state", "logits", "list"])
def test_trace_understands_supported_output_containers(output_kind):
    class Container(torch.nn.Module):
        def forward(self, x):
            return [x] if output_kind == "list" else SimpleNamespace(**{output_kind: x})
    trace = ActivationTrace(Container())
    with trace.capture(), torch.inference_mode():
        trace.model(torch.ones(2))
    assert trace.summary()["observed_module_calls"] == 1
    assert trace.first_nonfinite is None


def test_trace_is_one_use_after_successful_forward():
    trace = ActivationTrace(torch.nn.Identity())
    with trace.capture(), torch.inference_mode():
        trace.model(torch.ones(2))
    with pytest.raises(ContractError, match="reused"):
        with trace.capture():
            pass


def test_trace_removes_hooks_after_unrelated_forward_exception():
    class Fails(torch.nn.Module):
        def forward(self, x):
            raise ValueError("unrelated failure")
    model = Fails()
    trace = ActivationTrace(model)
    with pytest.raises(ValueError, match="unrelated failure"):
        with trace.capture():
            model(torch.ones(2))
    assert not trace.handles and not model._forward_hooks
    with pytest.raises(ContractError, match="reused"):
        with trace.capture():
            pass
