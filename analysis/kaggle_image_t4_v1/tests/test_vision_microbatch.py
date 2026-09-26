"""CPU-only scheduling checks; these are not checkpoint numerical parity."""

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
from transformers import SiglipVisionConfig, SiglipVisionModel  # noqa: E402

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from inference_core import ContractError  # noqa: E402
from submission_runtime import install_vision_microbatch  # noqa: E402


@pytest.fixture
def case():
    torch.manual_seed(31)
    config = SiglipVisionConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, image_size=28, patch_size=14,
        attention_dropout=0.0, vision_use_head=False,
    )
    config._attn_implementation = "eager"
    tower = SiglipVisionModel(config).eval()
    owner = torch.nn.Module()
    owner.vision_tower = tower
    owner.multi_modal_projector = torch.nn.Linear(32, 16).eval()
    base = SimpleNamespace(config=SimpleNamespace(model_type="gemma3"), model=owner)
    peft_model = SimpleNamespace(get_base_model=lambda: base)
    pixels = torch.randn(6, 3, 28, 28)
    return peft_model, owner, tower, pixels


def test_real_siglip_all_six_images_and_projector_match_batch(case):
    model, owner, tower, pixels = case
    projector = owner.multi_modal_projector
    parameters = tuple(tower.parameters())
    original_values = [p.detach().clone() for p in parameters]
    original_pixels = pixels.clone()
    with torch.inference_mode():
        expected_hidden = tower(pixel_values=pixels).last_hidden_state
        expected_projected = projector(expected_hidden)
    receipt = install_vision_microbatch(model, 1)
    with torch.inference_mode():
        actual = owner.vision_tower(pixel_values=pixels)
        actual_projected = projector(actual.last_hidden_state)
    assert actual.last_hidden_state.shape == (6, 4, 32)
    assert actual.pooler_output is None
    assert actual.last_hidden_state.dtype == expected_hidden.dtype
    torch.testing.assert_close(actual.last_hidden_state, expected_hidden, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual_projected, expected_projected, rtol=1e-5, atol=1e-5)
    assert owner.multi_modal_projector is projector
    assert owner.vision_tower.wrapped is tower
    assert tuple(map(id, owner.vision_tower.parameters())) == tuple(map(id, parameters))
    assert all(torch.equal(before, after) for before, after in zip(original_values, parameters))
    assert torch.equal(pixels, original_pixels)
    assert receipt["all_images_preserved_in_order"] is True
    assert receipt["projector_unchanged"] is True
    assert receipt["study_batch_size"] == 1


def test_original_module_hooks_receive_exact_ordered_single_image_views(case):
    model, owner, tower, pixels = case
    seen = []
    def observe(module, args, kwargs):
        value = kwargs["pixel_values"]
        seen.append((value.detach().clone(), value.data_ptr(), value.dtype))
    handle = tower.register_forward_pre_hook(observe, with_kwargs=True)
    install_vision_microbatch(model, 1)
    with torch.inference_mode():
        owner.vision_tower(pixel_values=pixels)
    handle.remove()
    assert len(seen) == 6
    for index, (value, pointer, dtype) in enumerate(seen):
        assert value.shape == (1, 3, 28, 28)
        assert torch.equal(value, pixels[index:index + 1])
        assert pointer == pixels[index:index + 1].data_ptr()
        assert dtype == pixels.dtype


def test_cpu_fp16_inputs_and_outputs_are_not_upcast_by_wrapper(case):
    model, owner, tower, pixels = case
    tower.to(dtype=torch.float16)  # Tiny synthetic CPU fixture, never the quantized model.
    pixels = pixels.to(dtype=torch.float16)
    install_vision_microbatch(model, 1)
    with torch.inference_mode():
        output = owner.vision_tower(pixel_values=pixels)
    assert output.last_hidden_state.dtype == torch.float16
    assert all(p.dtype == torch.float16 for p in tower.parameters())


@pytest.mark.parametrize("chunk_size", [0, 2, None, "1", 1.0, True])
def test_unreviewed_chunk_configuration_is_rejected(case, chunk_size):
    model, owner, tower, _ = case
    with pytest.raises(ContractError, match="one-image"):
        install_vision_microbatch(model, chunk_size)
    assert owner.vision_tower is tower


def test_wrong_model_or_vision_architecture_is_rejected(case):
    model, owner, _, _ = case
    base = model.get_base_model()
    base.config.model_type = "other"
    with pytest.raises(ContractError, match="Gemma3"):
        install_vision_microbatch(model, 1)
    base.config.model_type = "gemma3"
    owner.vision_tower.config.model_type = "other"
    with pytest.raises(ContractError, match="SigLIP"):
        install_vision_microbatch(model, 1)


def test_install_rejects_training_tower(case):
    model, _, tower, _ = case
    tower.train()
    with pytest.raises(ContractError, match="eval-mode"):
        install_vision_microbatch(model, 1)


def test_forward_rejects_grad_enabled_and_training(case):
    model, owner, tower, pixels = case
    install_vision_microbatch(model, 1)
    with torch.enable_grad(), pytest.raises(ContractError, match="inference-only"):
        owner.vision_tower(pixel_values=pixels)
    owner.vision_tower.train()
    with torch.inference_mode(), pytest.raises(ContractError, match="inference-only"):
        owner.vision_tower(pixel_values=pixels)
    owner.vision_tower.eval()
    tower.train()
    with torch.inference_mode(), pytest.raises(ContractError, match="inference-only"):
        owner.vision_tower(pixel_values=pixels)


@pytest.mark.parametrize("history", ["output_attentions", "output_hidden_states"])
def test_history_collection_is_rejected(case, history):
    model, owner, _, pixels = case
    install_vision_microbatch(model, 1)
    with torch.inference_mode(), pytest.raises(ContractError, match="histories"):
        owner.vision_tower(pixel_values=pixels, **{history: True})


@pytest.mark.parametrize("shape", [(0, 3, 28, 28), (3, 28, 28)])
def test_empty_or_non_image_batch_is_rejected(case, shape):
    model, owner, _, _ = case
    install_vision_microbatch(model, 1)
    with torch.inference_mode(), pytest.raises(ContractError, match="nonempty"):
        owner.vision_tower(pixel_values=torch.empty(shape))


def test_accelerate_forward_hook_on_original_tower_is_preserved(case):
    pytest.importorskip("accelerate")
    from accelerate.hooks import ModelHook, add_hook_to_module, remove_hook_from_module
    model, owner, tower, pixels = case
    class CountingHook(ModelHook):
        def __init__(self):
            self.inputs = []
        def pre_forward(self, module, *args, **kwargs):
            self.inputs.append(kwargs["pixel_values"].detach().clone())
            return args, kwargs
    hook = CountingHook()
    add_hook_to_module(tower, hook)
    install_vision_microbatch(model, 1)
    assert owner.vision_tower.wrapped._hf_hook is hook
    with torch.inference_mode():
        owner.vision_tower(pixel_values=pixels)
    assert len(hook.inputs) == 6
    assert torch.equal(torch.cat(hook.inputs), pixels)
    remove_hook_from_module(tower)


def test_unexpected_pooling_head_is_rejected(case):
    model, owner, tower, pixels = case
    def inject_pooling(module, args, output):
        output.pooler_output = output.last_hidden_state[:, 0]
        return output
    handle = tower.register_forward_hook(inject_pooling)
    install_vision_microbatch(model, 1)
    with torch.inference_mode(), pytest.raises(ContractError, match="pooling head"):
        owner.vision_tower(pixel_values=pixels)
    handle.remove()
