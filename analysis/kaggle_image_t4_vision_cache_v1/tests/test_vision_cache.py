"""CPU inference-only cache regressions; tiny random SigLIP is SYNTHETIC.

These tests validate cache identity, lifecycle and exact preserved operations.
They are not measurements of real MedGemma score parity or T4 throughput.
"""
from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('transformers')
from transformers import SiglipVisionConfig, SiglipVisionModel  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
V8 = ROOT.parent / 'kaggle_image_t4_v1'
sys.path.insert(0, str(V8 / 'scripts'))
sys.path.insert(0, str(ROOT / 'scripts'))
from inference_core import ContractError, LABELS  # noqa: E402
from submission_runtime import ImageTeacher, install_vision_microbatch  # noqa: E402
from vision_cache import StudyVisionCache, install_cached_teacher  # noqa: E402


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(old)


@pytest.fixture
def case():
    torch.manual_seed(51)
    config = SiglipVisionConfig(hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=4, image_size=14, patch_size=7,
        attention_dropout=0.0, vision_use_head=False)
    config._attn_implementation = 'eager'
    tower = SiglipVisionModel(config).eval()
    owner = torch.nn.Module()
    owner.vision_tower = tower
    owner.multi_modal_projector = torch.nn.Linear(16, 8).eval()
    base = torch.nn.Module()
    base.config = SimpleNamespace(model_type='gemma3')
    base.model = owner

    class SyntheticModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = base
        def get_base_model(self):
            return self.base

    model = SyntheticModel().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    receipt = install_vision_microbatch(model, 1)
    pixels = torch.randn(6, 3, 14, 14)
    return model, owner, tower, pixels, receipt


def cache_for(case):
    _model, owner, _tower, _pixels, _receipt = case
    return StudyVisionCache(owner.vision_tower).eval()


def test_twelve_calls_reuse_exact_six_image_output_and_projector(case):
    model, owner, original_tower, pixels, _receipt = case
    baseline = owner.vision_tower
    projector = owner.multi_modal_projector
    parameter_ids = [id(parameter) for parameter in model.parameters()]
    state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    with torch.inference_mode():
        expected = baseline(pixel_values=pixels).last_hidden_state
        projected_expected = projector(expected)
    seen = []
    handle = original_tower.register_forward_pre_hook(
        lambda _module, _args, kwargs: seen.append(kwargs['pixel_values'].clone()), with_kwargs=True)
    cached = cache_for(case)
    cached.begin_study('SYNTHETIC-a')
    with torch.inference_mode():
        for _ in range(12):
            output = cached(pixel_values=pixels.clone())
            assert torch.equal(output.last_hidden_state, expected)
            assert torch.equal(projector(output.last_hidden_state), projected_expected)
            assert output.last_hidden_state.dtype == expected.dtype
            assert output.last_hidden_state.device == expected.device
            assert output.pooler_output is None
            assert output.hidden_states is None
            assert output.attentions is None
    handle.remove()
    assert len(seen) == 6
    assert torch.equal(torch.cat(seen), pixels)
    assert owner.multi_modal_projector is projector
    assert [id(parameter) for parameter in model.parameters()] == parameter_ids
    assert all(torch.equal(state[key], value) for key, value in model.state_dict().items())
    summary = cached.summary()
    assert summary['hits'] == 11
    assert summary['misses'] == 1
    assert summary['underlying_vision_calls'] == 1


def test_same_uid_keeps_slot_but_new_uid_and_clear_invalidate(case):
    cached = cache_for(case)
    pixels = case[3]
    with torch.inference_mode():
        cached.begin_study('SYNTHETIC-a')
        cached(pixel_values=pixels)
        cached.begin_study('SYNTHETIC-a')
        cached(pixel_values=pixels)
        cached.begin_study('SYNTHETIC-b')
        cached(pixel_values=pixels)
        cached.clear()
        cached.begin_study('SYNTHETIC-b')
        cached(pixel_values=pixels)
    assert cached.summary()['hits'] == 1
    assert cached.summary()['misses'] == 3
    assert cached.summary()['underlying_vision_calls'] == 3


def test_fp16_cache_preserves_dtype_and_exact_uncached_values(case):
    model, owner, _tower, pixels, _receipt = case
    # Tiny synthetic CPU model only; production quantized model is never cast.
    model.to(dtype=torch.float16)
    pixels = pixels.to(dtype=torch.float16)
    cached = cache_for(case)
    cached.begin_study('SYNTHETIC-a')
    with torch.inference_mode():
        expected = owner.vision_tower(pixel_values=pixels).last_hidden_state
        projected = owner.multi_modal_projector(expected)
        for _ in range(2):
            actual = cached(pixel_values=pixels).last_hidden_state
            assert actual.dtype == torch.float16
            assert torch.equal(actual, expected)
            assert torch.equal(owner.multi_modal_projector(actual), projected)
    assert cached.summary()['misses'] == 1
    assert cached.summary()['hits'] == 1


def test_returned_output_mutation_cannot_corrupt_cached_tensor(case):
    cached = cache_for(case)
    cached.begin_study('SYNTHETIC-a')
    with torch.inference_mode():
        first = cached(pixel_values=case[3])
        expected = first.last_hidden_state.clone()
        first.last_hidden_state.add_(1)
        second = cached(pixel_values=case[3])
        assert torch.equal(second.last_hidden_state, expected)
        second.last_hidden_state.zero_()
        third = cached(pixel_values=case[3])
        assert torch.equal(third.last_hidden_state, expected)
        assert third is not second
        assert third.last_hidden_state.data_ptr() != second.last_hidden_state.data_ptr()


def test_no_active_study_is_rejected_without_vision_work(case):
    cached = cache_for(case)
    with torch.inference_mode(), pytest.raises(ContractError, match='active study'):
        cached(pixel_values=case[3])
    assert cached.summary()['underlying_vision_calls'] == 0


@pytest.mark.parametrize('bad', [None, '', 12])
def test_explicit_nonempty_study_identity_required(case, bad):
    cached = cache_for(case)
    with pytest.raises(ContractError):
        cached.begin_study(bad)


@pytest.mark.parametrize('bad', ['empty', 'rank', 'nan', 'inf'])
def test_malformed_first_input_never_reaches_underlying_tower(case, bad):
    cached = cache_for(case)
    cached.begin_study('SYNTHETIC-a')
    pixels = case[3].clone()
    if bad == 'empty':
        pixels = pixels[:0]
    elif bad == 'rank':
        pixels = pixels[0]
    else:
        pixels[0, 0, 0, 0] = float(bad)
    with torch.inference_mode(), pytest.raises(ContractError):
        cached(pixel_values=pixels)
    assert cached.summary()['underlying_vision_calls'] == 0
    assert cached.summary()['cache_released'] is True


@pytest.mark.parametrize('bad', ['raises', 'nonfinite', 'pooling', 'image_count'])
def test_underlying_failure_does_not_populate_a_cache_or_retry(case, bad):
    cached = cache_for(case)
    cached.begin_study('SYNTHETIC-a')
    calls = []

    def corrupt(_module, _args, output):
        calls.append(bad)
        if bad == 'raises':
            raise ContractError('SYNTHETIC original tower failure')
        if bad == 'nonfinite':
            output.last_hidden_state[0, 0, 0] = float('nan')
        elif bad == 'pooling':
            output.pooler_output = output.last_hidden_state[:, 0]
        else:
            output.last_hidden_state = output.last_hidden_state[:-1]
        return output

    handle = case[1].vision_tower.register_forward_hook(corrupt)
    with torch.inference_mode(), pytest.raises(ContractError):
        cached(pixel_values=case[3])
    handle.remove()
    assert calls == [bad]
    assert cached.summary()['misses'] == 0
    assert cached.summary()['underlying_vision_calls'] == 1
    assert cached.summary()['cache_released'] is True
    assert cached.summary()['active_study'] is None


@pytest.mark.parametrize('drift', ['content', 'order', 'shape', 'dtype', 'device', 'kwargs', 'autocast'])
def test_same_study_input_or_precision_drift_fails_without_recompute(case, drift):
    cached = cache_for(case)
    cached.begin_study('SYNTHETIC-a')
    pixels = case[3]
    with torch.inference_mode():
        cached(pixel_values=pixels)
        changed = pixels.clone()
        kwargs = {}
        if drift == 'content':
            changed[0, 0, 0, 0] += 1
        elif drift == 'order':
            changed = changed.flip(0)
        elif drift == 'shape':
            changed = changed[:-1]
        elif drift == 'dtype':
            changed = changed.to(torch.float16)
        elif drift == 'device':
            changed = changed.to('meta')
        elif drift == 'kwargs':
            kwargs = {'output_attentions': False}
        if drift == 'autocast':
            with torch.autocast('cpu', dtype=torch.bfloat16), pytest.raises(ContractError):
                cached(pixel_values=changed)
        else:
            with pytest.raises(ContractError):
                cached(pixel_values=changed, **kwargs)
    assert cached.summary()['underlying_vision_calls'] == 1
    assert cached.summary()['cache_released'] is True
    assert cached.summary()['active_study'] is None


def test_input_mutation_is_detected_against_owned_snapshot(case):
    cached = cache_for(case)
    cached.begin_study('SYNTHETIC-a')
    pixels = case[3].clone()
    with torch.inference_mode():
        cached(pixel_values=pixels)
        pixels.add_(0.25)
        with pytest.raises(ContractError):
            cached(pixel_values=pixels)
    assert cached.summary()['underlying_vision_calls'] == 1


@pytest.mark.parametrize('drift', ['signed_zero', 'stride'])
def test_byte_or_layout_drift_is_not_hidden_by_numeric_equality(case, drift):
    cached = cache_for(case)
    cached.begin_study('SYNTHETIC-a')
    pixels = case[3].clone()
    pixels[0, 0, 0, 0] = 0.0
    with torch.inference_mode():
        cached(pixel_values=pixels)
        if drift == 'signed_zero':
            changed = pixels.clone()
            changed[0, 0, 0, 0] = -0.0
            assert not torch.equal(changed.view(torch.uint8), pixels.view(torch.uint8))
        else:
            changed = pixels.transpose(2, 3).contiguous().transpose(2, 3)
            assert changed.stride() != pixels.stride()
        assert torch.equal(changed, pixels), 'The regression must defeat value-only equality'
        with pytest.raises(ContractError):
            cached(pixel_values=changed)
    assert cached.summary()['underlying_vision_calls'] == 1
    assert cached.summary()['cache_released'] is True


@pytest.mark.parametrize('mode', ['grad', 'no_grad_only', 'wrapper_training', 'submodule_training'])
def test_cache_rejects_non_inference_or_training_context(case, mode):
    cached = cache_for(case)
    cached.begin_study('SYNTHETIC-a')
    if mode == 'grad':
        with torch.enable_grad(), pytest.raises(ContractError):
            cached(pixel_values=case[3])
    elif mode == 'no_grad_only':
        with torch.no_grad(), pytest.raises(ContractError):
            cached(pixel_values=case[3])
    else:
        if mode == 'wrapper_training':
            cached.train()
        else:
            case[2].vision_model.encoder.layers[0].train()
        with torch.inference_mode(), pytest.raises(ContractError):
            cached(pixel_values=case[3])
    assert cached.summary()['underlying_vision_calls'] == 0


def test_trainable_weights_rejected(case):
    cached = cache_for(case)
    cached.begin_study('SYNTHETIC-a')
    next(case[2].parameters()).requires_grad_(True)
    with torch.inference_mode(), pytest.raises(ContractError):
        cached(pixel_values=case[3])


@pytest.mark.parametrize('drift', ['value', 'replacement', 'dtype'])
def test_cached_weights_cannot_change_between_targets(case, drift):
    cached = cache_for(case)
    cached.begin_study('SYNTHETIC-a')
    with torch.inference_mode():
        cached(pixel_values=case[3])
    if drift == 'value':
        with torch.no_grad():
            next(case[2].parameters()).add_(0.01)
    elif drift == 'replacement':
        layer = case[2].vision_model.embeddings.patch_embedding
        layer.weight = torch.nn.Parameter(layer.weight.detach().clone(), requires_grad=False)
    else:
        case[2].to(torch.float16)
    with torch.inference_mode(), pytest.raises(ContractError):
        cached(pixel_values=case[3])
    assert cached.summary()['underlying_vision_calls'] == 1


def synthetic_teacher(case, *, fail_label=None):
    model, owner, _tower, pixels, receipt = case
    calls = []

    class Teacher:
        def __init__(self):
            self.model = model
            self.torch = torch
            self.placement = 'single_gpu'
            self.vision_runtime = receipt
            self.images = {}
            self.preprocessing_seconds = 0.0
            self.inference_seconds = 0.0
            self.config = {'vision_microbatch_images': 1}

        def score_logits(self, uid, label):
            calls.append((uid, label))
            if label == fail_label:
                raise ContractError('SYNTHETIC target forward failure')
            hidden = owner.vision_tower(pixel_values=pixels).last_hidden_state
            result = owner.multi_modal_projector(hidden)
            return {'yes_probability': float(torch.sigmoid(result.mean()).item())}

    return Teacher(), calls


def cpu_memory(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 0)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda _index: 0)
    monkeypatch.setattr(torch.cuda, 'max_memory_reserved', lambda _index: 0)


def test_composition_delegates_frozen_predict_without_changing_twelve_target_order(case, monkeypatch, tmp_path):
    cpu_memory(monkeypatch)
    teacher, calls = synthetic_teacher(case)
    original_predict = ImageTeacher.predict
    dispatch = []

    def spy_predict(self, test, output_dir):
        dispatch.append('frozen_predict')
        return original_predict(self, test, output_dir)

    monkeypatch.setattr(ImageTeacher, 'predict', spy_predict)
    original_parameters = [id(value) for value in teacher.model.parameters()]
    projector = case[1].multi_modal_projector
    cached = install_cached_teacher(teacher)
    result, timings = cached.predict(pd.DataFrame({'StudyInstanceUID': ['SYNTHETIC-a','SYNTHETIC-b']}), tmp_path)
    assert calls == [(uid, target) for uid in ['SYNTHETIC-a','SYNTHETIC-b'] for target in LABELS]
    assert dispatch == ['frozen_predict']
    assert result.StudyInstanceUID.tolist() == ['SYNTHETIC-a','SYNTHETIC-b']
    assert len(timings) == 2
    assert [id(value) for value in teacher.model.parameters()] == original_parameters
    assert case[1].multi_modal_projector is projector
    summary = case[1].vision_tower.summary()
    assert summary['misses'] == 2
    assert summary['hits'] == 22
    assert summary['underlying_vision_calls'] == 2
    # predict must clear its final study, so asking for the same study again
    # computes once more rather than carrying an earlier run's cache forward.
    case[1].vision_tower.begin_study('SYNTHETIC-b')
    with torch.inference_mode():
        case[1].vision_tower(pixel_values=case[3])
    assert case[1].vision_tower.summary()['misses'] == 3


def test_composition_failure_clears_cache_and_does_not_retry(case, monkeypatch, tmp_path):
    cpu_memory(monkeypatch)
    teacher, calls = synthetic_teacher(case, fail_label=LABELS[2])
    cached = install_cached_teacher(teacher)
    with pytest.raises(ContractError, match='SYNTHETIC target forward failure'):
        cached.predict(pd.DataFrame({'StudyInstanceUID': ['SYNTHETIC-a']}), tmp_path)
    assert calls == [('SYNTHETIC-a', target) for target in LABELS[:3]]
    assert not (tmp_path / 'submission.csv').exists()
    assert not (tmp_path / 'submission.csv.partial').exists()
    cache = case[1].vision_tower
    assert cache.summary()['misses'] == 1
    cache.begin_study('SYNTHETIC-a')
    with torch.inference_mode():
        cache(pixel_values=case[3])
    assert cache.summary()['misses'] == 2


@pytest.mark.parametrize('diagnostic_fails', [False, True])
def test_diagnostic_bypasses_cache_and_cannot_prime_timed_predict(case, diagnostic_fails):
    teacher, _calls = synthetic_teacher(case)
    original_tower = case[1].vision_tower
    original_calls = []
    hook = original_tower.register_forward_pre_hook(lambda _module, _args: original_calls.append('forward'))

    def diagnostic(uid):
        teacher.score_logits(uid, LABELS[0])
        if diagnostic_fails:
            raise ContractError('SYNTHETIC diagnostic failure')
        teacher.score_logits(uid, LABELS[0])
        return {'kind': 'SYNTHETIC'}

    teacher.diagnostic = diagnostic
    cached = install_cached_teacher(teacher)
    cache = case[1].vision_tower
    with torch.inference_mode():
        if diagnostic_fails:
            with pytest.raises(ContractError, match='SYNTHETIC diagnostic failure'):
                cached.diagnostic('SYNTHETIC-a')
        else:
            assert cached.diagnostic('SYNTHETIC-a') == {'kind': 'SYNTHETIC'}
    assert len(original_calls) == (1 if diagnostic_fails else 2)
    assert cache.summary()['hits'] == 0
    assert cache.summary()['misses'] == 0
    assert cache.summary()['cache_released'] is True
    assert cache.summary()['active_study'] is None
    with torch.inference_mode():
        cached.score_logits('SYNTHETIC-a', LABELS[0])
        cached.score_logits('SYNTHETIC-a', LABELS[1])
    hook.remove()
    assert cache.summary()['misses'] == 1
    assert cache.summary()['hits'] == 1
    assert len(original_calls) == (2 if diagnostic_fails else 3)


@pytest.mark.parametrize('change', ['model_type', 'placement', 'microbatch'])
def test_install_rejects_recipe_drift_without_replacing_existing_tower(case, change):
    teacher, _calls = synthetic_teacher(case)
    original = case[1].vision_tower
    if change == 'model_type':
        teacher.model.get_base_model().config.model_type = 'other'
    elif change == 'placement':
        teacher.placement = 'two_gpu'
    else:
        teacher.vision_runtime['vision_microbatch_images'] = 2
    with pytest.raises(ContractError):
        install_cached_teacher(teacher)
    assert case[1].vision_tower is original


def test_install_cannot_wrap_cache_twice(case):
    teacher, _calls = synthetic_teacher(case)
    install_cached_teacher(teacher)
    original = case[1].vision_tower
    with pytest.raises(ContractError):
        install_cached_teacher(teacher)
    assert case[1].vision_tower is original
