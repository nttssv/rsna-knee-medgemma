"""Reuse only byte-equal, same-study vision inputs; frozen decoder still runs 12 times."""
from __future__ import annotations

from contextlib import contextmanager
import time

import torch
from transformers.modeling_outputs import BaseModelOutputWithPooling

from inference_core import ContractError


def _precision_context():
    return tuple((device, torch.is_autocast_enabled(device), str(torch.get_autocast_dtype(device)))
                 for device in ('cpu', 'cuda'))


def _weight_signature(module):
    signature = []
    for name, value in list(module.named_parameters()) + list(module.named_buffers()):
        if value.requires_grad:
            raise ContractError('Vision cache requires frozen parameters and buffers')
        try:
            version = value._version
        except RuntimeError as error:
            raise ContractError('Cannot verify vision weight mutation version') from error
        signature.append((name, id(value), value.data_ptr(), version, tuple(value.shape),
                          str(value.dtype), str(value.device)))
    return tuple(signature)


class StudyVisionCache(torch.nn.Module):
    """One-slot inference cache around the existing ordered vision microbatch.

    There is no UID-only hit: every processed image tensor and execution context
    must match. Drift within a study raises instead of silently recomputing.
    Returned buffers are independent of the stored cache, so the projector cannot
    mutate later hits. Cache never crosses study boundaries or error cleanup.
    """
    def __init__(self, wrapped):
        super().__init__()
        if any(module.training for module in wrapped.modules()):
            raise ContractError('Vision cache is inference-only and requires eval mode')
        self.wrapped = wrapped
        self.config = wrapped.config
        self.eval()
        self._validate_eval()
        self._weights = _weight_signature(wrapped)
        self._uid = None
        self._pixels = None
        self._hidden = None
        self._call_context = None
        self._bypass = False
        self.hits = self.misses = self.underlying_vision_calls = 0
        self._study_hits = self._study_misses = 0
        self.studies = []
        self._vision_events = []
        self.vision_dispatch_wall_seconds = 0.0

    def _validate_eval(self):
        if any(module.training for module in self.modules()):
            raise ContractError('Vision cache is inference-only and requires eval mode')

    def begin_study(self, uid):
        if not isinstance(uid, str) or not uid:
            raise ContractError('Vision cache requires an explicit nonempty study identity')
        if self._uid != uid:
            self.clear()
            self._uid = uid

    def clear(self):
        if self._uid is not None:
            self.studies.append({'study_id': self._uid, 'hits': self._study_hits,
                                 'misses': self._study_misses})
        self._uid = self._pixels = self._hidden = self._call_context = None
        self._study_hits = self._study_misses = 0

    @contextmanager
    def bypass(self):
        self.clear()
        self._bypass = True
        try:
            yield
        finally:
            self._bypass = False
            self.clear()

    def _context(self, pixel_values, kwargs):
        if pixel_values.ndim != 4 or pixel_values.shape[0] < 1:
            raise ContractError('Vision cache expects the full ordered image tensor')
        if any(type(value) not in (type(None), bool, int, float, str) for value in kwargs.values()):
            raise ContractError('Unsupported vision cache keyword argument')
        if any(kwargs.get(key) for key in ('output_attentions', 'output_hidden_states')):
            raise ContractError('Vision cache does not collect optional output histories')
        return (tuple(pixel_values.shape), str(pixel_values.dtype), str(pixel_values.device),
                tuple(pixel_values.stride()), str(pixel_values.layout),
                tuple(sorted(kwargs.items())), _precision_context())

    def forward(self, pixel_values, **kwargs):
        try:
            self._validate_eval()
            if torch.is_grad_enabled() or not torch.is_inference_mode_enabled():
                raise ContractError('Vision cache is inference-only; require torch.inference_mode')
            if _weight_signature(self.wrapped) != self._weights:
                raise ContractError('Vision weights changed after cache installation')
            if self._bypass:
                return self.wrapped(pixel_values=pixel_values, **kwargs)
            if self._uid is None:
                raise ContractError('No active study bound to vision cache')
            context = self._context(pixel_values, kwargs)
            if self._hidden is not None:
                if (context != self._call_context
                        or not torch.equal(pixel_values.contiguous().view(torch.uint8),
                                           self._pixels.contiguous().view(torch.uint8))):
                    raise ContractError('Same-study vision input/context drift; refusing stale cache')
                self.hits += 1
                self._study_hits += 1
            else:
                if not bool(torch.isfinite(pixel_values).all()):
                    raise ContractError('Nonfinite input to vision cache')
                start = time.monotonic()
                events = None
                if pixel_values.is_cuda:
                    events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                    events[0].record()
                self.underlying_vision_calls += 1
                output = self.wrapped(pixel_values=pixel_values, **kwargs)
                if events:
                    events[1].record()
                    self._vision_events.append(events)
                self.vision_dispatch_wall_seconds += time.monotonic() - start
                if (not isinstance(output, BaseModelOutputWithPooling)
                        or output.last_hidden_state is None
                        or any(getattr(output, name) is not None for name in
                               ('pooler_output', 'hidden_states', 'attentions'))):
                    raise ContractError('Unexpected original vision output contract')
                if output.last_hidden_state.shape[0] != pixel_values.shape[0]:
                    raise ContractError('Vision output image count changed')
                if not bool(torch.isfinite(output.last_hidden_state).all()):
                    raise ContractError('Nonfinite original vision output')
                self._pixels = pixel_values.detach().clone()
                self._hidden = output.last_hidden_state.detach().clone()
                self._call_context = context
                self.misses += 1
                self._study_misses += 1
            return BaseModelOutputWithPooling(last_hidden_state=self._hidden.clone())
        except BaseException:
            self.clear()
            raise

    def summary(self, collect_timing=True):
        gpu_ms = None
        if self._vision_events and collect_timing:
            for _, end in self._vision_events:
                end.synchronize()
            gpu_ms = sum(start.elapsed_time(end) for start, end in self._vision_events)
        studies = list(self.studies)
        if self._uid is not None:
            studies.append({'study_id': self._uid, 'hits': self._study_hits,
                            'misses': self._study_misses})
        return {'policy': 'same_study_exact_pixel_vision_cache_v1',
                'hits': self.hits, 'misses': self.misses,
                'underlying_vision_calls': self.underlying_vision_calls,
                'vision_gpu_milliseconds': gpu_ms,
                'vision_dispatch_wall_seconds': self.vision_dispatch_wall_seconds,
                'active_study': self._uid, 'cache_released': self._hidden is None,
                'studies': studies, 'decoder_forwards_per_study': 12,
                'projector_unchanged': True, 'arithmetic_unchanged': True}


class CachedImageTeacher:
    """Composition adapter: no historical method or global is patched."""
    def __init__(self, teacher, vision_cache):
        self.teacher = teacher
        self.vision_cache = vision_cache
        self.score_records = []

    def __getattr__(self, name):
        return getattr(self.teacher, name)

    def score_logits(self, uid, label):
        self.vision_cache.begin_study(str(uid))
        try:
            result = self.teacher.score_logits(uid, label)
            self.score_records.append({'study_id': str(uid), 'label': label, **result})
            return result
        except BaseException:
            self.vision_cache.clear()
            raise

    def diagnostic(self, uid):
        # Baseline diagnostics must execute the original tower, and must not
        # prime the first measured study. Production installs after preparation.
        with self.vision_cache.bypass():
            return self.teacher.diagnostic(uid)

    def predict(self, test, output_dir):
        from submission_runtime import ImageTeacher
        self.vision_cache.clear()
        try:
            return ImageTeacher.predict(self, test, output_dir)
        finally:
            self.vision_cache.clear()


def install_cached_teacher(teacher):
    base = teacher.model.get_base_model()
    if base.config.model_type != 'gemma3' or teacher.placement != 'single_gpu':
        raise ContractError('Study vision cache requires the frozen single-GPU Gemma3 path')
    if teacher.vision_runtime.get('vision_microbatch_images') != 1:
        raise ContractError('Existing one-image vision microbatch must remain unchanged')
    original = base.model.vision_tower
    if type(original).__name__ != 'OrderedVisionMicrobatch':
        raise ContractError('Expected the original ordered vision microbatch wrapper')
    cache = StudyVisionCache(original)
    base.model.vision_tower = cache
    return CachedImageTeacher(teacher, cache)
