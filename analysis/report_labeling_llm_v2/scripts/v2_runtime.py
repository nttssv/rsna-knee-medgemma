"""Lazy, offline-only Transformers adapter. No model imports at module import time."""
from dataclasses import dataclass
from importlib.metadata import version
import hashlib
import os
from pathlib import Path
import time

from core import config


def text_sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def eos_ids(value):
    values = value if isinstance(value, list) else [value]
    if not values or any(type(x) is not int or x < 0 for x in values):
        raise ValueError('Missing or invalid EOS IDs')
    return values


def resolve_stops(model_eos, model_pad, tokenizer_pad, tokenizer_eos):
    """Preserve a valid zero PAD ID; never use truthiness for token IDs."""
    eos = eos_ids(model_eos)
    pad = model_pad if model_pad is not None else tokenizer_pad
    if pad is None:
        pad = tokenizer_eos
    if type(pad) is not int or pad < 0:
        raise ValueError('Missing or invalid PAD ID')
    return eos, pad


def generation_status(ids, eos, elapsed, deadline, max_new_tokens):
    # Deadline takes precedence even if EOS arrived after the permitted interval.
    if elapsed >= deadline:
        return 'timeout'
    if not ids or len(ids) > max_new_tokens:
        return 'generation_truncated'
    # An EOS followed by other tokens is anomalous for batch-one generation.
    if ids[-1] in eos and not any(token in eos for token in ids[:-1]):
        return 'completed'
    return 'generation_truncated'


def generation_options(cfg, eos, pad):
    return dict(do_sample=False, num_beams=1, num_return_sequences=1,
                use_cache=True, max_new_tokens=cfg['max_new_tokens'],
                max_time=cfg['max_time_seconds'], eos_token_id=eos, pad_token_id=pad,
                return_dict_in_generate=False)


@dataclass
class Encoded:
    rendered_prompt: str
    inputs: object
    input_ids: list

    @property
    def input_tokens(self):
        return len(self.input_ids)


def messages_for(model, prompt):
    content = [{'type': 'text', 'text': prompt}] if model == 'medgemma' else prompt
    return [{'role': 'user', 'content': content}]


def encode_prompt(processor, tokenizer, model, prompt):
    controls = {'enable_thinking': False} if model == 'qwen' else {}
    messages = messages_for(model, prompt)
    rendered = processor.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True, **controls)
    if model == 'medgemma':
        inputs = processor.apply_chat_template(messages, tokenize=True,
                    add_generation_prompt=True, return_dict=True, return_tensors='pt',
                    truncation=False)
    else:
        inputs = tokenizer(rendered, add_special_tokens=False, return_tensors='pt',
                           truncation=False)
    ids = inputs['input_ids']
    if len(ids.shape) != 2 or ids.shape[0] != 1 or ids.shape[1] == 0:
        raise ValueError('Only one nonempty report input is permitted')
    values = ids[0].tolist()
    # Prove the rendered audit text corresponds to exactly what the model sees.
    audit = tokenizer(rendered, add_special_tokens=False, return_tensors='pt',
                      truncation=False)['input_ids'][0].tolist()
    if values != audit:
        raise ValueError('Rendered input differs from processor tokenization')
    return Encoded(rendered, inputs, values)


def offline_environment(cache, model):
    cache = Path(cache).resolve()
    snapshot = cache / ('models--'+config(model)['model_id'].replace('/','--')) / 'snapshots' / config(model)['revision']
    if not cache.is_dir() or not snapshot.is_dir() or not any(snapshot.iterdir()):
        raise ValueError('Explicit private cache lacks the pinned snapshot; no fallback or download')
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        HF_HOME=str(cache), HF_HUB_CACHE=str(cache), TRANSFORMERS_CACHE=str(cache),
        HF_HUB_DISABLE_IMPLICIT_TOKEN='1')
    return cache


class HFEncoder:
    def __init__(self, model, cache):
        cache = offline_environment(cache, model)
        from transformers import AutoProcessor, AutoTokenizer
        self.model_key, self.cfg = model, config(model)
        expected = config('runtime')['models'][model]
        cls = AutoProcessor if model == 'medgemma' else AutoTokenizer
        self.processor = cls.from_pretrained(self.cfg['model_id'],
            revision=self.cfg['revision'], cache_dir=str(cache), trust_remote_code=False,
            local_files_only=True)
        self.tokenizer = self.processor.tokenizer if model == 'medgemma' else self.processor
        template = self.processor.chat_template
        if not isinstance(template, str) or text_sha(template) != expected['chat_template_sha256']:
            raise ValueError('Chat template differs from reviewed pinned template')
        if (self.tokenizer.eos_token_id != expected['tokenizer_eos_token_id'] or
                self.tokenizer.pad_token_id != expected['tokenizer_pad_token_id']):
            raise ValueError('Tokenizer special tokens differ from reviewed recipe')
        self.metadata = dict(model_id=self.cfg['model_id'], model_revision=self.cfg['revision'],
            tokenizer_revision=self.cfg['revision'], chat_template_sha256=text_sha(template),
            tokenizer_eos_token_id=self.tokenizer.eos_token_id,
            tokenizer_pad_token_id=self.tokenizer.pad_token_id,
            thinking_control='enable_thinking=False' if model == 'qwen' else 'no_supported_switch_claimed')

    def encode(self, prompt):
        return encode_prompt(self.processor, self.tokenizer, self.model_key, prompt)

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=False,
                                     clean_up_tokenization_spaces=False)


def check_versions():
    actual = {name: version(name) for name in config('runtime')['required_versions']}
    if any(actual[name].split('+')[0] != wanted
           for name, wanted in config('runtime')['required_versions'].items()):
        raise RuntimeError('Runtime package versions differ from pinned recipe')
    return actual


class HFBackend:
    def __init__(self, encoder, cache):
        cache = offline_environment(cache, encoder.model_key)
        import torch
        import transformers
        self.versions = check_versions()
        cfg = encoder.cfg
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('CUDA with native BF16 support is required')
        if torch.cuda.mem_get_info()[0] / 2**30 < cfg['minimum_free_gpu_gib']:
            raise RuntimeError('Insufficient free GPU memory')
        self.encoder, self.cfg, self.torch = encoder, cfg, torch
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        transformers.set_seed(cfg['seed'])
        torch.cuda.reset_peak_memory_stats()
        cls = getattr(transformers, cfg['architecture'])
        self.model = cls.from_pretrained(cfg['model_id'], revision=cfg['revision'],
            dtype=torch.bfloat16, device_map={'': 'cuda:0'}, trust_remote_code=False,
            attn_implementation=cfg['attn_implementation'], cache_dir=str(cache),
            local_files_only=True)
        self.model.eval()
        if getattr(self.model.config, '_commit_hash', None) != cfg['revision']:
            raise ValueError('Loaded model revision is unverified')
        self.eos, self.pad = resolve_stops(self.model.generation_config.eos_token_id,
            self.model.generation_config.pad_token_id, encoder.tokenizer.pad_token_id,
            encoder.tokenizer.eos_token_id)
        expected = config('runtime')['models'][encoder.model_key]
        if self.eos != expected['generation_eos_token_ids'] or self.pad != expected['generation_pad_token_id']:
            raise ValueError('Model stopping tokens differ from reviewed recipe')
        self.options = generation_options(cfg, self.eos, self.pad)
        self.gen = transformers.GenerationConfig(**self.options)
        text_config = getattr(self.model.config, 'text_config', self.model.config)
        self.context_limit = getattr(text_config, 'max_position_embeddings', None)
        if type(self.context_limit) is not int:
            raise ValueError('Model context limit is unverified')
        self.metadata = dict(encoder.metadata, package_versions=self.versions,
            effective_generation_config=self.gen.to_dict(), context_limit=self.context_limit,
            gpu_name=torch.cuda.get_device_name(0), cuda_version=torch.version.cuda,
            seed=cfg['seed'], dtype=cfg['dtype'], attention=cfg['attn_implementation'])

    def generate(self, encoded):
        return generate_encoded(self, encoded)

    def peak_gib(self):
        return self.torch.cuda.max_memory_allocated() / 2**30


def generate_encoded(backend, encoded, clock=time.perf_counter):
    """Shared real/fake execution path; fake tests use CPU tensors, no model weights."""
    cfg, torch = backend.cfg, backend.torch
    base = dict(raw_output='', decoded_with_special_tokens='', output_token_ids=[],
                input_tokens=encoded.input_tokens, output_tokens=0, runtime_seconds=0.0)
    if (encoded.input_tokens > cfg['max_input_tokens'] or
            encoded.input_tokens + cfg['max_new_tokens'] > backend.context_limit):
        return dict(base, generation_status='context_overflow')
    torch.cuda.synchronize()
    start = clock()
    try:
        with torch.inference_mode():
            output = backend.model.generate(**encoded.inputs.to('cuda:0'), generation_config=backend.gen)
        torch.cuda.synchronize()
        elapsed = clock() - start
        if len(output.shape) != 2 or output.shape[0] != 1:
            raise RuntimeError('Unexpected generation shape')
        if output[0, :encoded.input_tokens].tolist() != encoded.input_ids:
            raise RuntimeError('Generated sequence does not preserve the input prefix')
        generated = output[0, encoded.input_tokens:].tolist()
        body = generated[:-1] if generated and generated[-1] in backend.eos else generated
        return dict(base, raw_output=backend.encoder.decode(body),
                    decoded_with_special_tokens=backend.encoder.decode(generated),
                    output_token_ids=generated, output_tokens=len(generated), runtime_seconds=elapsed,
                    generation_status=generation_status(generated, backend.eos, elapsed,
                        cfg['max_time_seconds'], cfg['max_new_tokens']))
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return dict(base, generation_status='oom', runtime_seconds=clock()-start)
    except RuntimeError as error:
        return dict(base, generation_status='runtime_error', error_type=type(error).__name__,
                    runtime_seconds=clock()-start)
