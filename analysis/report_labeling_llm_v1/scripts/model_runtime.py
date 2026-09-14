"""Optional Transformers backend. Imported/loaded only after explicit execution flags."""
import hashlib
import json
import time
from benchmark_core import config, text_sha


class Tokenizer:
    def __init__(self, name, cache, allow_download=False):
        from transformers import AutoProcessor, AutoTokenizer
        self.name, self.cfg = name, config(name)
        kwargs = dict(revision=self.cfg['revision'], cache_dir=str(cache),
                      trust_remote_code=False, local_files_only=not allow_download)
        self.processor = (AutoProcessor if name == 'medgemma' else AutoTokenizer).from_pretrained(self.cfg['model_id'], **kwargs)
        self.tokenizer = self.processor.tokenizer if name == 'medgemma' else self.processor

    def render(self, prompt):
        content = [{'type': 'text', 'text': prompt}] if self.name == 'medgemma' else prompt
        kwargs = {'enable_thinking': False} if self.name == 'qwen' else {}
        return self.processor.apply_chat_template([{'role': 'user', 'content': content}],
                   tokenize=False, add_generation_prompt=True, **kwargs)

    def encode(self, rendered):
        return self.tokenizer(rendered, add_special_tokens=False, return_tensors='pt', truncation=False)

    def metadata(self):
        return {'model_id': self.cfg['model_id'], 'tokenizer_revision': self.cfg['revision'],
                'chat_template': self.processor.chat_template,
                'eos_token_id': self.tokenizer.eos_token_id, 'pad_token_id': self.tokenizer.pad_token_id,
                'system_role': 'none; shared instructions in single user message',
                'enable_thinking': False, 'max_new_tokens': self.cfg['max_new_tokens']}


class HFGenerator:
    def __init__(self, encoder, cache, allow_download=False):
        import torch
        import transformers
        from transformers import GenerationConfig, set_seed
        cfg = encoder.cfg
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('This execution recipe requires a CUDA GPU with native BF16 support')
        if torch.cuda.mem_get_info()[0] / 2**30 < cfg['minimum_free_gpu_gib']:
            raise RuntimeError('Insufficient free GPU memory for the proposed BF16 configuration')
        self.encoder, self.torch, self.cfg = encoder, torch, cfg
        set_seed(cfg['seed'])
        cls = getattr(transformers, cfg['architecture'])
        self.model = cls.from_pretrained(cfg['model_id'], revision=cfg['revision'],
            dtype=torch.bfloat16, device_map={'': 'cuda:0'}, trust_remote_code=False,
            attn_implementation=cfg['attn_implementation'], cache_dir=str(cache), local_files_only=not allow_download)
        self.model.eval()
        self.gen = GenerationConfig(do_sample=False, num_beams=1, use_cache=True,
            max_new_tokens=cfg['max_new_tokens'], max_time=cfg['max_time_seconds'],
            eos_token_id=self.model.generation_config.eos_token_id,
            pad_token_id=self.model.generation_config.pad_token_id or encoder.tokenizer.eos_token_id)
        self.context_limit = getattr(getattr(self.model.config, 'text_config', self.model.config), 'max_position_embeddings', cfg['max_input_tokens'] + cfg['max_new_tokens'])

    def __call__(self, prompt):
        rendered = self.encoder.render(prompt)
        inputs = self.encoder.encode(rendered)
        length = inputs['input_ids'].shape[-1]
        base = {'text': '', 'rendered_prompt': rendered, 'rendered_prompt_sha256': text_sha(rendered),
                'input_tokens': length, 'output_tokens': 0, 'runtime_seconds': 0.0,
                'generation_config': self.gen.to_dict()}
        if length > self.cfg['max_input_tokens'] or length + self.cfg['max_new_tokens'] > self.context_limit:
            return dict(base, status='context_overflow')
        torch = self.torch
        torch.cuda.synchronize(); start = time.perf_counter()
        try:
            with torch.inference_mode():
                output = self.model.generate(**inputs.to('cuda:0'), generation_config=self.gen)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            generated = output[0, length:].tolist()
            eos = self.gen.eos_token_id
            eos = eos if isinstance(eos, list) else [eos]
            completed = bool(generated) and generated[-1] in eos
            status = 'completed' if completed else ('timeout' if elapsed >= self.cfg['max_time_seconds'] else 'generation_truncated')
            return dict(base, text=self.encoder.tokenizer.decode(generated, skip_special_tokens=True),
                        status=status, runtime_seconds=elapsed, output_tokens=len(generated))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return dict(base, status='oom', runtime_seconds=time.perf_counter() - start)
        except RuntimeError as error:
            # Store only the error class, avoiding paths/credentials from third-party exceptions.
            return dict(base, status='runtime_error', error_type=type(error).__name__, runtime_seconds=time.perf_counter() - start)
