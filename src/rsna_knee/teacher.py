"""The validated QLoRA recipe, with provider-independent paths and checkpoints."""

from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
import json
import os
import random
import time
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score
from .constants import LABELS, PLANES, DEFINITIONS
from . import imaging, checkpoints


def auc(predictions, truth, tag):
    truth = truth.set_index("StudyInstanceUID").loc[predictions.StudyInstanceUID]
    rows = []
    for label in LABELS:
        y = truth[label].to_numpy(float)
        p = predictions[label].to_numpy(float)
        known = np.isfinite(y)
        value = (
            roc_auc_score(y[known], p[known])
            if len(np.unique(y[known])) == 2
            else np.nan
        )
        rows.append(dict(model=tag, label=label, n=int(known.sum()), auc=value))
    return pd.DataFrame(rows)


def log(out, step, values):
    values = {k: float(v) for k, v in values.items() if np.isfinite(v)}
    with (out / "metrics.jsonl").open("a") as f:
        f.write(json.dumps(dict(step=step, values=values), allow_nan=False) + "\n")


class Teacher:
    def __init__(self, cfg, series, adapter=None, trainable=True):
        import torch
        from transformers import (
            AutoProcessor,
            AutoModelForImageTextToText,
            BitsAndBytesConfig,
        )
        from peft import (
            LoraConfig,
            PeftModel,
            get_peft_model,
            prepare_model_for_kbit_training,
        )

        if not torch.cuda.is_available():
            raise RuntimeError("This QLoRA recipe requires a CUDA GPU")
        if torch.cuda.get_device_capability(0)[0] < 8:
            raise RuntimeError("Native BF16 GPU required")
        if (
            torch.cuda.get_device_properties(0).total_memory / 2**30
            < cfg.minimum_gpu_gib
        ):
            raise RuntimeError("GPU is below the configured memory guard")
        self.cfg = cfg
        self.series = series
        self.images = {}
        self.processor = AutoProcessor.from_pretrained(
            cfg.model, revision=cfg.revision, token=os.environ.get("HF_TOKEN")
        )
        self.processor.tokenizer.padding_side = "right"
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForImageTextToText.from_pretrained(
            cfg.model,
            revision=cfg.revision,
            token=os.environ.get("HF_TOKEN"),
            quantization_config=quant,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            attn_implementation="eager",
        )
        # Match the original pilot's non-quantized parameter preparation at inference too.
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=trainable,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        if adapter:
            model = PeftModel.from_pretrained(
                model, str(adapter), is_trainable=trainable
            )
        elif trainable:
            targets = [
                n
                for n, _ in model.named_modules()
                if "language_model" in n
                and n.endswith((".q_proj", ".k_proj", ".v_proj", ".o_proj"))
            ]
            if not targets:
                raise RuntimeError("No expected language attention LoRA modules")
            model = get_peft_model(
                model,
                LoraConfig(
                    r=cfg.lora_rank,
                    lora_alpha=2 * cfg.lora_rank,
                    lora_dropout=0.05,
                    bias="none",
                    target_modules=targets,
                    task_type="CAUSAL_LM",
                ),
            )
        model.config.use_cache = False
        self.model = model
        self.answer_ids = []
        for answer in ["No", "Yes"]:
            ids = self.processor.tokenizer.encode(answer, add_special_tokens=False)
            if len(ids) != 1:
                raise RuntimeError("Answer tokenization changed")
            self.answer_ids.append(ids[0])

    def messages(self, uid, label, answer=None):
        if uid not in self.images:
            arr, mask, meta = imaging.study_images(
                uid, self.series, size=self.cfg.image_size
            )
            self.images[uid] = [
                (Image.fromarray(arr[i]).convert("RGB"), meta[i])
                for i in np.flatnonzero(mask)
            ]
        content = [
            dict(
                type="text",
                text="These are ordered sampled slices from one knee MRI study. Some anatomy may not be visible.",
            )
        ]
        for im, m in self.images[uid]:
            content.extend(
                [
                    dict(
                        type="text",
                        text=f"{PLANES[int(np.argmax(m[:3]))]}, slice position {m[3]:.2f}, fluid sensitive {int(m[4])}, fat suppression {int(m[5])}.",
                    ),
                    dict(type="image", image=im),
                ]
            )
        content.append(
            dict(
                type="text",
                text=f"Is there {DEFINITIONS[label]}? Answer with only Yes or No.",
            )
        )
        result = [dict(role="user", content=content)]
        if answer is not None:
            result.append(
                dict(role="assistant", content=[dict(type="text", text=answer)])
            )
        return result

    def encode(self, uid, label, answer=None):
        import torch

        p = self.processor.apply_chat_template(
            self.messages(uid, label),
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        if answer is None:
            return p
        batch = self.processor.apply_chat_template(
            self.messages(uid, label, answer),
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        prefix = p["input_ids"].shape[1]
        if not torch.equal(p["input_ids"], batch["input_ids"][:, :prefix]):
            raise RuntimeError("Chat template boundary changed")
        batch["labels"] = batch["input_ids"].clone()
        batch["labels"][:, :prefix] = -100
        active = batch["labels"][batch["labels"] != -100]
        if not len(active) or active[0].item() != self.answer_ids[int(answer == "Yes")]:
            raise RuntimeError("Answer boundary changed")
        return batch

    @staticmethod
    def on_gpu(batch):
        import torch

        return {
            k: v.to("cuda:0", dtype=torch.bfloat16)
            if v.is_floating_point()
            else v.to("cuda:0")
            for k, v in batch.items()
        }

    def predict(self, frame, path):
        import torch

        self.model.eval()
        rows = []
        with torch.inference_mode():
            for _, row in frame.iterrows():
                record = {
                    "StudyInstanceUID": row.StudyInstanceUID,
                    "split": row["split"],
                }
                for label in LABELS:
                    batch = self.on_gpu(self.encode(row.StudyInstanceUID, label))
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits = (
                            self.model(**batch, use_cache=False, logits_to_keep=1)
                            .logits[0, -1]
                            .float()
                        )
                    pair = logits[self.answer_ids]
                    if not torch.isfinite(pair).all():
                        raise RuntimeError("Non-finite prediction")
                    record[label] = float(pair.softmax(0)[1].cpu())
                    del batch, logits, pair
                rows.append(record)
                # Save recoverable progress without treating an incomplete file as complete.
                pd.DataFrame(rows).to_csv(Path(str(path) + ".partial"), index=False)
                print(f"Scored {len(rows)}/{len(frame)} studies", flush=True)
        Path(str(path) + ".partial").replace(path)
        return pd.DataFrame(rows)


def identity(cfg, out):
    code = b"".join(
        Path(p).read_bytes()
        for p in [__file__, imaging.__file__, Path(__file__).with_name("constants.py")]
    )
    return checkpoints.fingerprint(
        cfg.identity(),
        (out / "development_split.csv").read_bytes(),
        (out / "input_fingerprints.csv").read_bytes(),
        (out / "ground_truth.csv").read_bytes(),
        code,
    )


def train(cfg, pilot, series, out, resume=None, init_adapter=None):
    import torch

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    resume = checkpoints.resolve(resume) if resume else None
    teacher = Teacher(
        cfg, series, adapter=resume / "adapter" if resume else init_adapter
    )
    validation = pilot.loc[pilot["split"] == "validation"]
    training = pilot.loc[pilot["split"] == "train"]
    run_identity = identity(cfg, out)
    (out / "training_config.json").write_text(
        json.dumps(
            {
                **asdict(cfg),
                "identity": run_identity,
                "precision": "nf4_with_bf16_autocast",
                "init_adapter": str(init_adapter) if init_adapter else None,
                "resumed_from": str(resume) if resume else None,
                "versions": {
                    p: version(p)
                    for p in [
                        "torch",
                        "transformers",
                        "peft",
                        "bitsandbytes",
                        "pydicom",
                    ]
                },
            },
            indent=2,
        )
    )
    optimizer = torch.optim.AdamW(
        [p for p in teacher.model.parameters() if p.requires_grad], lr=cfg.learning_rate
    )
    sampler = random.Random(cfg.seed)
    start = 0
    history = []
    if resume:
        start, history = checkpoints.restore(resume, optimizer, sampler, run_identity)
        if start > cfg.max_steps:
            raise ValueError("max_steps is less than checkpoint step")
        # Skip baseline scoring here: it would overwrite the meaning of the original baseline.
    else:
        before = teacher.predict(validation, out / "validation_before.csv")
        log(out, 0, {"validation/mean_auc": auc(before, pilot, "initial").auc.mean()})
    examples = [
        (r.StudyInstanceUID, label, "Yes" if r[label] == 1 else "No")
        for _, r in training.iterrows()
        for label in LABELS
        if pd.notna(r[label])
    ]
    teacher.model.train()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    for step in range(start, cfg.max_steps):
        losses = []
        for _ in range(cfg.accumulation):
            uid, label, answer = sampler.choice(examples)
            batch = teacher.on_gpu(teacher.encode(uid, label, answer))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = teacher.model(**batch, use_cache=False).loss
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            (loss / cfg.accumulation).backward()
            losses.append(float(loss.detach().cpu()))
            del batch, loss
        norm = torch.nn.utils.clip_grad_norm_(
            [p for p in teacher.model.parameters() if p.requires_grad], 1.0
        )
        if not torch.isfinite(norm):
            raise RuntimeError("Non-finite gradients")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        history.append(dict(step=step + 1, loss=float(np.mean(losses))))
        pd.DataFrame(history).to_csv(out / "training_history.csv", index=False)
        log(
            out,
            step + 1,
            {
                "train/loss": history[-1]["loss"],
                "train/gradient_norm": float(norm.detach().cpu()),
                "train/learning_rate": cfg.learning_rate,
                "gpu/allocated_gib": torch.cuda.memory_allocated() / 2**30,
            },
        )
        print(
            f"Step {step + 1}/{cfg.max_steps}: loss={history[-1]['loss']:.5f}",
            flush=True,
        )
        if (step + 1) % cfg.checkpoint_every == 0 or step + 1 == cfg.max_steps:
            checkpoints.save(
                out / "checkpoints",
                step + 1,
                teacher.model,
                optimizer,
                sampler,
                history,
                run_identity,
            )
    seconds = time.monotonic() - started
    peak = torch.cuda.max_memory_allocated() / 2**30
    teacher.model.save_pretrained(out / "teacher_adapter", safe_serialization=True)
    teacher.processor.save_pretrained(out / "teacher_adapter")
    after = teacher.predict(validation, out / "validation_after.csv")
    frames = [auc(after, pilot, "adapted")]
    if (out / "validation_before.csv").exists():
        frames.insert(
            0, auc(pd.read_csv(out / "validation_before.csv"), pilot, "initial")
        )
    metrics = pd.concat(frames, ignore_index=True)
    metrics.to_csv(out / "validation_metrics.csv", index=False)
    log(out, cfg.max_steps, {"validation/mean_auc": frames[-1].auc.mean()})
    if cfg.export_train_targets:
        teacher.predict(training, out / "teacher_targets_train_in_sample.csv")
    (out / "pilot_status.json").write_text(
        json.dumps(
            dict(
                status="teacher_pilot_complete",
                training_seconds_this_session=seconds,
                peak_training_allocated_gib=peak,
                optimizer_steps=cfg.max_steps,
                training_studies=len(training),
                validation_studies=len(validation),
                teacher_scores="uncalibrated_yes_no_ranking_scores",
            ),
            indent=2,
        )
    )
