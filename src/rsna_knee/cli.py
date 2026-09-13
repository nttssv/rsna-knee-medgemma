from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
from .config import load


def main():
    parser = argparse.ArgumentParser(description="Portable MedGemma knee MRI pilot")
    parser.add_argument("--config", type=Path, default=Path("configs/pilot.toml"))
    parser.add_argument(
        "--state-dir", type=Path, help="Override RSNA_STATE_DIR; defaults to ./state"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser(
        "doctor", help="Inspect local environment without downloading or training"
    )
    for name in ["audit", "train", "evaluate", "export-cases"]:
        p = sub.add_parser(name)
        p.add_argument("--run-dir", type=Path)
        p.add_argument("--data-dir", type=Path)
        if name in ["audit", "train", "evaluate"]:
            p.add_argument("--split-file", type=Path)
        if name == "train":
            g = p.add_mutually_exclusive_group()
            g.add_argument("--resume", type=Path)
            g.add_argument("--init-adapter", type=Path)
            p.add_argument("--max-steps", type=int)
        if name == "evaluate":
            p.add_argument("--adapter", type=Path, required=True)
    review = sub.add_parser("review")
    review.add_argument("--run-dir", type=Path, required=True)
    review.add_argument("--port", type=int, default=7862)
    args = parser.parse_args()
    cfg, paths, state = load(args.config, args.state_dir)
    if getattr(args, "max_steps", None) is not None:
        cfg = replace(cfg, max_steps=args.max_steps)
    if args.command == "review":
        from .review import serve

        serve(args.run_dir, args.port)
        return
    if args.command == "doctor":
        result = {
            "state_dir": str(state),
            "data_present": (paths["data"] / "train.csv").exists(),
            "model": cfg.model,
            "revision": cfg.revision,
        }
        try:
            import torch

            result.update(
                torch=torch.__version__, cuda_available=torch.cuda.is_available()
            )
            if torch.cuda.is_available():
                p = torch.cuda.get_device_properties(0)
                result.update(
                    gpu=p.name,
                    gpu_gib=p.total_memory / 2**30,
                    bf16=torch.cuda.get_device_capability(0)[0] >= 8,
                )
        except ImportError:
            result["torch"] = "not installed (CPU audit still available)"
        print(json.dumps(result, indent=2))
        return
    data = (args.data_dir or paths["data"]).resolve()
    if args.command == "export-cases":
        if not args.run_dir:
            parser.error("--run-dir is required for export-cases")
        from .review import export

        export(cfg, data, args.run_dir.resolve())
        return
    os.umask(0o077)
    out = (
        args.run_dir
        or paths["runs"] / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{args.command}"
    ).resolve()
    # A resume writes into a new run so existing results are never overwritten.
    out.mkdir(parents=True, exist_ok=False)
    paths["cache"].mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(paths["cache"] / "huggingface"))
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    resume = None
    if args.command == "train" and args.resume:
        from .checkpoints import resolve

        resume = resolve(args.resume).resolve()
        origin = resume.parent.parent
        if not args.split_file:
            args.split_file = origin / "development_split.csv"
        if (origin / "validation_before.csv").is_file():
            shutil.copy2(
                origin / "validation_before.csv", out / "validation_before.csv"
            )
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = None
    (out / "provenance.json").write_text(
        json.dumps(
            {
                "git_commit": commit,
                "config_file": args.config.name,
                "command": args.command,
            },
            indent=2,
        )
    )
    try:
        from .audit import run

        pilot, series = run(cfg, data, out, args.split_file)
        if args.command == "train":
            from .teacher import train

            train(
                cfg, pilot, series, out, resume=resume, init_adapter=args.init_adapter
            )
        elif args.command == "evaluate":
            from .teacher import Teacher, auc

            teacher = Teacher(cfg, series, adapter=args.adapter, trainable=False)
            result = teacher.predict(
                pilot.loc[pilot["split"] == "validation"], out / "validation_after.csv"
            )
            auc(result, pilot, "adapted").to_csv(
                out / "validation_metrics.csv", index=False
            )
        (out / "runner_status.json").write_text(
            json.dumps({"status": "complete", "mode": args.command})
        )
    except BaseException as error:
        (out / "runner_status.json").write_text(
            json.dumps(
                {
                    "status": "failed",
                    "mode": args.command,
                    "error_type": type(error).__name__,
                }
            )
        )
        raise
    print(f"Completed: {out}", flush=True)


if __name__ == "__main__":
    main()
