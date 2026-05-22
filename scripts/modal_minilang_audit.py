"""Run strict mini-language audits on Modal GPUs.

This keeps the local machine out of the critical path for Qwen3-1.7B reruns.
The local entrypoint writes returned run artifacts into `runs/`.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import modal


app = modal.App("caic-minilang-audit")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "accelerate>=0.30",
        "numpy>=1.26",
        "pandas>=2.2",
        "peft>=0.11",
        "safetensors>=0.4",
        "torch>=2.3",
        "tqdm>=4.66",
        "transformers>=4.51,<5",
    )
    .add_local_dir(
        ".",
        remote_path="/app",
        ignore=[
            ".git",
            ".pytest_cache",
            "__pycache__",
            "runs",
            "*.pyc",
        ],
    )
)


BASE_ARGS = [
    "--model",
    "Qwen/Qwen3-1.7B",
    "--seed",
    "1",
    "--lessons",
    "4",
    "--lesson-examples",
    "8",
    "--trace-probes",
    "4",
    "--eval-mode",
    "exhaustive_modified",
    "--exclude-eval-lesson-overlaps",
    "--exclude-eval-trace-overlaps",
    "--layers",
    "20",
    "--trace-last-tokens",
    "1",
    "--target-mode",
    "output_delta",
    "--teacher-forcing-trace",
    "--token-teacher-forcing-trace",
    "--trace-context",
    "lesson",
    "--batch-size",
    "8",
    "--max-length",
    "1536",
    "--device",
    "cuda",
    "--dtype",
    "bfloat16",
]


def execute_config(label: str, extra_args: list[str]) -> dict:
    import os

    output = f"/tmp/{label}"
    command = [
        "python",
        "scripts/minilang_write.py",
        "--output",
        output,
        *BASE_ARGS,
        *extra_args,
    ]
    env = os.environ.copy()
    env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    env["TRANSFORMERS_VERBOSITY"] = "error"
    completed = subprocess.run(
        command,
        cwd="/app",
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    root = Path(output)
    files: dict[str, str] = {}
    for name in ("config.json", "metrics.jsonl", "updates.jsonl", "lessons.jsonl", "eval_questions.jsonl", "eval_details.jsonl"):
        path = root / name
        if path.exists():
            files[name] = path.read_text(encoding="utf-8")
    return {
        "label": label,
        "returncode": completed.returncode,
        "output": completed.stdout[-20_000:],
        "files": files,
    }


@app.function(gpu="A10G", image=image, timeout=7200)
def run_config(label: str, extra_args: list[str]) -> dict:
    return execute_config(label, extra_args)


@app.function(gpu="A10G", image=image, timeout=7200)
def run_config_json(label: str, extra_args_json: str) -> dict:
    result = execute_config(label, json.loads(extra_args_json))
    summary: dict[str, object] = {
        "label": label,
        "returncode": result["returncode"],
        "output": result["output"],
    }
    metrics_text = result["files"].get("metrics.jsonl", "")
    if metrics_text:
        rows = [json.loads(line) for line in metrics_text.splitlines() if line.strip()]
        summary["metrics"] = rows
    summary["file_names"] = sorted(result["files"])
    return summary


def write_artifact(result: dict) -> None:
    label = result["label"]
    root = Path("runs") / label
    root.mkdir(parents=True, exist_ok=True)
    for name, text in result["files"].items():
        (root / name).write_text(text, encoding="utf-8")
    (root / "modal_result.json").write_text(
        json.dumps(
            {
                "label": label,
                "returncode": result["returncode"],
                "output_tail": result["output"],
                "files": sorted(result["files"]),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


@app.local_entrypoint()
def main() -> None:
    configs = [
        ("minilang_audit_strict_context_exhaustive_seed1_gpu", ["--skip-write", "--no-write-mlp"]),
        ("minilang_audit_strict_l20_attno_mlp_outputdelta_seed1_gpu", ["--write-attention-o"]),
        ("minilang_audit_strict_l20_attno_mlp_outputdelta_shuffle_seed1_gpu", ["--write-attention-o", "--shuffle-targets"]),
        ("minilang_audit_strict_l20_attno_mlp_outputdelta_optneg_seed1_gpu", ["--write-attention-o", "--option-negative-keys", "--max-option-negative-prompts", "48"]),
    ]
    for label, extra_args in configs:
        result = run_config.remote(label, extra_args)
        write_artifact(result)
        print(json.dumps({"label": label, "returncode": result["returncode"], "files": sorted(result["files"])}))
        if result["returncode"] != 0:
            print(result["output"])
            raise SystemExit(result["returncode"])
