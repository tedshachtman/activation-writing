"""Run CAIC experiment commands on Modal GPU compute.

Examples:

    modal run scripts/modal_caic_runner.py --smoke

    modal run scripts/modal_caic_runner.py --command \
      "python scripts/build_minilang_teacher_gated.py --device cuda --dtype float16 --output /modal-runs/gated"

    python scripts/continual_benchmark_grid.py --modal --run --preset qrico_key16

Outputs written under `/modal-runs` persist in the `caic-runs` Modal volume.
Use `modal volume ls caic-runs /` and
`modal volume get caic-runs <volume-path> <local-path>` to download them.
"""

from __future__ import annotations

from pathlib import Path
import textwrap
import subprocess

import modal


LOCAL_REPO = Path(__file__).resolve().parents[1]
REMOTE_REPO = "/root/activation-writing"


def ignore_mount(path: Path) -> bool:
    """Keep local caches, run artifacts, and model blobs out of Modal images."""

    parts = set(path.parts)
    ignored_dirs = {
        ".git",
        ".modal",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "build",
        "caic.egg-info",
        "dist",
        "modal-runs",
        "runs",
    }
    if parts & ignored_dirs:
        return True
    ignored_suffixes = (
        ".bin",
        ".ckpt",
        ".db",
        ".gguf",
        ".log",
        ".npy",
        ".npz",
        ".pt",
        ".pth",
        ".pyc",
        ".safetensors",
        ".sqlite",
        ".tmp",
    )
    return path.name == ".DS_Store" or path.name.endswith(ignored_suffixes)


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install_from_requirements(LOCAL_REPO / "requirements.txt")
    .add_local_dir(LOCAL_REPO, remote_path=REMOTE_REPO, ignore=ignore_mount)
)

run_volume = modal.Volume.from_name("caic-runs", create_if_missing=True)

app = modal.App("caic-experiments", image=image)


@app.function(
    gpu="a10g",
    timeout=60 * 60 * 4,
    volumes={"/modal-runs": run_volume},
)
def run_shell(command: str) -> str:
    import os

    os.chdir(REMOTE_REPO)
    env = os.environ.copy()
    env["PYTHONPATH"] = REMOTE_REPO
    env["HF_HOME"] = "/modal-runs/hf-cache"
    env["TRANSFORMERS_CACHE"] = "/modal-runs/hf-cache"
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    env["TOKENIZERS_PARALLELISM"] = "false"
    process = subprocess.Popen(
        command,
        shell=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        bufsize=1,
    )
    output_parts: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        output_parts.append(line)
    returncode = process.wait()
    output = "".join(output_parts)
    run_volume.commit()
    if returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {returncode}:\n"
            f"{command}\n\n{output}"
        )
    return output


@app.function(gpu="a10g", timeout=10 * 60)
def gpu_smoke() -> str:
    completed = subprocess.run(
        "nvidia-smi && python - <<'PY'\nimport torch\nprint('torch', torch.__version__)\nprint('cuda available', torch.cuda.is_available())\nprint('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')\nPY",
        shell=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stdout)
    return completed.stdout


@app.local_entrypoint()
def main(command: str = "", smoke: bool = False) -> None:
    if smoke:
        print(gpu_smoke.remote())
        return
    if not command:
        command = "python -m pytest -q"
    print(f"Running on Modal a10g: {textwrap.shorten(command, width=160)}")
    run_shell.remote(command)
