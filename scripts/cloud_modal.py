"""Codex Cloud launcher for Modal-backed CAIC jobs.

This script avoids assumptions that often fail in cloud agent containers:

- no configured git remote;
- no ``gh`` CLI;
- no local Modal profile file.

It supports two execution backends:

``direct``
    Run Modal from the current checkout. Requires ``MODAL_TOKEN_ID`` and
    ``MODAL_TOKEN_SECRET`` in the current process environment.

``workflow``
    Dispatch ``.github/workflows/modal-benchmark.yml`` through the GitHub REST
    API. Requires ``GH_TOKEN``, ``GITHUB_TOKEN``, or ``GITHUB_PAT`` with
    workflow dispatch access. GitHub repository Actions secrets provide the
    Modal credentials to the workflow job.

``auto``
    Prefer direct Modal when Modal secrets are present, otherwise use workflow
    dispatch when a GitHub token is present.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib import error, parse, request


DEFAULT_REPO = "tedshachtman/activation-writing"
DEFAULT_WORKFLOW = "modal-benchmark.yml"


def has_modal_env() -> bool:
    return bool(os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET"))


def github_token() -> str:
    return (
        os.environ.get("GH_TOKEN")
        or os.environ.get("GITHUB_TOKEN")
        or os.environ.get("GITHUB_PAT")
        or ""
    )


def run_checked(command: list[str]) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def dispatch_workflow(
    *,
    repo: str,
    workflow: str,
    ref: str,
    mode: str,
    preset: str,
    tag: str,
    command: str,
) -> None:
    token = github_token()
    if not token:
        raise SystemExit(
            "No GitHub token found. Set GH_TOKEN, GITHUB_TOKEN, or GITHUB_PAT, "
            "or use --backend direct with MODAL_TOKEN_ID/MODAL_TOKEN_SECRET."
        )

    url = f"https://api.github.com/repos/{repo}/actions/workflows/{workflow}/dispatches"
    payload = {
        "ref": ref,
        "inputs": {
            "mode": mode,
            "preset": preset,
            "tag": tag,
            "command": command,
        },
    }
    req = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=30) as response:
            if response.status != 204:
                raise SystemExit(f"Unexpected GitHub dispatch status: {response.status}")
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"GitHub workflow dispatch failed: HTTP {exc.code}\n{body}") from exc

    print(f"Dispatched {workflow} on {repo}@{ref} with mode={mode}.", flush=True)
    latest = latest_workflow_run_url(repo=repo, workflow=workflow, token=token)
    if latest:
        print(f"Latest run: {latest}", flush=True)
    else:
        print(f"Runs page: https://github.com/{repo}/actions/workflows/{workflow}", flush=True)


def latest_workflow_run_url(*, repo: str, workflow: str, token: str) -> str:
    time.sleep(3)
    query = parse.urlencode({"event": "workflow_dispatch", "per_page": "1"})
    url = f"https://api.github.com/repos/{repo}/actions/workflows/{workflow}/runs?{query}"
    req = request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception:
        return ""
    runs = data.get("workflow_runs") or []
    if not runs:
        return ""
    return str(runs[0].get("html_url") or "")


def direct_modal(mode: str, preset: str, tag: str, command: str) -> None:
    if not has_modal_env():
        raise SystemExit(
            "MODAL_TOKEN_ID and MODAL_TOKEN_SECRET are required for direct Modal runs. "
            "Use --backend workflow if a GitHub token is available instead."
        )

    if mode == "smoke":
        run_checked([sys.executable, "-m", "modal", "run", "scripts/modal_smoke.py"])
        run_checked([sys.executable, "-m", "modal", "run", "scripts/modal_caic_runner.py", "--smoke"])
        return
    if mode == "preset":
        run_checked(
            [
                sys.executable,
                "scripts/continual_benchmark_grid.py",
                "--modal",
                "--run",
                "--preset",
                preset,
                "--tag",
                tag,
            ]
        )
        return
    if mode == "command":
        run_checked(
            [
                sys.executable,
                "-m",
                "modal",
                "run",
                "scripts/modal_caic_runner.py",
                "--command",
                command,
            ]
        )
        return
    raise SystemExit(f"Unknown mode: {mode}")


def choose_backend(requested: str) -> str:
    if requested != "auto":
        return requested
    if has_modal_env():
        return "direct"
    if github_token():
        return "workflow"
    raise SystemExit(
        "No runnable backend found. For direct Modal, set MODAL_TOKEN_ID and "
        "MODAL_TOKEN_SECRET. For GitHub workflow dispatch, set GH_TOKEN, "
        "GITHUB_TOKEN, or GITHUB_PAT."
    )


def doctor() -> None:
    print(f"cwd: {Path.cwd()}")
    print(f"modal env: {'yes' if has_modal_env() else 'no'}")
    print(f"github token env: {'yes' if github_token() else 'no'}")
    print(f"default repo: {DEFAULT_REPO}")
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "modal", "--version"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        modal_status = completed.stdout.strip() or f"exit {completed.returncode}"
    except Exception as exc:
        modal_status = f"unavailable: {exc}"
    print(f"python -m modal: {modal_status}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=["doctor", "smoke", "preset", "command"],
        help="What to run. Use doctor to print environment capabilities.",
    )
    parser.add_argument("--backend", choices=["auto", "direct", "workflow"], default="auto")
    parser.add_argument("--preset", default="qrico_key16")
    parser.add_argument("--tag", default="cloud")
    parser.add_argument("--command", default="python -m pytest -q")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--workflow", default=DEFAULT_WORKFLOW)
    parser.add_argument("--ref", default="main")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "doctor":
        doctor()
        return

    backend = choose_backend(args.backend)
    print(f"Using backend: {backend}", flush=True)
    if backend == "direct":
        direct_modal(args.mode, args.preset, args.tag, args.command)
    else:
        dispatch_workflow(
            repo=args.repo,
            workflow=args.workflow,
            ref=args.ref,
            mode=args.mode,
            preset=args.preset,
            tag=args.tag,
            command=args.command,
        )


if __name__ == "__main__":
    main()
