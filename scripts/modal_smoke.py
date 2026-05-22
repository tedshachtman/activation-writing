"""Minimal Modal smoke test for remote execution."""

from __future__ import annotations

import modal


app = modal.App("caic-modal-smoke")


@app.function()
def hello() -> str:
    import platform

    return f"modal ok: python={platform.python_version()} machine={platform.machine()}"


@app.local_entrypoint()
def main() -> None:
    print(hello.remote())
