"""Download the Qwen3.5 model targets used by the benchmark loop.

The full Qwen-Scope SAE repo is about 12 GiB. By default this script downloads
the model weights plus a representative SAE layer subset. Use --full-sae when
there is enough disk for all 24 layers.
"""

from __future__ import annotations

import argparse
import json
from typing import Iterable

from huggingface_hub import snapshot_download


FAST_MODEL = "Qwen/Qwen3.5-0.8B-Base"
INTERP_MODEL = "Qwen/Qwen3.5-2B-Base"
QWEN_SCOPE = "Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50"

MODEL_PATTERNS = [
    "*.json",
    "*.safetensors",
    "*.txt",
    "*.model",
    "*.py",
    "LICENSE",
    "README.md",
]
DEFAULT_SAE_LAYERS = (0, 4, 8, 12, 16, 20, 23)


def _parse_layers(value: str) -> list[int] | None:
    if value == "all":
        return None
    layers = []
    for raw in value.split(","):
        raw = raw.strip()
        if raw:
            layers.append(int(raw))
    if not layers:
        raise argparse.ArgumentTypeError("expected comma-separated layer ids or 'all'")
    return sorted(set(layers))


def _sae_patterns(layers: Iterable[int] | None) -> list[str]:
    if layers is None:
        return ["README.md", "layer*.sae.pt"]
    return ["README.md"] + [f"layer{layer}.sae.pt" for layer in layers]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--skip-sae", action="store_true")
    parser.add_argument(
        "--sae-layers",
        type=_parse_layers,
        default=list(DEFAULT_SAE_LAYERS),
        help="Comma-separated SAE layers to download, or 'all'.",
    )
    args = parser.parse_args()

    manifest: dict[str, str] = {}
    for repo in (FAST_MODEL, INTERP_MODEL):
        manifest[repo] = snapshot_download(
            repo_id=repo,
            allow_patterns=MODEL_PATTERNS,
            cache_dir=args.cache_dir,
            local_files_only=args.local_files_only,
            max_workers=8,
        )

    if not args.skip_sae:
        manifest[QWEN_SCOPE] = snapshot_download(
            repo_id=QWEN_SCOPE,
            allow_patterns=_sae_patterns(args.sae_layers),
            cache_dir=args.cache_dir,
            local_files_only=args.local_files_only,
            max_workers=3,
        )

    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
