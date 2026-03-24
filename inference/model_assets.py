"""Resolve model artifact directories from local disk or Hugging Face Hub."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from huggingface_hub import snapshot_download

DEFAULT_MODEL_REPO_ID = "salmavkh/ai-startup-review-models"
_LOCAL_ARTIFACTS_ROOT = Path("artifacts")


def _clean_env(value: str | None) -> str:
    return str(value or "").strip()


@lru_cache(maxsize=32)
def _download_subdir(repo_id: str, subdir: str, token: str | None) -> str:
    """Download a specific subdirectory from a model repo and return snapshot path."""
    return snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        allow_patterns=[f"{subdir}/**"],
        token=token,
    )


@lru_cache(maxsize=32)
def resolve_artifact_dir(subdir: str) -> str:
    """
    Resolve an artifact subdir to a local path.

    Order:
    1) Local `artifacts/<subdir>` if present
    2) Download from MODEL_REPO_ID (default: salmavkh/ai-startup-review-models)
    """
    normalized = str(subdir or "").strip().strip("/")
    if not normalized:
        raise ValueError("subdir must be non-empty")

    local_dir = _LOCAL_ARTIFACTS_ROOT / normalized
    if local_dir.exists():
        return str(local_dir)

    repo_id = _clean_env(os.getenv("MODEL_REPO_ID")) or DEFAULT_MODEL_REPO_ID
    token = _clean_env(os.getenv("HF_TOKEN")) or None

    snapshot_dir = _download_subdir(repo_id=repo_id, subdir=normalized, token=token)
    resolved = Path(snapshot_dir) / normalized
    if not resolved.exists():
        raise FileNotFoundError(
            f"Could not find '{normalized}' in model repo '{repo_id}'."
        )
    return str(resolved)
