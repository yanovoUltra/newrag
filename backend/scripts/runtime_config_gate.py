"""Validate a NewRAG runtime profile without printing credential values."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from app.core.config import Settings
from app.core.runtime import (
    effective_runtime_mode,
    effective_runtime_profile,
    model_feature_status,
    model_service_status,
    validate_model_runtime,
)

ROOT = Path(__file__).resolve().parents[2]


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _fingerprint(settings: Settings) -> str:
    endpoint_hashes = {
        name: hashlib.sha256(value.encode("utf-8")).hexdigest()
        for name, value in {
            "embedding": settings.embedding_api_base,
            "generation": settings.llm_base_url,
            "rerank": settings.rerank_api_base + settings.rerank_api_path,
            "ocr": settings.ppocr_job_url,
        }.items()
    }
    contract = {
        "runtime_profile": settings.runtime_profile,
        "model_runtime_mode": settings.model_runtime_mode,
        "embedding": [settings.embedding_backend, settings.embedding_model],
        "generation": [settings.llm_provider, settings.llm_model, settings.llm_light_model],
        "rerank": [
            settings.rerank_backend,
            settings.rerank_model,
            settings.rerank_indicator_model,
            settings.rerank_non_indicator_model,
            settings.rerank_api_auth_required,
        ],
        "ocr": [settings.ocr_enabled, settings.ppocr_model],
        "endpoints_sha256": endpoint_hashes,
    }
    payload = json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true", help="probe non-billable endpoint health")
    args = parser.parse_args()

    settings = Settings()
    features = model_feature_status(settings)
    probe = args.probe or settings.runtime_endpoint_probe
    services = model_service_status(settings, probe=probe)
    errors: list[str] = []
    try:
        validate_model_runtime(settings)
    except RuntimeError as exc:
        errors.append(str(exc))

    output = {
        "status": "pass" if not errors else "fail",
        "runtime_profile": effective_runtime_profile(settings, features),
        "runtime_mode": effective_runtime_mode(settings, features),
        "config_fingerprint_sha256": _fingerprint(settings),
        "git_sha": _git_sha(),
        "image_digest": os.getenv("NEWRAG_IMAGE_DIGEST", "unavailable"),
        "model_features": features,
        "model_services": services,
        "errors": errors,
    }
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
