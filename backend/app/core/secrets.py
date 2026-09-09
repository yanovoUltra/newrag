"""Runtime secret resolution without writing credentials into project files."""

from __future__ import annotations

import os
from pathlib import Path

SERVICE_NAME = "NewRAG"
ALLOWED_SECRET_NAMES = frozenset(
    {
        "EMBEDDING_API_KEY",
        "LLM_API_KEY",
        "LLM_LIGHT_API_KEY",
        "LLM_TOKEN_PLAN_API_KEY",
        "PPOCR_TOKEN",
        "RERANK_API_KEY",
        "REGISTRY_DB_PASSWORD",
        "KEYCLOAK_DB_PASSWORD",
        "KEYCLOAK_ADMIN_PASSWORD",
    }
)


def resolve_secret(name: str, configured: str = "") -> str:
    """Resolve from explicit environment, Docker Secret file, then OS keyring.

    ``configured`` preserves environment-variable compatibility. Production
    operators should leave project ``.env`` placeholders empty and use ``*_FILE``
    or the operating-system key store.
    """

    normalized = name.strip().upper()
    if normalized not in ALLOWED_SECRET_NAMES:
        raise ValueError(f"Unsupported secret name: {normalized}")
    if configured:
        return configured
    file_name = os.getenv(f"{normalized}_FILE", "").strip()
    if file_name:
        path = Path(file_name)
        if not path.is_file():
            raise RuntimeError(f"Secret file is unavailable: {normalized}")
        if path.stat().st_size > 16_384:
            raise RuntimeError(f"Secret file is unexpectedly large: {normalized}")
        return path.read_text(encoding="utf-8").strip()
    try:
        import keyring

        return keyring.get_password(SERVICE_NAME, normalized) or ""
    except Exception:
        return ""
