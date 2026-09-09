"""Fail-closed model runtime contract shared by API, worker and public config."""

from __future__ import annotations

import socket
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin, urlparse

from app.core.config import Settings
from app.core.secrets import resolve_secret

MODEL_FEATURES = ("embedding", "generation", "rerank", "ocr")
RUNTIME_PROFILES = {
    "local-real",
    "local-parser-benchmark",
}


def _rerank_has_auth(settings: Settings) -> bool:
    return not settings.rerank_api_auth_required or bool(
        resolve_secret("RERANK_API_KEY", settings.rerank_api_key)
    )


def model_feature_status(settings: Settings) -> dict[str, bool]:
    """Return configured capabilities without claiming that endpoints are healthy."""

    embedding = settings.embedding_backend == "flagembedding" or (
        settings.embedding_backend == "api"
        and bool(resolve_secret("EMBEDDING_API_KEY", settings.embedding_api_key))
    )
    generation = settings.llm_provider == "openai" and bool(
        resolve_secret(settings.llm_api_secret_name, settings.llm_api_key)
    )
    rerank = settings.rerank_backend == "api" and _rerank_has_auth(settings)
    ocr = settings.ocr_enabled and bool(
        resolve_secret("PPOCR_TOKEN", settings.ppocr_token)
    )
    return {
        "embedding": embedding,
        "generation": generation,
        "rerank": rerank,
        "ocr": ocr,
    }


def effective_runtime_mode(
    settings: Settings, features: dict[str, bool] | None = None
) -> str:
    mode = settings.model_runtime_mode.strip().lower()
    if mode not in {"auto", "demo", "real"}:
        raise RuntimeError("MODEL_RUNTIME_MODE must be auto, demo or real")
    if mode != "auto":
        return mode
    available = features or model_feature_status(settings)
    return "real" if available["embedding"] and available["generation"] else "demo"


def effective_runtime_profile(
    settings: Settings, features: dict[str, bool] | None = None
) -> str:
    """Return the explicit profile or a labelled legacy compatibility value."""

    profile = settings.runtime_profile.strip().lower()
    if profile:
        if profile not in RUNTIME_PROFILES:
            raise RuntimeError(
                "RUNTIME_PROFILE must be local-real or local-parser-benchmark"
            )
        return profile
    return f"legacy-{effective_runtime_mode(settings, features)}"


def _service_urls(settings: Settings) -> dict[str, tuple[str, str]]:
    return {
        "embedding": (settings.embedding_api_base, ""),
        "generation": (settings.llm_base_url, ""),
        "rerank": (settings.rerank_api_base, settings.rerank_health_path),
        "ocr": (settings.ppocr_job_url, ""),
    }


def _tcp_reachable(url: str, timeout: float) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    with socket.create_connection((parsed.hostname, port), timeout=timeout):
        return True


def _http_health(url: str, path: str, timeout: float) -> bool:
    import httpx

    health_url = urljoin(url.rstrip("/") + "/", path.lstrip("/"))
    response = httpx.get(health_url, timeout=timeout)
    if response.status_code != 200:
        return False
    try:
        payload = response.json()
    except ValueError:
        return True
    status = str(payload.get("status", payload.get("app", "ok"))).lower()
    return status in {"ok", "healthy", "ready"}


def _probe_service(url: str, health_path: str, timeout: float) -> tuple[bool, bool | None]:
    try:
        reachable = _tcp_reachable(url, timeout)
    except (OSError, ValueError):
        return False, False if health_path else None
    if not reachable:
        return False, False if health_path else None
    if not health_path:
        return True, None
    try:
        return True, _http_health(url, health_path, timeout)
    except Exception:  # noqa: BLE001 - status only; never expose provider errors
        return True, False


def model_service_status(settings: Settings, *, probe: bool = False) -> dict[str, dict]:
    """Report CONFIGURED/REACHABLE/HEALTHY separately without leaking endpoints."""

    configured = model_feature_status(settings)
    urls = _service_urls(settings)
    enabled = {
        "embedding": settings.embedding_backend != "mock",
        "generation": settings.llm_provider != "mock",
        "rerank": settings.rerank_backend != "none",
        "ocr": settings.ocr_enabled,
    }
    result: dict[str, dict] = {}
    for name in MODEL_FEATURES:
        state = (
            "disabled"
            if not enabled[name]
            else ("configured" if configured[name] else "misconfigured")
        )
        result[name] = {
            "state": state,
            "configured": configured[name],
            "reachable": None,
            "healthy": None,
        }

    if not probe:
        return result

    pending: dict = {}
    with ThreadPoolExecutor(max_workers=4, thread_name_prefix="model-health") as executor:
        for name in MODEL_FEATURES:
            if not configured[name]:
                continue
            url, health_path = urls[name]
            if name == "embedding" and settings.embedding_backend == "flagembedding":
                continue
            future = executor.submit(
                _probe_service, url, health_path, settings.runtime_probe_timeout
            )
            pending[future] = name
        for future in as_completed(pending):
            name = pending[future]
            reachable, healthy = future.result()
            result[name]["reachable"] = reachable
            result[name]["healthy"] = healthy
            if healthy is True:
                result[name]["state"] = "healthy"
            elif reachable:
                result[name]["state"] = "reachable" if healthy is None else "unhealthy"
            else:
                result[name]["state"] = "unreachable"
    return result


def _required_features(settings: Settings, profile: str, mode: str) -> set[str]:
    if mode != "real":
        return set()
    required = {"embedding", "generation"}
    if settings.rerank_backend == "api" or profile == "local-real":
        required.add("rerank")
    if settings.ocr_enabled:
        required.add("ocr")
    return required


def validate_model_runtime(settings: Settings) -> dict[str, bool]:
    """Reject misleading profile/configuration combinations before serving traffic."""

    features = model_feature_status(settings)
    mode = effective_runtime_mode(settings, features)
    profile = effective_runtime_profile(settings, features)
    configured_mode = settings.model_runtime_mode.strip().lower()

    if settings.ocr_enabled and not features["ocr"]:
        raise RuntimeError("OCR_ENABLED=true requires a resolvable PPOCR_TOKEN")

    if profile == "local-parser-benchmark":
        if configured_mode != "demo":
            raise RuntimeError(f"{profile} requires MODEL_RUNTIME_MODE=demo")
        model_free = (
            settings.embedding_backend == "mock"
            and settings.llm_provider == "mock"
            and settings.rerank_backend == "none"
            and not settings.ocr_enabled
        )
        if not model_free:
            raise RuntimeError(f"{profile} must disable all application model services")
    elif profile == "local-real":
        if configured_mode != "real":
            raise RuntimeError(f"{profile} requires MODEL_RUNTIME_MODE=real")
        if settings.embedding_backend == "mock" or settings.llm_provider == "mock":
            raise RuntimeError(f"{profile} forbids mock embedding and generation")
        if settings.rerank_backend != "api":
            raise RuntimeError(f"{profile} requires RERANK_BACKEND=api")
    elif configured_mode == "demo" and any(features.values()):
        raise RuntimeError("demo profile cannot expose real model features")

    required = _required_features(settings, profile, mode)
    missing = sorted(name for name in required if not features[name])
    if missing:
        raise RuntimeError(
            "real profile is missing required model features: " + ", ".join(missing)
        )

    if settings.runtime_endpoint_probe and mode == "real":
        services = model_service_status(settings, probe=True)
        unavailable = []
        for name in required:
            detail = services[name]
            if detail["reachable"] is False or detail["healthy"] is False:
                unavailable.append(name)
        if unavailable:
            raise RuntimeError(
                "real profile has unreachable or unhealthy model services: "
                + ", ".join(sorted(unavailable))
            )
    return features
