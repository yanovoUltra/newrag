from app.api.v1.documents import MIN_FISCAL_YEAR, _current_year
from app.core.config import get_settings
from app.parsers.base import SUPPORTED_EXTENSIONS


def test_public_config_matches_server_settings(client):
    response = client.get("/api/v1/config/public")

    assert response.status_code == 200
    data = response.json()
    assert data == {
        "max_upload_mb": get_settings().max_upload_mb,
        "accepted_extensions": sorted(SUPPORTED_EXTENSIONS),
        "fiscal_year_min": MIN_FISCAL_YEAR,
        "current_year": _current_year(),
    }


def test_livez_is_fast_process_only(client):
    response = client.get("/livez")

    assert response.status_code == 200
    assert response.json() == {"app": "ok"}


def test_cache_metrics_endpoint_is_read_only(client, monkeypatch):
    from app.api.v1 import metrics

    expected = {
        "redis_available": True,
        "redis_total_keys": 12,
        "redis_used_memory_bytes": 4096,
        "embed_query_keys": 2,
        "embed_document_keys": 0,
        "embed_cache_hits": 3,
        "embed_cache_misses": 1,
        "embed_cache_bypassed": 7,
        "embed_cache_unavailable": 0,
        "embed_cache_hit_rate": 0.75,
    }
    monkeypatch.setattr(metrics, "cache_metrics_snapshot", lambda: expected)

    response = client.get("/api/v1/metrics/cache")

    assert response.status_code == 200
    assert response.json() == expected
