"""API 认证：HMAC 验签 / 时间戳窗口 / nonce 防重放（sign_headers + verify_request + 中间件）。"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.security import sign_headers, verify_request


def _sec(**kw) -> Settings:
    base = dict(
        auth_enabled=True,
        auth_client_key="test-client",
        auth_secret="test-secret",
        auth_timestamp_window=300,
    )
    base.update(kw)
    return Settings(**base)


@pytest.fixture()
def auth_settings(monkeypatch):
    """让 security 模块读取开启认证的配置（verify_request / sign_headers 共用）。"""
    monkeypatch.setattr("app.core.security.get_settings", lambda: _sec())
    return _sec()


class TestVerifyRequest:
    @pytest.fixture(autouse=True)
    def _use_auth(self, auth_settings):
        pass

    def test_valid_signature(self):
        body = '{"question": "2024 年营收是多少？"}'.encode("utf-8")
        h = sign_headers("POST", "/api/v1/chat", body)
        ok, reason = verify_request("POST", "/api/v1/chat", body, {k.lower(): v for k, v in h.items()})
        assert ok, reason

    def test_missing_headers(self):
        ok, reason = verify_request("GET", "/api/v1/documents", b"", {})
        assert not ok and "缺少认证头" in reason

    def test_wrong_client_key(self):
        h = sign_headers("POST", "/api/v1/chat", b"{}")
        h["X-Client-Key"] = "evil"
        ok, _ = verify_request("POST", "/api/v1/chat", b"{}", {k.lower(): v for k, v in h.items()})
        assert not ok

    def test_tampered_body(self):
        body = '{"question": "原问题"}'.encode("utf-8")
        h = sign_headers("POST", "/api/v1/chat", body)
        ok, _ = verify_request("POST", "/api/v1/chat", '{"question": "被篡改"}'.encode("utf-8"), {k.lower(): v for k, v in h.items()})
        assert not ok

    def test_tampered_path(self):
        body = b"{}"
        h = sign_headers("POST", "/api/v1/chat", body)
        ok, _ = verify_request("POST", "/api/v1/documents", body, {k.lower(): v for k, v in h.items()})
        assert not ok

    def test_expired_timestamp(self):
        body = b"{}"
        h = sign_headers("POST", "/api/v1/chat", body, timestamp=int(time.time()) - 3600)
        ok, reason = verify_request("POST", "/api/v1/chat", body, {k.lower(): v for k, v in h.items()})
        assert not ok and "窗口" in reason

    def test_nonce_replay_rejected(self):
        body = b"{}"
        # 每轮唯一 nonce：避免固定值残留在 Redis（TTL）导致跨进程误判
        nonce = f"replay-{int(time.time() * 1000)}"
        h = sign_headers("POST", "/api/v1/chat", body, nonce=nonce)
        lowered = {k.lower(): v for k, v in h.items()}
        assert verify_request("POST", "/api/v1/chat", body, lowered)[0]
        # 同 nonce 重放 → 拒绝
        ok, reason = verify_request("POST", "/api/v1/chat", body, lowered)
        assert not ok and "重放" in reason

    def test_nonce_unique_allowed(self):
        body = b"{}"
        run = int(time.time() * 1000)
        for i in range(3):
            h = sign_headers("POST", "/api/v1/chat", body, nonce=f"n-{run}-{i}")
            ok, reason = verify_request("POST", "/api/v1/chat", body, {k.lower(): v for k, v in h.items()})
            assert ok, reason


class TestAuthMiddlewareHTTP:
    """HTTP 层：最小 FastAPI 应用 + AuthMiddleware（默认关闭时放行，开启时拦截）。"""

    @pytest.fixture()
    def http_app(self, monkeypatch):
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse

        from app.core.security import AuthMiddleware

        # 通过 monkeypatch 让中间件读到开启认证的配置
        monkeypatch.setattr("app.core.security.get_settings", lambda: _sec())
        app = FastAPI()
        app.add_middleware(AuthMiddleware)

        @app.get("/api/v1/ping")
        def ping():
            return {"ok": True}

        @app.get("/healthz")
        def healthz():
            return {"ok": True}

        return app

    def test_protected_without_headers_401(self, http_app):
        with TestClient(http_app) as c:
            r = c.get("/api/v1/ping")
            assert r.status_code == 401

    def test_protected_with_signature_200(self, http_app):
        with TestClient(http_app) as c:
            h = sign_headers("GET", "/api/v1/ping", b"")
            r = c.get("/api/v1/ping", headers=h)
            assert r.status_code == 200 and r.json()["ok"]

    def test_healthz_always_open(self, http_app):
        with TestClient(http_app) as c:
            assert c.get("/healthz").status_code == 200

    def test_options_preflight_open(self, http_app):
        with TestClient(http_app) as c:
            r = c.options("/api/v1/ping", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "GET"})
            # 预检不应被认证拦截（最小应用未定义 OPTIONS 路由，405 属正常；关键是非 401）
            assert r.status_code != 401

    def test_replay_http(self, http_app):
        with TestClient(http_app) as c:
            nonce = f"http-replay-{int(time.time() * 1000)}"
            h = sign_headers("GET", "/api/v1/ping", b"", nonce=nonce)
            assert c.get("/api/v1/ping", headers=h).status_code == 200
            assert c.get("/api/v1/ping", headers=h).status_code == 401
