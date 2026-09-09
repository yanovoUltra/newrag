#!/usr/bin/env python3
"""Create or update the first NewRAG tenant user without logging passwords.

The bootstrap-admin password and the user's initial password are read as two
newline-separated values from stdin. The script is intended to run in a
short-lived container on the private Compose network.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request


def _request(
    url: str,
    *,
    method: str = "GET",
    token: str | None = None,
    payload: dict | list | None = None,
    form: dict[str, str] | None = None,
) -> tuple[int, object | None]:
    headers: dict[str, str] = {"Accept": "application/json"}
    body: bytes | None = None
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if form is not None:
        headers["Content-Type"] = "application/x-www-form-urlencoded"
        body = urllib.parse.urlencode(form).encode()
    elif payload is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(payload).encode()
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            raw = response.read()
            return response.status, json.loads(raw) if raw else None
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"Keycloak request failed: status={exc.code} detail={detail}") from exc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://keycloak:8080/auth")
    parser.add_argument("--username", default="newrag-admin")
    parser.add_argument("--org-id", default="default")
    parser.add_argument(
        "--visibility",
        choices=("public", "internal", "restricted"),
        default="restricted",
    )
    args = parser.parse_args()

    secrets = [line.rstrip("\r\n") for line in sys.stdin.readlines()]
    if len(secrets) < 2 or not secrets[0] or not secrets[1]:
        raise RuntimeError("expected bootstrap-admin and user passwords on stdin")
    admin_password, user_password = secrets[:2]

    _, token_result = _request(
        f"{args.server}/realms/master/protocol/openid-connect/token",
        method="POST",
        form={
            "grant_type": "password",
            "client_id": "admin-cli",
            "username": "bootstrap-admin",
            "password": admin_password,
        },
    )
    if not isinstance(token_result, dict) or not token_result.get("access_token"):
        raise RuntimeError("Keycloak admin token response was invalid")
    token = str(token_result["access_token"])
    admin_base = f"{args.server}/admin/realms/newrag"

    query = urllib.parse.urlencode({"username": args.username, "exact": "true"})
    _, users = _request(f"{admin_base}/users?{query}", token=token)
    matches = users if isinstance(users, list) else []
    user_payload = {
        "username": args.username,
        "enabled": True,
        "attributes": {
            "org_id": [args.org_id],
            "visibility": [args.visibility],
        },
        "credentials": [
            {"type": "password", "value": user_password, "temporary": False}
        ],
    }
    created = not matches
    if created:
        _request(f"{admin_base}/users", method="POST", token=token, payload=user_payload)
        _, users = _request(f"{admin_base}/users?{query}", token=token)
        matches = users if isinstance(users, list) else []
    if len(matches) != 1 or not isinstance(matches[0], dict):
        raise RuntimeError("unable to resolve the bootstrapped user")
    user_id = str(matches[0]["id"])
    if not created:
        _request(
            f"{admin_base}/users/{user_id}",
            method="PUT",
            token=token,
            payload=user_payload,
        )

    roles: list[dict] = []
    for role_name in ("tenant_admin", "analyst", "viewer"):
        _, role = _request(f"{admin_base}/roles/{role_name}", token=token)
        if not isinstance(role, dict):
            raise RuntimeError(f"realm role is missing: {role_name}")
        roles.append(role)
    _request(
        f"{admin_base}/users/{user_id}/role-mappings/realm",
        method="POST",
        token=token,
        payload=roles,
    )

    print(
        json.dumps(
            {
                "username": args.username,
                "created": created,
                "org_id": args.org_id,
                "visibility": args.visibility,
                "roles": [role["name"] for role in roles],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
