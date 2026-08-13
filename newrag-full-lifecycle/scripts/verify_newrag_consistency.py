#!/usr/bin/env python3
"""Read-only SQLite/Qdrant consistency verifier for NewRAG-like schemas."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


def request_json(url: str, method: str = "GET", body: dict[str, Any] | None = None) -> dict[str, Any]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"Qdrant HTTP {exc.code}: {detail}") from exc


def scroll_points(base: str, collection: str) -> list[dict[str, Any]]:
    url = f"{base.rstrip('/')}/collections/{collection}/points/scroll"
    points: list[dict[str, Any]] = []
    offset: Any = None
    while True:
        body: dict[str, Any] = {"limit": 256, "with_payload": True, "with_vector": False}
        if offset is not None:
            body["offset"] = offset
        result = request_json(url, "POST", body).get("result", {})
        batch = result.get("points", [])
        points.extend(batch)
        offset = result.get("next_page_offset")
        if offset is None or not batch:
            return points


def table_exists(db: sqlite3.Connection, table: str) -> bool:
    row = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
    return row is not None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path, help="SQLite registry path")
    parser.add_argument("--qdrant-url", default="http://127.0.0.1:6333")
    parser.add_argument("--collection", default="chunks")
    parser.add_argument("--org", default="default")
    parser.add_argument("--strict-org-only", action="store_true", help="Fail if any vector belongs to another org")
    args = parser.parse_args()

    if not args.db.is_file():
        parser.error(f"database does not exist: {args.db}")

    db = sqlite3.connect(f"file:{args.db.resolve()}?mode=ro", uri=True)
    integrity = db.execute("PRAGMA integrity_check").fetchone()[0]
    doc_rows = db.execute("SELECT id, status FROM documents WHERE org_id=?", (args.org,)).fetchall()
    db_doc_ids = {row[0] for row in doc_rows}
    statuses = Counter(row[1] for row in doc_rows)

    field_count = 0
    source_ids: set[str] = set()
    if table_exists(db, "financial_fields"):
        field_count = db.execute("SELECT count(*) FROM financial_fields WHERE org_id=?", (args.org,)).fetchone()[0]
        columns = {row[1] for row in db.execute("PRAGMA table_info(financial_fields)")}
        if "source_chunk_id" in columns:
            source_ids = {
                row[0]
                for row in db.execute(
                    "SELECT DISTINCT source_chunk_id FROM financial_fields WHERE org_id=? AND source_chunk_id<>''",
                    (args.org,),
                )
            }

    points = scroll_points(args.qdrant_url, args.collection)
    org_counts = Counter(str((point.get("payload") or {}).get("org_id", "")) for point in points)
    org_points = [point for point in points if str((point.get("payload") or {}).get("org_id", "")) == args.org]
    vector_doc_ids = {
        str((point.get("payload") or {}).get("doc_id"))
        for point in org_points
        if (point.get("payload") or {}).get("doc_id")
    }
    vector_chunk_ids = {
        str((point.get("payload") or {}).get("chunk_id", point.get("id"))) for point in org_points
    }

    missing_vectors = sorted(db_doc_ids - vector_doc_ids)
    missing_db_docs = sorted(vector_doc_ids - db_doc_ids)
    missing_evidence = sorted(source_ids - vector_chunk_ids)
    foreign_orgs = {key: value for key, value in org_counts.items() if key != args.org and value}
    errors: list[str] = []
    if integrity != "ok":
        errors.append("sqlite_integrity")
    if missing_vectors:
        errors.append("db_docs_missing_vectors")
    if missing_db_docs:
        errors.append("vector_docs_missing_db")
    if missing_evidence:
        errors.append("field_sources_missing_vectors")
    if args.strict_org_only and foreign_orgs:
        errors.append("foreign_org_vectors")

    report = {
        "ok": not errors,
        "errors": errors,
        "sqlite_integrity": integrity,
        "documents": len(doc_rows),
        "document_statuses": dict(statuses),
        "financial_fields": field_count,
        "collection_points": len(points),
        "org_points": len(org_points),
        "org_counts": dict(org_counts),
        "db_docs_missing_vectors": missing_vectors,
        "vector_docs_missing_db": missing_db_docs,
        "field_source_ids": len(source_ids),
        "field_sources_missing_vectors": missing_evidence,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
