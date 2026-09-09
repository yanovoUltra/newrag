"""PostgreSQL/SQLite-backed lightweight evidence graph for bounded multi-hop recall.

The graph stores only traceable relationships extracted from registered financial
fields.  It never replaces text retrieval and it never broadens tenant or
visibility filters.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict, deque

from sqlalchemy import delete, or_, select

from app.core.config import get_settings
from app.core.logging import get_logger
from app.fields.metrics import extract_metric_from_question
from app.fields.subject import find_mentioned_companies, find_subject_company
from app.models.entities import Document, EvidenceEdge, EvidenceNode, FinancialField
from app.store import qdrant as qdrant_store
from app.store.registry import get_session

logger = get_logger(__name__)

_YEAR_RE = re.compile(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)")


def _node(
    *,
    doc: Document,
    node_type: str,
    key: str,
    label: str,
    chunk_id: str = "",
    fiscal_year: int | None = None,
    confidence: float = 1.0,
    attributes: dict | None = None,
) -> EvidenceNode:
    return EvidenceNode(
        org_id=doc.org_id,
        visibility=doc.visibility,
        node_type=node_type,
        canonical_key=f"{doc.id}:{node_type}:{key}",
        label=label,
        attributes_json=json.dumps(attributes or {}, ensure_ascii=False, sort_keys=True),
        source_doc_id=doc.id,
        source_chunk_id=chunk_id,
        fiscal_year=fiscal_year,
        confidence=confidence,
    )


def _edge(
    *,
    doc: Document,
    edge_type: str,
    source: EvidenceNode,
    target: EvidenceNode,
    chunk_id: str = "",
    fiscal_year: int | None = None,
    confidence: float = 1.0,
    citation: dict | None = None,
) -> EvidenceEdge:
    return EvidenceEdge(
        org_id=doc.org_id,
        visibility=doc.visibility,
        edge_type=edge_type,
        source_node_id=source.id,
        target_node_id=target.id,
        source_doc_id=doc.id,
        source_chunk_id=chunk_id,
        fiscal_year=fiscal_year,
        confidence=confidence,
        citation_json=json.dumps(citation or {}, ensure_ascii=False, sort_keys=True),
    )


def rebuild_document_graph(doc_id: str) -> dict[str, int]:
    """Idempotently rebuild one document's graph after field/chunk backfill."""

    if not get_settings().evidence_graph_enabled:
        return {"nodes": 0, "edges": 0}

    with get_session() as session:
        doc = session.get(Document, doc_id)
        if doc is None:
            raise ValueError("document_not_found")
        fields = list(
            session.scalars(
                select(FinancialField).where(
                    FinancialField.doc_id == doc_id,
                    FinancialField.source_chunk_id != "",
                )
            )
        )
        session.execute(delete(EvidenceEdge).where(EvidenceEdge.source_doc_id == doc_id))
        session.execute(delete(EvidenceNode).where(EvidenceNode.source_doc_id == doc_id))

        document = _node(
            doc=doc,
            node_type="document",
            key=doc.id,
            label=doc.filename,
            fiscal_year=doc.fiscal_year,
        )
        session.add(document)

        nodes: dict[tuple[str, str], EvidenceNode] = {("document", doc.id): document}
        evidence_nodes: dict[str, EvidenceNode] = {}
        for field in fields:
            company = (field.company or doc.filename).strip()
            company_key = company.casefold()
            metric_key = field.metric.casefold()
            year_key = str(field.year)
            for node_type, key, label, year in (
                ("company", company_key, company, None),
                ("metric", metric_key, field.metric_label or field.metric, None),
                ("fiscal_year", year_key, year_key, field.year),
            ):
                if (node_type, key) not in nodes:
                    nodes[(node_type, key)] = _node(
                        doc=doc,
                        node_type=node_type,
                        key=key,
                        label=label,
                        fiscal_year=year,
                    )
                    session.add(nodes[(node_type, key)])

            chunk_id = field.source_chunk_id
            if chunk_id not in evidence_nodes:
                evidence_nodes[chunk_id] = _node(
                    doc=doc,
                    node_type="evidence_chunk",
                    key=chunk_id,
                    label=field.section_path or field.metric_label or field.metric,
                    chunk_id=chunk_id,
                    fiscal_year=field.year,
                    attributes={"page": field.page, "section_path": field.section_path},
                )
                session.add(evidence_nodes[chunk_id])

        session.flush()
        edges: list[EvidenceEdge] = []
        linked_company: set[str] = set()
        linked_chunk: set[str] = set()
        for field in fields:
            company_key = (field.company or doc.filename).strip().casefold()
            company = nodes[("company", company_key)]
            metric = nodes[("metric", field.metric.casefold())]
            year = nodes[("fiscal_year", str(field.year))]
            evidence = evidence_nodes[field.source_chunk_id]
            citation = {
                "doc_id": doc.id,
                "chunk_id": field.source_chunk_id,
                "page": field.page,
                "section_path": field.section_path,
            }
            if company_key not in linked_company:
                edges.append(_edge(doc=doc, edge_type="REPORTS", source=company, target=document))
                linked_company.add(company_key)
            if field.source_chunk_id not in linked_chunk:
                edges.append(
                    _edge(
                        doc=doc,
                        edge_type="PARENT_OF",
                        source=document,
                        target=evidence,
                        chunk_id=field.source_chunk_id,
                        fiscal_year=field.year,
                        citation=citation,
                    )
                )
                linked_chunk.add(field.source_chunk_id)
            edges.extend(
                (
                    _edge(
                        doc=doc,
                        edge_type="SOURCE_OF",
                        source=evidence,
                        target=metric,
                        chunk_id=field.source_chunk_id,
                        fiscal_year=field.year,
                        citation=citation,
                    ),
                    _edge(
                        doc=doc,
                        edge_type="REPORTS",
                        source=evidence,
                        target=year,
                        chunk_id=field.source_chunk_id,
                        fiscal_year=field.year,
                        citation=citation,
                    ),
                )
            )

        year_nodes = sorted(
            (node for (kind, _), node in nodes.items() if kind == "fiscal_year"),
            key=lambda item: item.fiscal_year or 0,
            reverse=True,
        )
        for current, previous in zip(year_nodes, year_nodes[1:]):
            edges.append(
                _edge(
                    doc=doc,
                    edge_type="PREVIOUS_PERIOD",
                    source=current,
                    target=previous,
                    fiscal_year=current.fiscal_year,
                )
            )
        session.add_all(edges)
        session.commit()
        logger.info("evidence graph rebuilt: doc=%s nodes=%d edges=%d", doc_id, len(nodes) + len(evidence_nodes), len(edges))
        return {"nodes": len(nodes) + len(evidence_nodes), "edges": len(edges)}


def _reachable_evidence(
    session,
    start_ids: set[str],
    org_id: str,
    visible: list[str],
    max_hops: int,
) -> set[str]:
    visited = set(start_ids)
    queue = deque((node_id, 0) for node_id in start_ids)
    while queue:
        node_id, depth = queue.popleft()
        if depth >= max_hops:
            continue
        edges = session.scalars(
            select(EvidenceEdge).where(
                EvidenceEdge.org_id == org_id,
                EvidenceEdge.visibility.in_(visible),
                or_(
                    EvidenceEdge.source_node_id == node_id,
                    EvidenceEdge.target_node_id == node_id,
                ),
            )
        )
        for edge in edges:
            adjacent = (
                edge.target_node_id if edge.source_node_id == node_id else edge.source_node_id
            )
            if adjacent not in visited:
                visited.add(adjacent)
                queue.append((adjacent, depth + 1))
    return set(
        session.scalars(
            select(EvidenceNode.id).where(
                EvidenceNode.id.in_(visited),
                EvidenceNode.org_id == org_id,
                EvidenceNode.visibility.in_(visible),
                EvidenceNode.node_type == "evidence_chunk",
            )
        )
    )


def graph_evidence_refs(
    question: str,
    org_id: str,
    user_visibility: str,
    *,
    max_hops: int = 2,
    limit: int = 50,
) -> list[tuple[str, str]]:
    """Return tenant-filtered evidence references for explicit multi-hop anchors."""

    if not get_settings().evidence_graph_enabled:
        return []
    visible = qdrant_store.visible_levels(user_visibility)
    metric, metric_year = extract_metric_from_question(question)
    years = {int(value) for value in _YEAR_RE.findall(question)}
    if metric_year:
        years.add(metric_year)

    with get_session() as session:
        company_labels = set(
            session.scalars(
                select(EvidenceNode.label).where(
                    EvidenceNode.org_id == org_id,
                    EvidenceNode.visibility.in_(visible),
                    EvidenceNode.node_type == "company",
                )
            )
        )
        company = find_subject_company(question, list(company_labels))
        companies = [company] if company else find_mentioned_companies(question, list(company_labels))

        anchors: list[set[str]] = []
        if companies:
            anchors.append(
                set(
                    session.scalars(
                        select(EvidenceNode.id).where(
                            EvidenceNode.org_id == org_id,
                            EvidenceNode.visibility.in_(visible),
                            EvidenceNode.node_type == "company",
                            EvidenceNode.label.in_(companies),
                        )
                    )
                )
            )
        if metric:
            anchors.append(
                set(
                    session.scalars(
                        select(EvidenceNode.id).where(
                            EvidenceNode.org_id == org_id,
                            EvidenceNode.visibility.in_(visible),
                            EvidenceNode.node_type == "metric",
                            EvidenceNode.canonical_key.like(f"%:metric:{metric.casefold()}"),
                        )
                    )
                )
            )
        if years:
            anchors.append(
                set(
                    session.scalars(
                        select(EvidenceNode.id).where(
                            EvidenceNode.org_id == org_id,
                            EvidenceNode.visibility.in_(visible),
                            EvidenceNode.node_type == "fiscal_year",
                            EvidenceNode.fiscal_year.in_(years),
                        )
                    )
                )
            )
        anchors = [anchor for anchor in anchors if anchor]
        if not anchors:
            return []
        evidence_sets = [
            _reachable_evidence(session, anchor, org_id, visible, max_hops) for anchor in anchors
        ]
        selected = set.intersection(*evidence_sets) if evidence_sets else set()
        if not selected:
            selected = set.union(*evidence_sets)
        rows = session.execute(
            select(EvidenceNode.source_doc_id, EvidenceNode.source_chunk_id)
            .where(
                EvidenceNode.id.in_(selected),
                EvidenceNode.org_id == org_id,
                EvidenceNode.visibility.in_(visible),
            )
            .limit(limit)
        ).all()
    return [(doc_id, chunk_id) for doc_id, chunk_id in rows if chunk_id]


def graph_evidence_hits(
    question: str,
    org_id: str,
    user_visibility: str,
    *,
    limit: int = 20,
) -> list[dict]:
    """Resolve graph references to Qdrant payloads with a second permission check."""

    refs = graph_evidence_refs(question, org_id, user_visibility, limit=limit)
    by_doc: dict[str, list[str]] = defaultdict(list)
    for doc_id, chunk_id in refs:
        by_doc[doc_id].append(chunk_id)
    visible = set(qdrant_store.visible_levels(user_visibility))
    hits: list[dict] = []
    for doc_id, chunk_ids in by_doc.items():
        payloads = qdrant_store.fetch_payloads(doc_id, chunk_ids)
        for chunk_id in chunk_ids:
            payload = payloads.get(chunk_id) or {}
            if payload.get("org_id") != org_id or payload.get("visibility") not in visible:
                continue
            hits.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "doc_name": payload.get("doc_name", ""),
                    "page": payload.get("page"),
                    "section_path": payload.get("section_path", ""),
                    "chunk_type": payload.get("chunk_type", "text"),
                    "content": payload.get("content", ""),
                    "parent_id": payload.get("parent_id"),
                    "score": 1.0,
                    "fused_score": 1.0,
                    "rerank_score": None,
                    "org_id": org_id,
                    "visibility": payload.get("visibility"),
                    "is_structural": bool(payload.get("is_structural")),
                    "evidence_aspect": "evidence_graph",
                    "retrieval_source": "evidence_graph",
                }
            )
    return hits[:limit]
