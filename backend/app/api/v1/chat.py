"""SSE 流式问答。"""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, HTTPException, Request
from sse_starlette.sse import EventSourceResponse

from app.core.config import get_settings
from app.core.identity import read_visibility, request_principal, require_roles, resolve_org
from app.core.logging import get_trace_id
from app.pipelines.answer import stream_answer
from app.schemas import ChatRequest

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("")
async def chat(req: ChatRequest, request: Request):
    if not req.question.strip():
        raise HTTPException(400, "问题不能为空")
    settings = get_settings()
    principal = request_principal(request)
    require_roles(principal, "tenant_admin", "analyst", "viewer")
    org_id = resolve_org(principal, req.org_id)
    user_visibility = read_visibility(principal, req.user_visibility)

    async def event_gen():
        try:
            # 会话级超时：整条 SSE 流超过 chat_timeout 秒 → 发 error + done 并关闭
            async with asyncio.timeout(settings.chat_timeout):
                async for evt in stream_answer(
                    req.question,
                    org_id=org_id,
                    user_visibility=user_visibility,
                    session_id=req.session_id,
                    response_mode=req.response_mode,
                    use_tools=req.use_tools,
                    actor_id=principal.subject if principal is not None else "anonymous",
                ):
                    data = evt["data"]
                    if evt["event"] == "meta":
                        data = {**data, "session_id": req.session_id}
                    yield {"event": evt["event"], "data": json.dumps(data, ensure_ascii=False)}
        except TimeoutError:
            yield {
                "event": "error",
                "data": json.dumps(
                    {
                        "code": "answer_timeout",
                        "message": "回答超时，请重试。",
                        "request_id": get_trace_id(),
                    },
                    ensure_ascii=False,
                ),
            }
            yield {"event": "done", "data": json.dumps({"usage": {}}, ensure_ascii=False)}

    return EventSourceResponse(event_gen())
