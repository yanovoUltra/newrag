"""SSE 流式问答。"""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from app.core.config import get_settings
from app.pipelines.answer import stream_answer
from app.schemas import ChatRequest

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("")
async def chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(400, "问题不能为空")
    settings = get_settings()

    async def event_gen():
        try:
            # 会话级超时：整条 SSE 流超过 chat_timeout 秒 → 发 error + done 并关闭
            async with asyncio.timeout(settings.chat_timeout):
                async for evt in stream_answer(
                    req.question,
                    org_id=req.org_id,
                    user_visibility=req.user_visibility,
                    session_id=req.session_id,
                ):
                    data = evt["data"]
                    if evt["event"] == "meta":
                        data = {**data, "session_id": req.session_id, "question": req.question}
                    yield {"event": evt["event"], "data": json.dumps(data, ensure_ascii=False)}
        except TimeoutError:
            yield {"event": "error", "data": json.dumps({"message": f"回答超时（>{settings.chat_timeout}s），请重试。"}, ensure_ascii=False)}
            yield {"event": "done", "data": json.dumps({"usage": {}}, ensure_ascii=False)}

    return EventSourceResponse(event_gen())
