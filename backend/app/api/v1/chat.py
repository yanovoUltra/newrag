"""SSE 流式问答。"""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from app.pipelines.answer import stream_answer
from app.schemas import ChatRequest

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("")
async def chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(400, "问题不能为空")

    async def event_gen():
        async for evt in stream_answer(
            req.question,
            org_id=req.org_id,
            user_visibility=req.user_visibility,
        ):
            data = evt["data"]
            if evt["event"] == "meta":
                data = {**data, "session_id": req.session_id, "question": req.question}
            yield {"event": evt["event"], "data": json.dumps(data, ensure_ascii=False)}

    return EventSourceResponse(event_gen())
