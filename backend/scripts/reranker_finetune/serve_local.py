"""Serve Qwen3-Reranker-8B plus an optional PEFT adapter as a local HTTP API.

The response shape matches the DashScope rerank endpoint already consumed by
``app.retrieval.reranker.ApiReranker`` so production search code needs no fork.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import threading
from contextlib import asynccontextmanager, nullcontext
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .train_qlora import INSTRUCTION_FALLBACK, PREFIX, SUFFIX, format_pair


class RerankInput(BaseModel):
    query: str = Field(min_length=1, max_length=8_000)
    documents: list[str] = Field(min_length=1, max_length=512)


class RerankParameters(BaseModel):
    top_n: int | None = Field(default=None, ge=1, le=512)
    instruction: str = INSTRUCTION_FALLBACK


class RerankRequest(BaseModel):
    model: str = "qwen3-reranker-8b-local"
    input: RerankInput
    parameters: RerankParameters = Field(default_factory=RerankParameters)


def format_results(scores: list[float], documents: list[str], top_n: int | None) -> list[dict[str, Any]]:
    limit = min(top_n or len(scores), len(scores))
    ranking = sorted(range(len(scores)), key=lambda index: (scores[index], -index), reverse=True)[:limit]
    return [
        {"index": index, "relevance_score": float(scores[index]), "document": documents[index]}
        for index in ranking
    ]


class LocalQwenReranker:
    def __init__(
        self,
        model_path: Path,
        adapter_path: Path | None,
        max_length: int,
        batch_size: int,
        *,
        indicator_adapter: Path | None = None,
        non_indicator_adapter: Path | None = None,
        default_profile: str = "base",
    ):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for Qwen3-Reranker-8B local serving")
        self._torch = torch
        self._max_length = max_length
        self._batch_size = batch_size
        self._lock = threading.Lock()
        self._tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        self._tokenizer.pad_token = self._tokenizer.pad_token or self._tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            low_cpu_mem_usage=True,
            offload_state_dict=True,
        )
        adapters: dict[str, Path] = {}
        if indicator_adapter:
            adapters["finance-indicator"] = indicator_adapter
        if non_indicator_adapter or adapter_path:
            adapters["finance-non-indicator"] = non_indicator_adapter or adapter_path
        if adapters:
            first_name, first_path = next(iter(adapters.items()))
            model = PeftModel.from_pretrained(model, first_path, adapter_name=first_name)
            for name, path in list(adapters.items())[1:]:
                model.load_adapter(path, adapter_name=name)
        self._model = model.eval()
        self._profiles = frozenset({"base", *adapters})
        if default_profile not in self._profiles:
            raise ValueError(f"default profile is unavailable: {default_profile}")
        self._default_profile = default_profile
        self._prefix_tokens = self._tokenizer.encode(PREFIX, add_special_tokens=False)
        self._suffix_tokens = self._tokenizer.encode(SUFFIX, add_special_tokens=False)
        self._token_no = self._tokenizer.convert_tokens_to_ids("no")
        self._token_yes = self._tokenizer.convert_tokens_to_ids("yes")
        self._body_budget = max_length - len(self._prefix_tokens) - len(self._suffix_tokens)
        if self._body_budget < 128:
            raise ValueError("max_length is too small for the official reranker prompt")

    @property
    def profiles(self) -> list[str]:
        return sorted(self._profiles)

    @property
    def default_profile(self) -> str:
        return self._default_profile

    def _resolve_profile(self, requested: str) -> str:
        profile = self._default_profile if requested == "qwen3-reranker-8b-local" else requested
        if profile not in self._profiles:
            raise ValueError(f"unknown or unloaded model profile: {profile}")
        return profile

    def score(
        self,
        query: str,
        documents: list[str],
        instruction: str,
        requested_profile: str = "qwen3-reranker-8b-local",
    ) -> list[float]:
        scores: list[float] = []
        profile = self._resolve_profile(requested_profile)
        with self._lock:
            if profile == "base":
                adapter_context = (
                    self._model.disable_adapter()
                    if hasattr(self._model, "disable_adapter")
                    else nullcontext()
                )
            else:
                self._model.set_adapter(profile)
                adapter_context = nullcontext()
            with self._torch.inference_mode(), adapter_context:
                for start in range(0, len(documents), self._batch_size):
                    items = documents[start : start + self._batch_size]
                    bodies = [format_pair(instruction, query, document) for document in items]
                    encoded = self._tokenizer(
                        bodies,
                        padding=False,
                        truncation=True,
                        max_length=self._body_budget,
                        add_special_tokens=False,
                    )["input_ids"]
                    sequences = [self._prefix_tokens + ids + self._suffix_tokens for ids in encoded]
                    batch = self._tokenizer.pad(
                        {"input_ids": sequences}, padding=True, return_tensors="pt"
                    ).to(self._model.device)
                    logits = self._model(**batch, logits_to_keep=1).logits[:, -1, :][
                        :, [self._token_no, self._token_yes]
                    ].float()
                    scores.extend((logits[:, 1] - logits[:, 0]).cpu().tolist())
        return scores


def create_app(
    model_path: Path,
    adapter_path: Path | None,
    max_length: int = 768,
    batch_size: int = 2,
    *,
    indicator_adapter: Path | None = None,
    non_indicator_adapter: Path | None = None,
    default_profile: str = "base",
) -> FastAPI:
    state: dict[str, LocalQwenReranker] = {}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        state["reranker"] = LocalQwenReranker(
            model_path,
            adapter_path,
            max_length,
            batch_size,
            indicator_adapter=indicator_adapter,
            non_indicator_adapter=non_indicator_adapter,
            default_profile=default_profile,
        )
        yield
        state.clear()

    app = FastAPI(title="NewRAG Local Qwen3 Reranker", lifespan=lifespan)

    @app.get("/livez")
    async def livez() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz() -> dict[str, Any]:
        if "reranker" not in state:
            raise HTTPException(status_code=503, detail="model is loading")
        reranker = state["reranker"]
        return {
            "status": "ok",
            "model": "qwen3-reranker-8b-local",
            "default_profile": reranker.default_profile,
            "profiles": reranker.profiles,
        }

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return await readyz()

    @app.post("/api/v1/services/rerank/text-rerank/text-rerank")
    async def rerank(request: RerankRequest) -> dict[str, Any]:
        reranker = state.get("reranker")
        if reranker is None:
            raise HTTPException(status_code=503, detail="model is loading")
        try:
            scores = await asyncio.to_thread(
                reranker.score,
                request.input.query,
                request.input.documents,
                request.parameters.instruction,
                request.model,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "output": {
                "results": format_results(scores, request.input.documents, request.parameters.top_n)
            },
            "usage": {"documents": len(request.input.documents)},
        }

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--indicator-adapter", type=Path)
    parser.add_argument("--non-indicator-adapter", type=Path)
    parser.add_argument(
        "--default-profile",
        choices=("base", "finance-indicator", "finance-non-indicator"),
        default="base",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import uvicorn

    uvicorn.run(
        create_app(
            args.model.resolve(),
            args.adapter.resolve() if args.adapter else None,
            args.max_length,
            args.batch_size,
            indicator_adapter=(
                args.indicator_adapter.resolve() if args.indicator_adapter else None
            ),
            non_indicator_adapter=(
                args.non_indicator_adapter.resolve() if args.non_indicator_adapter else None
            ),
            default_profile=args.default_profile,
        ),
        host=args.host,
        port=args.port,
        workers=1,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
