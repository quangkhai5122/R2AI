"""OpenAI-compatible HTTP server for the Kaggle-hosted Qwen model.

The server keeps one transformers model in GPU memory and exposes:

    GET  /health
    GET  /v1/models
    POST /v1/chat/completions

It is intentionally single-flight: one T4 should not receive concurrent
transformers.generate calls. The web backend already speaks the OpenAI chat
completions protocol, so no frontend code is needed for the bridge.
"""
from __future__ import annotations

import argparse
import atexit
import hmac
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Literal

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=200_000)


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str | None = None
    messages: list[ChatMessage] = Field(min_length=1, max_length=64)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=256, ge=1, le=4096)
    stream: bool = False


@dataclass(frozen=True)
class ServerConfig:
    model: str
    api_key: str
    max_tokens: int


class ModelRuntime:
    """Lazy-load and serialize access to the Hugging Face model."""

    def __init__(self, config: ServerConfig, *, load_4bit: bool,
                 max_input_tokens: int, batch_size: int = 1) -> None:
        self.config = config
        self.load_4bit = load_4bit
        self.max_input_tokens = max_input_tokens
        self.batch_size = max(1, batch_size)
        self._client = None
        self._load_lock = threading.Lock()
        self._generate_lock = threading.Lock()

    def _get_client(self):
        if self._client is None:
            with self._load_lock:
                if self._client is None:
                    # Heavy imports happen only when the server starts. This
                    # keeps --help and contract tests usable without CUDA.
                    from vifinqa.codegen.llm_client import HfBatchClient

                    self._client = HfBatchClient(
                        model=self.config.model,
                        load_4bit=self.load_4bit,
                        batch_size=self.batch_size,
                        max_input_tokens=self.max_input_tokens,
                    )
        return self._client

    def generate(self, messages: list[dict[str, str]], *, temperature: float,
                 max_tokens: int) -> str:
        client = self._get_client()
        bounded_tokens = min(max(1, max_tokens), self.config.max_tokens)
        # transformers.generate is not safe to overlap on a single T4 model.
        with self._generate_lock:
            outputs = client.chat_batch(
                [messages], n=1, temperature=temperature,
                max_tokens=bounded_tokens,
            )
        return outputs[0][0] if outputs and outputs[0] else ""


runtime: ModelRuntime | None = None
server_config: ServerConfig | None = None
ngrok_tunnel = None


def _auth_ok(authorization: str | None) -> bool:
    """Allow local unauthenticated use; require Bearer for a public tunnel."""
    expected = server_config.api_key if server_config else ""
    if not expected:
        return True
    if not authorization or not authorization.startswith("Bearer "):
        return False
    supplied = authorization.removeprefix("Bearer ").strip()
    return hmac.compare_digest(supplied, expected)


def _require_auth(authorization: str | None) -> None:
    if not _auth_ok(authorization):
        raise HTTPException(status_code=401, detail="Invalid or missing Bearer token")


def _completion_response(model: str, content: str, started: float) -> dict:
    now = int(time.time())
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": now,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {
            # Token accounting is provider-specific and not needed by the web
            # adapter. Keep valid OpenAI-compatible integer fields.
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
        "x_generation_ms": int((time.perf_counter() - started) * 1000),
    }


def create_app(config: ServerConfig, *, load_4bit: bool,
               max_input_tokens: int, batch_size: int = 1) -> FastAPI:
    global runtime, server_config
    server_config = config
    runtime = ModelRuntime(
        config, load_4bit=load_4bit,
        max_input_tokens=max_input_tokens,
        batch_size=batch_size,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # Load before accepting requests, so the first web request does not
        # unexpectedly spend several minutes downloading/loading the model.
        await _load_runtime()
        yield

    app = FastAPI(title="Qwen Kaggle Model Server", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "model": config.model,
            "loaded": runtime is not None and runtime._client is not None,
            "max_tokens": config.max_tokens,
        }

    @app.get("/v1/models")
    def models(authorization: str | None = Header(default=None)) -> dict:
        _require_auth(authorization)
        return {
            "object": "list",
            "data": [{
                "id": config.model,
                "object": "model",
                "owned_by": "kaggle",
            }],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(
        payload: ChatCompletionRequest,
        authorization: str | None = Header(default=None),
    ) -> dict:
        _require_auth(authorization)
        if payload.stream:
            raise HTTPException(
                status_code=400,
                detail="stream=true is not supported; use stream=false",
            )
        requested_model = payload.model or config.model
        if requested_model != config.model:
            raise HTTPException(
                status_code=404,
                detail=f"Model not served: {requested_model}",
            )
        if runtime is None:
            raise HTTPException(status_code=503, detail="Model runtime unavailable")
        started = time.perf_counter()
        try:
            content = runtime.generate(
                [message.model_dump() for message in payload.messages],
                temperature=payload.temperature,
                max_tokens=payload.max_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            # Do not return internal stack traces or credentials to the client.
            print(f"[model-server] generation_error {type(exc).__name__}: {exc}", flush=True)
            raise HTTPException(status_code=500, detail="Model generation failed") from exc
        if not content.strip():
            raise HTTPException(status_code=502, detail="Model returned empty content")
        return _completion_response(config.model, content, started)

    return app


async def _load_runtime() -> None:
    """Run blocking model construction away from the event loop."""
    import asyncio

    if runtime is not None:
        await asyncio.to_thread(runtime._get_client)


def _stop_ngrok(tunnel) -> None:
    if tunnel is None:
        return
    try:
        from pyngrok import ngrok
        ngrok.disconnect(tunnel.public_url)
    except Exception as exc:  # noqa: BLE001
        print(f"[model-server] tunnel_cleanup_error {type(exc).__name__}: {exc}", flush=True)


def _start_ngrok(port: int, auth_token: str):
    if not auth_token:
        raise SystemExit(
            "NGROK_AUTHTOKEN is required when --public-url ngrok is used"
        )
    from pyngrok import conf, ngrok

    conf.get_default().auth_token = auth_token
    tunnel = ngrok.connect(addr=port, proto="http")
    print(f"PUBLIC_BASE_URL={tunnel.public_url}/v1", flush=True)
    print("PUBLIC_HEALTH_URL=" + tunnel.public_url + "/health", flush=True)
    print("Copy the base URL into FINQUERY_LLM_BASE_URL; keep /v1.", flush=True)
    return tunnel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.getenv(
        "MODEL_SERVER_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct"))
    parser.add_argument("--host", default=os.getenv("MODEL_SERVER_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("MODEL_SERVER_PORT", "8001")))
    parser.add_argument("--api-key", default=os.getenv("MODEL_SERVER_API_KEY", ""),
                        help="Bearer token; required for --public-url ngrok")
    parser.add_argument("--max-tokens", type=int, default=int(os.getenv(
        "MODEL_SERVER_MAX_TOKENS", "512")))
    parser.add_argument("--max-input-tokens", type=int, default=int(os.getenv(
        "MODEL_SERVER_MAX_INPUT_TOKENS", "5000")))
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Keep at 1 for interactive single-request serving")
    parser.add_argument("--load-4bit", action="store_true",
                        help="Use bitsandbytes NF4; recommended on Kaggle T4")
    parser.add_argument("--public-url", choices=["none", "ngrok"], default="none")
    parser.add_argument("--log-level", default="info")
    return parser.parse_args()


def main() -> None:
    global ngrok_tunnel
    args = parse_args()
    if args.public_url == "ngrok" and not args.api_key:
        raise SystemExit("Set MODEL_SERVER_API_KEY before exposing the server publicly")

    config = ServerConfig(
        model=args.model,
        api_key=args.api_key,
        max_tokens=max(1, min(args.max_tokens, 4096)),
    )
    app = create_app(
        config,
        load_4bit=args.load_4bit,
        max_input_tokens=max(256, args.max_input_tokens),
        batch_size=max(1, args.batch_size),
    )

    if args.public_url == "ngrok":
        ngrok_tunnel = _start_ngrok(args.port, os.getenv("NGROK_AUTHTOKEN", ""))
        atexit.register(lambda: _stop_ngrok(ngrok_tunnel))

    print(f"MODEL={args.model}", flush=True)
    print(f"LOCAL_BASE_URL=http://127.0.0.1:{args.port}/v1", flush=True)
    print("MODEL_SERVER_READY", flush=True)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
