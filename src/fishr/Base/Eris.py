from typing import AsyncIterator

from msgspec import Struct

from fishr.Http import make_client
from fishr.Loop import asyncio
from fishr.Utils import aiter_lines, json_encode

CHAT_URL = "https://freeai.help/api/chat/completions"

ERIS_MODELS = {
    "eris/deepseek-v4-flash": "deepseek-ai/deepseek-v4-flash",
    "eris/deepseek-v4-pro": "deepseek-ai/deepseek-v4-pro",
    "eris/glm-5.1": "z-ai/glm-5.1",
    "eris/minimax-m3": "minimaxai/minimax-m3",
    "eris/kimi-k2.6": "moonshotai/kimi-k2.6",
}

DEFAULT_MODEL = "eris/deepseek-v4-flash"

HEADERS = {
    "accept": "application/json, text/plain, */*",
    "accept-language": "en-US,en;q=0.9",
    "content-type": "application/json",
    "origin": "https://freeai.help",
    "referer": "https://freeai.help/en/chat/deepseek",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36",
}


def resolve_model(model: str) -> str:
    base = model.split("/", 1)[-1] if "/" in model else model
    key = f"eris/{base}"
    if key in ERIS_MODELS:
        return key
    return DEFAULT_MODEL


def _build_payload(
    messages: list[dict],
    model: str,
    stream: bool,
    enable_thinking: bool = False,
) -> bytes:
    api_model = ERIS_MODELS.get(model, ERIS_MODELS[DEFAULT_MODEL])
    payload = {
        "model": api_model,
        "messages": messages,
        "enableThinking": enable_thinking,
    }
    return json_encode.encode(payload)


def _parse_sse_line(line: str) -> tuple[str, str, bool] | None:
    """Parse a single SSE data line. Returns (content, reasoning, done) or None."""
    if not line.startswith("data:"):
        return None
    data = line[5:].strip()
    if data == "[DONE]":
        return ("", "", True)
    try:
        import json as _json

        obj = _json.loads(data)
        return (
            obj.get("content", ""),
            obj.get("reasoning", ""),
            obj.get("done", False),
        )
    except Exception:
        return None


class ErisResponse(Struct, frozen=True):
    content: str
    model: str


class ErisStream:
    __slots__ = ("resp", "model")

    def __init__(self, resp, model: str) -> None:
        self.resp = resp
        self.model = model

    def __iter__(self):
        for line in self.resp.iter_lines():
            text = line.decode(errors="ignore") if isinstance(line, bytes) else line
            parsed = _parse_sse_line(text)
            if parsed is None:
                continue
            content, reasoning, done = parsed
            if done:
                break
            if reasoning:
                yield (reasoning, True)
            if content:
                yield (content, False)
        self.resp.close()

    async def __aiter__(self) -> AsyncIterator[tuple[str, bool]]:
        async for line in aiter_lines(self.resp):
            parsed = _parse_sse_line(line)
            if parsed is None:
                continue
            content, reasoning, done = parsed
            if done:
                break
            if reasoning:
                yield (reasoning, True)
            if content:
                yield (content, False)


class Eris:
    __slots__ = ("http_client", "stream_client")

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self.http_client = make_client(headers=HEADERS)
        self.stream_client = make_client(headers=HEADERS)

    def ask(
        self,
        prompt: str,
        *,
        model: str = DEFAULT_MODEL,
        stream: bool = False,
        enable_thinking: bool = False,
    ) -> ErisResponse | ErisStream:
        messages = [{"role": "user", "content": prompt}]
        return self.chat(
            messages, model=model, stream=stream, enable_thinking=enable_thinking
        )

    async def ask_async(
        self,
        prompt: str,
        *,
        model: str = DEFAULT_MODEL,
        stream: bool = False,
        enable_thinking: bool = False,
    ) -> ErisResponse | ErisStream:
        return await asyncio.to_thread(
            self.ask,
            prompt,
            model=model,
            stream=stream,
            enable_thinking=enable_thinking,
        )

    def chat(
        self,
        messages: list[dict],
        *,
        model: str = DEFAULT_MODEL,
        stream: bool = False,
        enable_thinking: bool = False,
    ) -> ErisResponse | ErisStream:
        resolved = resolve_model(model)
        body = _build_payload(messages, resolved, stream, enable_thinking)

        if stream:
            resp = self.stream_client.post(
                CHAT_URL,
                content=body,
                stream=True,
                timeout=600,
                read_timeout=600,
            )
            return ErisStream(resp, resolved)

        resp = self.http_client.post(
            CHAT_URL, content=body, timeout=600, read_timeout=600
        )
        content = ""
        reasoning = ""
        for line in resp.iter_lines():
            text = line.decode(errors="ignore") if isinstance(line, bytes) else line
            parsed = _parse_sse_line(text)
            if parsed is None:
                continue
            c, r, done = parsed
            content += c
            reasoning += r
            if done:
                break
        if reasoning:
            content = reasoning + "\n" + content
        return ErisResponse(content=content, model=resolved)

    async def chat_async(
        self,
        messages: list[dict],
        *,
        model: str = DEFAULT_MODEL,
        stream: bool = False,
        enable_thinking: bool = False,
    ) -> ErisResponse | ErisStream:
        return await asyncio.to_thread(
            self.chat,
            messages,
            model=model,
            stream=stream,
            enable_thinking=enable_thinking,
        )

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args) -> None:
        pass


__all__ = [
    "Eris",
    "ErisResponse",
    "ErisStream",
    "resolve_model",
]
