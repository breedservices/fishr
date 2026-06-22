from typing import AsyncIterator

from msgspec import Struct

from fishr.Http import make_client
from fishr.Loop import asyncio
from fishr.Utils import aiter_lines, json_decode, json_encode

CHAT_URL = "https://chat.dphn.ai/api/chat"

DATA_PREFIX = "data: "

MODELS = {
    "dphnai/24b": "dolphinserver:24B",
    "dphnai/6b": "dolphinserver2:6b",
}

DEFAULT_MODEL = "dphnai/6b"

TEMPLATES = (
    "logical",
    "creative",
    "summarize",
    "code-beginner",
    "code-advanced",
)

HEADERS = {
    "accept": "text/event-stream",
    "content-type": "application/json",
    "origin": "https://chat.dphn.ai",
    "referer": "https://chat.dphn.ai/",
}


def resolve_model(model: str) -> str:
    base = model.split("/", 1)[-1] if "/" in model else model
    if f"dphnai/{base}" in MODELS:
        return f"dphnai/{base}"
    if base in MODELS:
        return base
    return DEFAULT_MODEL


def _api_model(resolved: str) -> str:
    return MODELS.get(resolved, MODELS[DEFAULT_MODEL])


def _parse_chunk(line: str) -> str:
    if not line.startswith(DATA_PREFIX):
        return ""
    chunk = line[len(DATA_PREFIX) :]
    if chunk.strip() == "[DONE]":
        return ""
    try:
        data = json_decode.decode(chunk)
        if isinstance(data, dict):
            choices = data.get("choices")
            if isinstance(choices, list) and choices:
                delta = choices[0].get("delta")
                if isinstance(delta, dict):
                    return delta.get("content", "")
    except Exception:
        pass
    return ""


def _collect_sse(resp) -> str:
    out: list[str] = []
    for line in resp.iter_lines():
        text = line.decode(errors="ignore") if isinstance(line, bytes) else line
        content = _parse_chunk(text.strip())
        if content:
            out.append(content)
    resp.close()
    return "".join(out)


def _build_payload(messages: list[dict], model: str, template: str | None) -> bytes:
    payload = {
        "messages": messages,
        "model": _api_model(model),
    }
    if template:
        payload["template"] = template
    return json_encode.encode(payload)


class DphnAIResponse(Struct, frozen=True):
    content: str
    model: str


class DphnAIStream:
    __slots__ = ("resp", "model")

    def __init__(self, resp, model: str) -> None:
        self.resp = resp
        self.model = model

    def __iter__(self):
        for line in self.resp.iter_lines():
            text = line.decode(errors="ignore") if isinstance(line, bytes) else line
            content = _parse_chunk(text.strip())
            if content:
                yield content
        self.resp.close()

    async def __aiter__(self) -> AsyncIterator[str]:
        async for line in aiter_lines(self.resp):
            content = _parse_chunk(line.strip())
            if content:
                yield content


class DphnAI:
    __slots__ = ("http_client", "stream_client")

    def __init__(self) -> None:
        self.http_client = make_client(headers=HEADERS)
        self.stream_client = make_client(headers=HEADERS)

    def ask(
        self,
        prompt: str,
        *,
        model: str = DEFAULT_MODEL,
        stream: bool = False,
        template: str | None = None,
    ) -> DphnAIResponse | DphnAIStream:
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, model=model, stream=stream, template=template)

    async def ask_async(
        self,
        prompt: str,
        *,
        model: str = DEFAULT_MODEL,
        stream: bool = False,
        template: str | None = None,
    ) -> DphnAIResponse | DphnAIStream:
        return await asyncio.to_thread(
            self.ask,
            prompt,
            model=model,
            stream=stream,
            template=template,
        )

    def chat(
        self,
        messages: list[dict],
        *,
        model: str = DEFAULT_MODEL,
        stream: bool = False,
        template: str | None = None,
    ) -> DphnAIResponse | DphnAIStream:
        resolved = resolve_model(model)
        body = _build_payload(messages, resolved, template)

        if stream:
            resp = self.stream_client.post(
                CHAT_URL,
                content=body,
                stream=True,
            )
            return DphnAIStream(resp, resolved)

        resp = self.http_client.post(CHAT_URL, content=body)
        content = _collect_sse(resp)
        return DphnAIResponse(content=content, model=resolved)

    async def chat_async(
        self,
        messages: list[dict],
        *,
        model: str = DEFAULT_MODEL,
        stream: bool = False,
        template: str | None = None,
    ) -> DphnAIResponse | DphnAIStream:
        return await asyncio.to_thread(
            self.chat,
            messages,
            model=model,
            stream=stream,
            template=template,
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
    "DphnAI",
    "DphnAIResponse",
    "DphnAIStream",
    "resolve_model",
]
