from time import time
from typing import AsyncIterator

from msgspec import Struct

from fishr.Http import make_client
from fishr.Loop import asyncio
from fishr.Utils import aiter_lines, json_encode

CHAT_URL = "https://api.binjie.fun/api/generateStream"

YQCLOUD_MODELS = [
    "gpt-4",
]

DEFAULT_MODEL = "gpt-4"

HEADERS = {
    "accept": "application/json, text/plain, */*",
    "accept-language": "en-US,en;q=0.9",
    "content-type": "application/json",
    "origin": "https://chat9.yqcloud.top",
    "referer": "https://chat9.yqcloud.top/",
    "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
}


def resolve_model(model: str) -> str:
    base = model.split("/", 1)[-1] if "/" in model else model
    if base in YQCLOUD_MODELS:
        return base
    return DEFAULT_MODEL


def _gen_user_id() -> str:
    return f"#/chat/{int(time() * 1000)}"


def _format_prompt(messages: list[dict]) -> str:
    parts: list[str] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            parts.append(f"System: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
        else:
            parts.append(f"User: {content}")
    return "\n\n".join(parts)


def _extract_system(messages: list[dict]) -> tuple[str, list[dict]]:
    if messages and messages[0].get("role") == "system":
        return messages[0]["content"], messages[1:]
    return "", messages


def _build_payload(
    prompt: str,
    user_id: str,
    system: str,
    stream: bool,
) -> bytes:
    payload = {
        "prompt": prompt,
        "userId": user_id,
        "network": True,
        "system": system,
        "withoutContext": False,
        "stream": stream,
    }
    return json_encode.encode(payload)


class YqcloudResponse(Struct, frozen=True):
    content: str
    model: str


class YqcloudStream:
    __slots__ = ("resp", "model")

    def __init__(self, resp, model: str) -> None:
        self.resp = resp
        self.model = model

    def __iter__(self):
        for line in self.resp.iter_lines():
            text = line.decode(errors="ignore") if isinstance(line, bytes) else line
            if text:
                yield text
        self.resp.close()

    async def __aiter__(self) -> AsyncIterator[str]:
        async for line in aiter_lines(self.resp):
            if line:
                yield line


class Yqcloud:
    __slots__ = ("http_client", "stream_client", "user_id")

    def __init__(self, model: str = "yqcloud/gpt-4") -> None:
        self.http_client = make_client(headers=HEADERS)
        self.stream_client = make_client(headers=HEADERS)
        self.user_id = _gen_user_id()

    def ask(
        self,
        prompt: str,
        *,
        model: str = "yqcloud/gpt-4",
        stream: bool = False,
    ) -> YqcloudResponse | YqcloudStream:
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, model=model, stream=stream)

    async def ask_async(
        self,
        prompt: str,
        *,
        model: str = "yqcloud/gpt-4",
        stream: bool = False,
    ) -> YqcloudResponse | YqcloudStream:
        return await asyncio.to_thread(
            self.ask,
            prompt,
            model=model,
            stream=stream,
        )

    def chat(
        self,
        messages: list[dict],
        *,
        model: str = "yqcloud/gpt-4",
        stream: bool = False,
    ) -> YqcloudResponse | YqcloudStream:
        resolved = resolve_model(model)
        system, current = _extract_system(messages)
        prompt = _format_prompt(current)
        body = _build_payload(prompt, self.user_id, system, stream)

        if stream:
            resp = self.stream_client.post(
                CHAT_URL,
                content=body,
                stream=True,
            )
            return YqcloudStream(resp, resolved)

        resp = self.http_client.post(CHAT_URL, content=body)
        content = resp.text.strip() if resp.text else ""
        return YqcloudResponse(content=content, model=resolved)

    async def chat_async(
        self,
        messages: list[dict],
        *,
        model: str = "yqcloud/gpt-4",
        stream: bool = False,
    ) -> YqcloudResponse | YqcloudStream:
        return await asyncio.to_thread(
            self.chat,
            messages,
            model=model,
            stream=stream,
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
    "Yqcloud",
    "YqcloudResponse",
    "YqcloudStream",
    "resolve_model",
]
