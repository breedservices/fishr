from uuid import uuid4

from msgspec import Struct

from fishr.Http import make_client
from fishr.Loop import asyncio
from fishr.Utils import json_decode, json_encode

CHAT_URL = "https://quillbot.com/api/ai-chat/chat/conversation/{}"

HEADERS = {
    "accept": "text/event-stream",
    "content-type": "application/json",
    "origin": "https://quillbot.com",
    "platform-type": "webapp",
    "qb-product": "AI-CHAT",
    "useridtoken": "empty-token",
    "webapp-version": "42.61.1",
}

QUILLBOT_MODELS = [
    "quillbot",
    "quillbot-search",
]

DEFAULT_MODEL = "quillbot"


def resolve_model(model: str) -> str:
    base = model.split("/", 1)[-1] if "/" in model else model
    if base in QUILLBOT_MODELS:
        return base
    return DEFAULT_MODEL


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


def _build_headers(conversation_id: str) -> dict[str, str]:
    return {
        **HEADERS,
        "referer": f"https://quillbot.com/ai-chat/c/{conversation_id}",
    }


def _build_payload(
    prompt: str,
    model: str,
    web_search: bool,
) -> dict:
    payload = {
        "message": {"content": prompt},
        "context": {
            "editorContext": "",
            "selectionContext": "",
            "userDialect": "en-us",
            "apiVersion": 2,
        },
        "origin": {
            "name": "ai-chat.chat",
            "url": "https://quillbot.com",
        },
    }
    if model == "quillbot-search" or web_search:
        payload["tools"] = {"web_search_builtin": {}}
    return payload


def _parse_response(text: str) -> str:
    out: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            data = json_decode.decode(line)
            if isinstance(data, dict) and data.get("type") == "content":
                content = data.get("content", "")
                if content:
                    out.append(content)
        except Exception:
            pass
    return "".join(out)


class QuillbotResponse(Struct, frozen=True):
    content: str
    model: str


class QuillbotStream:
    __slots__ = ("resp", "model")

    def __init__(self, resp, model: str) -> None:
        self.resp = resp
        self.model = model

    def __iter__(self):
        for line in self.resp.iter_lines():
            text = line.decode(errors="ignore") if isinstance(line, bytes) else line
            text = text.strip()
            if not text:
                continue
            try:
                data = json_decode.decode(text)
                if isinstance(data, dict) and data.get("type") == "content":
                    content = data.get("content", "")
                    if content:
                        yield content
            except Exception:
                pass
        self.resp.close()

    async def __aiter__(self):
        from fishr.Utils import aiter_lines

        async for line in aiter_lines(self.resp):
            line = line.strip()
            if not line:
                continue
            try:
                data = json_decode.decode(line)
                if isinstance(data, dict) and data.get("type") == "content":
                    content = data.get("content", "")
                    if content:
                        yield content
            except Exception:
                pass


class Quillbot:
    __slots__ = ("http_client", "stream_client")

    def __init__(self, model: str = "quillbot/quillbot") -> None:
        self.http_client = make_client(headers=HEADERS)
        self.stream_client = make_client(headers=HEADERS)

    def ask(
        self,
        prompt: str,
        *,
        model: str = "quillbot/quillbot",
        web_search: bool = False,
        stream: bool = False,
    ) -> QuillbotResponse | QuillbotStream:
        resolved = resolve_model(model)
        conversation_id = str(uuid4())
        url = CHAT_URL.format(conversation_id)
        formatted = _format_prompt([{"role": "user", "content": prompt}])
        headers = _build_headers(conversation_id)
        payload = _build_payload(formatted, resolved, web_search)

        if stream:
            resp = self.stream_client.post(
                url,
                json=payload,
                headers=headers,
                stream=True,
            )
            return QuillbotStream(resp, resolved)

        resp = self.http_client.post(url, json=payload, headers=headers)
        content = _parse_response(resp.text)
        return QuillbotResponse(content=content, model=resolved)

    async def ask_async(
        self,
        prompt: str,
        *,
        model: str = "quillbot/quillbot",
        web_search: bool = False,
        stream: bool = False,
    ) -> QuillbotResponse | QuillbotStream:
        return await asyncio.to_thread(
            self.ask,
            prompt,
            model=model,
            web_search=web_search,
            stream=stream,
        )

    def chat(
        self,
        messages: list[dict],
        *,
        model: str = "quillbot/quillbot",
        web_search: bool = False,
        stream: bool = False,
    ) -> QuillbotResponse | QuillbotStream:
        resolved = resolve_model(model)
        conversation_id = str(uuid4())
        url = CHAT_URL.format(conversation_id)
        formatted = _format_prompt(messages)
        headers = _build_headers(conversation_id)
        payload = _build_payload(formatted, resolved, web_search)

        if stream:
            resp = self.stream_client.post(
                url,
                json=payload,
                headers=headers,
                stream=True,
            )
            return QuillbotStream(resp, resolved)

        resp = self.http_client.post(url, json=payload, headers=headers)
        content = _parse_response(resp.text)
        return QuillbotResponse(content=content, model=resolved)

    async def chat_async(
        self,
        messages: list[dict],
        *,
        model: str = "quillbot/quillbot",
        web_search: bool = False,
        stream: bool = False,
    ) -> QuillbotResponse | QuillbotStream:
        return await asyncio.to_thread(
            self.chat,
            messages,
            model=model,
            web_search=web_search,
            stream=stream,
        )


__all__ = [
    "Quillbot",
    "QuillbotResponse",
    "QuillbotStream",
    "resolve_model",
]
