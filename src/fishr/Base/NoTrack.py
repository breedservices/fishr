from uuid import uuid4

from msgspec import Struct

from fishr.Http import make_client
from fishr.Loop import asyncio
from fishr.Utils import aiter_lines, json_decode, json_encode

DISPATCH_URL = "https://notrack.ai/api/dispatch"
CHATS_URL = "https://notrack.ai/api/chats"

DATA_PREFIX = "data: "

MODELS = {
    "notrack/fast": "C",
    "notrack/standard": "B",
    "notrack/reasoning": "A",
}

MODES = ("usual",)

HEADERS = {
    "content-type": "application/json",
    "origin": "https://notrack.ai",
    "referer": "https://notrack.ai/chat",
}


def resolve_model(model: str) -> str:
    base = model.split("/", 1)[-1] if "/" in model else model
    if base in MODELS:
        return base
    if base in ("fast", "standard", "reasoning"):
        return base
    return "fast"


def _model_code(model: str) -> str:
    resolved = resolve_model(model)
    return MODELS.get(f"notrack/{resolved}", "C")


def _gen_uid() -> str:
    return f"uid={uuid4()}"


def _parse_line(line: str) -> str:
    if not line.startswith(DATA_PREFIX):
        return ""
    chunk = line[len(DATA_PREFIX) :]
    try:
        data = json_decode.decode(chunk)
        if isinstance(data, dict):
            if data.get("type") == "done":
                return ""
            if data.get("type") == "delta":
                return data.get("chunk", "")
    except Exception:
        pass
    return ""


def _collect_sse(resp) -> str:
    out: list[str] = []
    for line in resp.iter_lines():
        text = line.decode(errors="ignore") if isinstance(line, bytes) else line
        content = _parse_line(text)
        if content:
            out.append(content)
    resp.close()
    return "".join(out)


def _build_payload(
    user_input: str,
    model: str,
    chat_id: str | None,
) -> bytes:
    payload = {
        "user_input": user_input,
        "model": _model_code(model),
        "mode": "usual",
        "max_turns": 6,
        "chat_id": chat_id,
    }
    return json_encode.encode(payload)


class NoTrackResponse(Struct, frozen=True):
    content: str
    model: str
    chat_id: str | None = None


class NoTrackStream:
    __slots__ = ("resp", "model", "chat_id")

    def __init__(self, resp, model: str, chat_id: str | None = None) -> None:
        self.resp = resp
        self.model = model
        self.chat_id = chat_id

    def __iter__(self):
        extracted_id = self.chat_id
        for line in self.resp.iter_lines():
            text = line.decode(errors="ignore") if isinstance(line, bytes) else line
            text = text.strip()
            if not text:
                continue
            if text.startswith(DATA_PREFIX):
                chunk = text[len(DATA_PREFIX) :]
                try:
                    data = json_decode.decode(chunk)
                    if isinstance(data, dict):
                        if data.get("type") == "chat_meta" and data.get("chat_id"):
                            extracted_id = data["chat_id"]
                        if data.get("type") == "delta":
                            content = data.get("chunk", "")
                            if content:
                                yield content
                except Exception:
                    pass
        self.chat_id = extracted_id
        self.resp.close()

    async def __aiter__(self):
        extracted_id = self.chat_id
        async for line in aiter_lines(self.resp):
            line = line.strip()
            if not line:
                continue
            if line.startswith(DATA_PREFIX):
                chunk = line[len(DATA_PREFIX) :]
                try:
                    data = json_decode.decode(chunk)
                    if isinstance(data, dict):
                        if data.get("type") == "chat_meta" and data.get("chat_id"):
                            extracted_id = data["chat_id"]
                        if data.get("type") == "delta":
                            content = data.get("chunk", "")
                            if content:
                                yield content
                except Exception:
                    pass
        self.chat_id = extracted_id


class NoTrack:
    __slots__ = (
        "http_client",
        "stream_client",
        "chat_id",
    )

    def __init__(self, model: str = "notrack/fast") -> None:
        cookie = _gen_uid()
        headers = {**HEADERS, "cookie": cookie}
        self.http_client = make_client(headers=headers)
        self.stream_client = make_client(headers=headers)
        self.chat_id: str | None = None

    def ask(
        self,
        prompt: str,
        *,
        model: str = "notrack/fast",
        web_search: bool = False,
        stream: bool = False,
    ) -> NoTrackResponse | NoTrackStream:
        messages = [{"role": "user", "content": prompt}]
        return self.chat(
            messages,
            model=model,
            web_search=web_search,
            stream=stream,
        )

    async def ask_async(
        self,
        prompt: str,
        *,
        model: str = "notrack/fast",
        web_search: bool = False,
        stream: bool = False,
    ) -> NoTrackResponse | NoTrackStream:
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
        model: str = "notrack/fast",
        web_search: bool = False,
        stream: bool = False,
    ) -> NoTrackResponse | NoTrackStream:
        resolved = resolve_model(model)
        user_input = messages[-1]["content"] if messages else ""
        body = _build_payload(user_input, resolved, self.chat_id)

        if stream:
            resp = self.stream_client.post(
                DISPATCH_URL,
                content=body,
                stream=True,
            )
            return NoTrackStream(resp, resolved, self.chat_id)

        resp = self.http_client.post(DISPATCH_URL, content=body)
        content = _collect_sse(resp)
        return NoTrackResponse(
            content=content,
            model=resolved,
            chat_id=self.chat_id,
        )

    async def chat_async(
        self,
        messages: list[dict],
        *,
        model: str = "notrack/fast",
        web_search: bool = False,
        stream: bool = False,
    ) -> NoTrackResponse | NoTrackStream:
        return await asyncio.to_thread(
            self.chat,
            messages,
            model=model,
            web_search=web_search,
            stream=stream,
        )

    def new_chat(self) -> None:
        self.chat_id = None

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args) -> None:
        pass


__all__ = [
    "NoTrack",
    "NoTrackResponse",
    "NoTrackStream",
    "resolve_model",
]
