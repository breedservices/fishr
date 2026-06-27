import logging
from typing import AsyncIterator

from msgspec import Struct

from fishr.Http import make_client
from fishr.Loop import asyncio
from fishr.Utils import aiter_lines, json_decode, json_encode

Log = logging.getLogger("fishr.telnyx")
CHAT_URL = "https://telnyx.com/api/inference"

TELNYX_MODELS = {
    "telnyx/glm-5.2": "zai-org/GLM-5.2",
    "telnyx/glm-5.1": "zai-org/GLM-5.1-FP8",
    "telnyx/kimi-k2.6": "moonshotai/Kimi-K2.6",
    "telnyx/minimax-m3": "MiniMaxAI/MiniMax-M3-MXFP8",
}
DEFAULT_MODEL = "telnyx/glm-5.1"

HEADERS = {
    "accept": "text/event-stream",
    "content-type": "application/json",
    "origin": "https://telnyx.com",
    "referer": "https://telnyx.com/",
}


def resolve_model(model: str) -> str:
    base = model.split("/", 1)[-1] if "/" in model else model
    key = f"telnyx/{base}"
    if key in TELNYX_MODELS:
        return key
    return DEFAULT_MODEL


def _parse_chunk(obj: dict) -> tuple[str, str, bool]:
    """Parse an OpenAI-style chat.completion.chunk into (content, thinking, done)."""
    Done = False
    Content = ""
    Thinking = ""
    Choices = obj.get("choices") or []
    if Choices:
        Choice = Choices[0]
        if Choice.get("finish_reason"):
            Done = True
        Delta = Choice.get("delta") or {}
        if isinstance(Delta, dict):
            Content = Delta.get("content") or ""
            Thinking = Delta.get("reasoning_content") or ""
    return Content, Thinking, Done


def _parse_line(line: str) -> tuple[str, str, bool]:
    if not line.startswith("data:"):
        return "", "", False
    Data = line[5:].strip()
    if Data in ("[DONE]", "null", ""):
        return "", "", True
    try:
        Obj = json_decode.decode(Data)
    except Exception:
        return "", "", False
    return _parse_chunk(Obj)


class TelnyxResponse(Struct, frozen=True):
    content: str
    model: str
    thinking: str = ""


class TelnyxStream:
    __slots__ = ("Resp", "Model")

    def __init__(self, Resp, Model: str) -> None:
        self.Resp = Resp
        self.Model = Model

    def __iter__(self):
        for line in self.Resp.iter_lines():
            Text = line.decode(errors="ignore") if isinstance(line, bytes) else line
            Content, Thinking, Done = _parse_line(Text)
            if Thinking:
                yield (Thinking, True)
            if Content:
                yield (Content, False)
            if Done:
                break
        self.Resp.close()

    async def __aiter__(self) -> AsyncIterator[tuple[str, bool]]:
        async for line in aiter_lines(self.Resp):
            Content, Thinking, Done = _parse_line(line)
            if Thinking:
                yield (Thinking, True)
            if Content:
                yield (Content, False)
            if Done:
                break


class _EmptyResp:
    """Minimal stand-in for a primp response when the upstream fails."""

    def iter_lines(self):
        if False:
            yield b""

    def close(self) -> None:
        pass


class Telnyx:
    """
    Usage::

        ```py
        from fishr import Telnyx

        tx = Telnyx()
        result = tx.ask("Hello!")
        print(result.content)
        ```
    """

    __slots__ = ("HttpClient", "StreamClient")

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self.HttpClient = make_client(headers=HEADERS)
        self.StreamClient = make_client(headers=HEADERS)

    def _build_payload(
        self,
        Messages: list[dict],
        Model: str,
        Stream: bool,
        EnableThinking: bool = False,
    ) -> bytes:
        ApiModel = TELNYX_MODELS.get(Model, TELNYX_MODELS[DEFAULT_MODEL])
        Payload = {
            "model": ApiModel,
            "messages": Messages,
            "temperature": 0.7,
            "max_tokens": 16384,
            "stream": Stream,
            "stream_options": {"include_usage": True},
            "enable_thinking": EnableThinking,
        }
        return json_encode.encode(Payload)

    def ask(
        self,
        prompt: str,
        *,
        model: str = DEFAULT_MODEL,
        stream: bool = False,
        enable_thinking: bool = False,
    ) -> TelnyxResponse | TelnyxStream:
        Messages = [{"role": "user", "content": prompt}]
        return self.chat(
            Messages,
            model=model,
            stream=stream,
            enable_thinking=enable_thinking,
        )

    async def ask_async(
        self,
        prompt: str,
        *,
        model: str = DEFAULT_MODEL,
        stream: bool = False,
        enable_thinking: bool = False,
    ) -> TelnyxResponse | TelnyxStream:
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
    ) -> TelnyxResponse | TelnyxStream:
        Resolved = resolve_model(model)
        Body = self._build_payload(messages, Resolved, stream, enable_thinking)

        if stream:
            Resp = self.StreamClient.post(
                CHAT_URL,
                content=Body,
                headers=HEADERS,
                stream=True,
                timeout=600,
            )
            if Resp.status_code >= 400:
                Log.warning(
                    "telnyx stream failed: %s %s",
                    Resp.status_code,
                    Resp.text[:200] if hasattr(Resp, "text") else b"",
                )
                return TelnyxStream(_EmptyResp(), Resolved)
            return TelnyxStream(Resp, Resolved)

        Resp = self.HttpClient.post(
            CHAT_URL, content=Body, headers=HEADERS, timeout=600
        )
        if Resp.status_code >= 400:
            Log.warning(
                "telnyx request failed: %s %s",
                Resp.status_code,
                Resp.text[:200],
            )
            return TelnyxResponse(content="", model=Resolved, thinking="")

        Content = ""
        Thinking = ""
        for line in Resp.iter_lines():
            Text = line.decode(errors="ignore") if isinstance(line, bytes) else line
            ChunkContent, ChunkThinking, Done = _parse_line(Text)
            Content += ChunkContent
            Thinking += ChunkThinking
            if Done:
                break
        return TelnyxResponse(content=Content, model=Resolved, thinking=Thinking)

    async def chat_async(
        self,
        messages: list[dict],
        *,
        model: str = DEFAULT_MODEL,
        stream: bool = False,
        enable_thinking: bool = False,
    ) -> TelnyxResponse | TelnyxStream:
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
    "Telnyx",
    "TelnyxResponse",
    "TelnyxStream",
    "resolve_model",
]
