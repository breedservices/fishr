import logging
from typing import Any, AsyncIterator

from msgspec import Struct

from fishr.Http import make_client
from fishr.Loop import asyncio
from fishr.Utils import aiter_lines, json_decode, json_encode

Log = logging.getLogger("fishr.kai")
CHAT_URL = "https://api.kilo.ai/api/gateway/chat/completions"

# kai/<short-name> -> upstream model id on the gateway.
KAI_MODELS = {
    "kai/auto": "kilo-auto/free",
    "kai/m.1": "poolside/laguna-m.1:free",
    "kai/xs-2.1": "poolside/laguna-xs-2.1:free",
    "kai/xs.2": "poolside/laguna-xs.2:free",
    "kai/s-2.1": "poolside/laguna-s-2.1:free",
    "kai/north-mini": "cohere/north-mini-code:free",
    "kai/nemo3-ultra": "nvidia/nemotron-3-ultra-550b-a55b:free",
    "kai/nemo3-super": "nvidia/nemotron-3-super-120b-a12b:free",
    "kai/nemo3-nemo": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "kai/3.7-flash": "stepfun/step-3.7-flash:free",
    "kai/ling-3.0-flash": "inclusionai/ling-3.0-flash:free",
    "kai/openfree": "openrouter/free",
}
DEFAULT_MODEL = "kai/auto"

HEADERS = {
    "accept": "application/json",
    "content-type": "application/json",
    "origin": "https://api.kilo.ai",
    "referer": "https://api.kilo.ai/",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36",
}


def resolve_model(model: str) -> str:
    base = model.split("/", 1)[-1] if "/" in model else model
    key = f"kai/{base}"
    if key in KAI_MODELS:
        return key
    return DEFAULT_MODEL


def _upstream_model(resolved: str) -> str:
    return KAI_MODELS.get(resolved, KAI_MODELS[DEFAULT_MODEL])


def _build_payload(
    messages: list[dict],
    resolved: str,
    stream: bool,
    tools: list[dict] | None = None,
    tool_choice: Any = None,
    temperature: float = 0.7,
    max_tokens: int = 16384,
) -> bytes:
    payload: dict = {
        "model": _upstream_model(resolved),
        "messages": messages,
        "stream": stream,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if tools is not None:
        payload["tools"] = tools
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice
    return json_encode.encode(payload)


def _parse_chunk(obj: dict) -> dict:
    """Parse an OpenAI chat.completion(.chunk) dict.

    Returns {content, thinking, done, finish_reason, tool_calls}.
    tool_calls is a list of {index, id, type, name, arguments}.
    """
    out: dict[str, Any] = {
        "content": "",
        "thinking": "",
        "done": False,
        "finish_reason": "",
        "tool_calls": [],
    }
    choices = obj.get("choices") or []
    if choices:
        choice = choices[0]
        if choice.get("finish_reason"):
            out["done"] = True
            out["finish_reason"] = choice.get("finish_reason") or ""
        # Non-streaming uses `message`; streaming uses `delta`.
        body = choice.get("delta") or choice.get("message") or {}
        if isinstance(body, dict):
            out["content"] = body.get("content") or ""
            out["thinking"] = (
                body.get("reasoning") or body.get("reasoning_content") or ""
            )
            tcs = body.get("tool_calls")
            if isinstance(tcs, list):
                for tc in tcs:
                    if not isinstance(tc, dict):
                        continue
                    fn = tc.get("function") or {}
                    out["tool_calls"].append(
                        {
                            "index": tc.get("index", 0),
                            "id": tc.get("id", ""),
                            "type": tc.get("type", "function"),
                            "name": fn.get("name", ""),
                            "arguments": fn.get("arguments", ""),
                        }
                    )
    return out


def _parse_line(line: str) -> dict:
    if not line.startswith("data:"):
        return _empty_chunk()
    data = line[5:].strip()
    if data in ("[DONE]", "null", ""):
        out = _empty_chunk()
        out["done"] = True
        return out
    try:
        obj = json_decode.decode(data)
    except Exception:
        return _empty_chunk()
    if not isinstance(obj, dict):
        return _empty_chunk()
    return _parse_chunk(obj)


def _empty_chunk() -> dict:
    return {
        "content": "",
        "thinking": "",
        "done": False,
        "finish_reason": "",
        "tool_calls": [],
    }


def _merge_tool_calls(acc: list[dict], new: list[dict]) -> None:
    """Merge streamed tool-call deltas into the accumulator by index."""
    for n in new:
        idx = n.get("index", 0)
        while len(acc) <= idx:
            acc.append(
                {
                    "index": len(acc),
                    "id": "",
                    "type": "function",
                    "name": "",
                    "arguments": "",
                }
            )
        slot = acc[idx]
        if n.get("id") and not slot["id"]:
            slot["id"] = n["id"]
        if n.get("name") and not slot["name"]:
            slot["name"] = n["name"]
        slot["arguments"] += n.get("arguments", "")


def _finalize_tool_calls(acc: list[dict]) -> list[dict]:
    out: list[dict] = []
    for i, tc in enumerate(acc):
        name = tc.get("name", "")
        args = tc.get("arguments", "")
        if not name and not args:
            continue
        out.append(
            {
                "id": tc.get("id") or f"call_{i}",
                "type": tc.get("type", "function"),
                "name": name,
                "arguments": args,
            }
        )
    return out


class KaiResponse(Struct, frozen=True):
    content: str
    model: str
    thinking: str = ""
    finish_reason: str = ""
    tool_calls: tuple[dict, ...] = ()


class KaiStream:
    __slots__ = ("Resp", "Model")

    def __init__(self, Resp, Model: str) -> None:
        self.Resp = Resp
        self.Model = Model

    def __iter__(self):
        for line in self.Resp.iter_lines():
            text = line.decode(errors="ignore") if isinstance(line, bytes) else line
            chunk = _parse_line(text)
            if chunk["thinking"]:
                yield (chunk["thinking"], True)
            if chunk["content"]:
                yield (chunk["content"], False)
            if chunk["tool_calls"]:
                yield (chunk["tool_calls"], "tool_calls")
            if chunk["done"]:
                break
        self.Resp.close()

    async def __aiter__(self) -> AsyncIterator:
        async for line in aiter_lines(self.Resp):
            chunk = _parse_line(line)
            if chunk["thinking"]:
                yield (chunk["thinking"], True)
            if chunk["content"]:
                yield (chunk["content"], False)
            if chunk["tool_calls"]:
                yield (chunk["tool_calls"], "tool_calls")
            if chunk["done"]:
                break
        self.Resp.close()


class _EmptyResp:
    """Minimal stand-in for a primp response when the upstream fails."""

    def iter_lines(self):
        if False:
            yield b""

    def close(self) -> None:
        pass


class Kai:
    """
    Kilo AI Gateway (OpenAI-compatible, native tool calling).

    Free models on the gateway require no API key.

    Usage::

        ```py
        from fishr import Kai

        kai = Kai()
        result = kai.ask("Hello!")
        print(result.content)

        # tool calling
        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
        }]
        result = kai.chat(
            [{"role": "user", "content": "Weather in Tokyo?"}],
            tools=tools,
        )
        for tc in result.tool_calls:
            print(tc["name"], tc["arguments"])
        ```
    """

    __slots__ = ("HttpClient", "StreamClient")

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self.HttpClient = make_client(headers=HEADERS)
        self.StreamClient = make_client(headers=HEADERS)

    def _build_payload(
        self,
        messages: list[dict],
        resolved: str,
        stream: bool,
        tools: list[dict] | None = None,
        tool_choice: Any = None,
        temperature: float = 0.7,
        max_tokens: int = 16384,
    ) -> bytes:
        return _build_payload(
            messages, resolved, stream, tools, tool_choice, temperature, max_tokens
        )

    def ask(
        self,
        prompt: str,
        *,
        model: str = DEFAULT_MODEL,
        stream: bool = False,
        tools: list[dict] | None = None,
        tool_choice: Any = None,
    ) -> KaiResponse | KaiStream:
        messages = [{"role": "user", "content": prompt}]
        return self.chat(
            messages, model=model, stream=stream, tools=tools, tool_choice=tool_choice
        )

    async def ask_async(
        self,
        prompt: str,
        *,
        model: str = DEFAULT_MODEL,
        stream: bool = False,
        tools: list[dict] | None = None,
        tool_choice: Any = None,
    ) -> KaiResponse | KaiStream:
        return await asyncio.to_thread(
            self.ask,
            prompt,
            model=model,
            stream=stream,
            tools=tools,
            tool_choice=tool_choice,
        )

    def chat(
        self,
        messages: list[dict],
        *,
        model: str = DEFAULT_MODEL,
        stream: bool = False,
        tools: list[dict] | None = None,
        tool_choice: Any = None,
        temperature: float = 0.7,
        max_tokens: int = 16384,
    ) -> KaiResponse | KaiStream:
        resolved = resolve_model(model)
        body = self._build_payload(
            messages, resolved, stream, tools, tool_choice, temperature, max_tokens
        )

        if stream:
            resp = self.StreamClient.post(
                CHAT_URL,
                content=body,
                headers=HEADERS,
                stream=True,
                timeout=600,
            )
            if resp.status_code >= 400:
                Log.warning(
                    "kai stream failed: %s %s",
                    resp.status_code,
                    resp.text[:200] if hasattr(resp, "text") else b"",
                )
                return KaiStream(_EmptyResp(), resolved)
            return KaiStream(resp, resolved)

        resp = self.HttpClient.post(
            CHAT_URL, content=body, headers=HEADERS, timeout=600
        )
        if resp.status_code >= 400:
            Log.warning(
                "kai request failed: %s %s",
                resp.status_code,
                resp.text[:200] if hasattr(resp, "text") else b"",
            )
            return KaiResponse(content="", model=resolved)

        # Non-streaming gateway response is a single JSON body (not SSE).
        try:
            raw = json_decode.decode(resp.text)
        except Exception as exc:
            Log.warning("kai parse error: %s", exc)
            return KaiResponse(content="", model=resolved)

        if not isinstance(raw, dict):
            return KaiResponse(content="", model=resolved)

        chunk = _parse_chunk(raw)
        finish_reason = chunk["finish_reason"]
        tool_acc: list[dict] = []
        if chunk["tool_calls"]:
            _merge_tool_calls(tool_acc, chunk["tool_calls"])
        final = _finalize_tool_calls(tool_acc)
        return KaiResponse(
            content=chunk["content"],
            model=resolved,
            thinking=chunk["thinking"],
            finish_reason=finish_reason,
            tool_calls=tuple(final),
        )

    async def chat_async(
        self,
        messages: list[dict],
        *,
        model: str = DEFAULT_MODEL,
        stream: bool = False,
        tools: list[dict] | None = None,
        tool_choice: Any = None,
        temperature: float = 0.7,
        max_tokens: int = 16384,
    ) -> KaiResponse | KaiStream:
        return await asyncio.to_thread(
            self.chat,
            messages,
            model=model,
            stream=stream,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
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
    "Kai",
    "KaiResponse",
    "KaiStream",
    "resolve_model",
    "KAI_MODELS",
]
