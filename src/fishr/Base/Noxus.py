from random import choices
from string import ascii_letters, ascii_lowercase, digits
from time import time
from urllib.parse import urlencode
from uuid import uuid4

from msgspec import DecodeError, Struct
from primp import Client as PrimpClient

from fishr.Http import make_client
from fishr.Loop import asyncio
from fishr.Types import noxus_bot_ids
from fishr.Utils import aiter_lines, build_multipart, json_decode, json_encode

BASE = "https://chatgptfree.ai"
AJAX = f"{BASE}/wp-admin/admin-ajax.php"
DATA_PREFIX = "data: "

HEADERS = {
    "accept": "*/*",
    "origin": BASE,
    "referer": f"{BASE}/",
}

STREAM_HEADERS = {
    "accept": "text/event-stream",
    "origin": BASE,
    "referer": f"{BASE}/",
}

ALLOWED_EVENTS = frozenset({"message", "delta", "token", "content"})

ALLOWED_IMAGE_MIMES = frozenset({"image/webp", "image/png", "image/jpeg", "image/jpg"})


def _gen_msg_id(bot_id: int) -> str:
    ts = int(time() * 1000)
    suffix = "".join(choices(ascii_lowercase, k=5))
    return f"aipkit-client-msg-{bot_id}-{ts}-{suffix}"


def _gen_cache_key() -> str:
    chars = ascii_letters + digits
    key = "".join(choices(chars, k=32))
    return f"aipkit_sse_{key}"


def _gen_conversation_uuid() -> str:
    return str(uuid4())


def _gen_session_id() -> str:
    return str(uuid4())


def _post_multipart(client: PrimpClient, fields: dict[str, str]) -> dict:
    body, boundary = build_multipart(fields)
    resp = client.post(
        AJAX,
        content=body,
        headers={"content-type": f"multipart/form-data; boundary={boundary}"},
    )
    return json_decode.decode(resp.text)


def _fetch_nonce(client: PrimpClient, bot_id: int) -> str:
    fields = {
        "action": "aipkit_get_frontend_chat_nonce",
        "bot_id": str(bot_id),
    }
    data = _post_multipart(client, fields)
    if (
        isinstance(data, dict)
        and data.get("success")
        and isinstance(data.get("data"), dict)
    ):
        nonce = data["data"].get("nonce")
        if nonce:
            return nonce
    raise RuntimeError("Noxus: nonce refresh failed")


def _extract_content(data) -> str:
    if isinstance(data, str):
        return data
    delta = data.get("delta")
    if isinstance(delta, str) and delta:
        return delta
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0]
        if isinstance(choice, dict):
            d = choice.get("delta")
            if isinstance(d, dict):
                c = d.get("content")
                if isinstance(c, str):
                    return c
            msg = choice.get("message")
            if isinstance(msg, dict):
                c = msg.get("content")
                if isinstance(c, str):
                    return c
    for key in ("token", "text", "content"):
        val = data.get(key)
        if isinstance(val, str) and val:
            return val
    inner = data.get("data")
    if isinstance(inner, dict):
        for key in ("token", "text", "content"):
            val = inner.get(key)
            if isinstance(val, str) and val:
                return val
    inner2 = data.get("message")
    if isinstance(inner2, dict):
        for key in ("token", "text", "content"):
            val = inner2.get(key)
            if isinstance(val, str) and val:
                return val
    return ""


def _parse_line(line: str, event_type: str | None) -> tuple[str, str | None]:
    if event_type and event_type not in ALLOWED_EVENTS:
        return "", None
    if not line.startswith(DATA_PREFIX):
        return "", None
    chunk = line[len(DATA_PREFIX) :]
    if chunk in ("[DONE]", ""):
        return "", None
    try:
        data = json_decode.decode(chunk)
    except DecodeError:
        return "", None
    rid = data.get("response_id") or data.get("id")
    response_id = None
    if isinstance(rid, str) and rid.startswith("resp_"):
        response_id = rid
    content = _extract_content(data)
    return content, response_id


def _collect_lines(resp) -> tuple[str, str | None]:
    out: list[str] = []
    event_type = None
    response_id = None
    for line in resp.iter_lines():
        stripped = line.strip()
        if not stripped:
            event_type = None
            continue
        if stripped.startswith("event:"):
            event_type = stripped[6:].strip()
            continue
        content, rid = _parse_line(stripped, event_type)
        if rid:
            response_id = rid
        if content:
            out.append(content)
    resp.close()
    return "".join(out), response_id


class Image(Struct, frozen=True):
    mime_type: str
    base64_data: str


class NoxusMessage(Struct, frozen=True):
    role: str
    content: str
    image: Image | None = None


class NoxusResponse(Struct, frozen=True):
    content: str
    conversation_uuid: str
    previous_response_id: str | None = None
    response: str = ""


class Stream:
    __slots__ = ("resp",)

    def __init__(self, resp) -> None:
        self.resp = resp

    def __iter__(self):
        event_type = None
        for line in self.resp.iter_lines():
            stripped = line.strip()
            if not stripped:
                event_type = None
                continue
            if stripped.startswith("event:"):
                event_type = stripped[6:].strip()
                continue
            content, _ = _parse_line(stripped, event_type)
            if content:
                yield content
        self.resp.close()

    async def __aiter__(self):
        event_type = None
        async for line in aiter_lines(self.resp):
            stripped = line.strip()
            if not stripped:
                event_type = None
                continue
            if stripped.startswith("event:"):
                event_type = stripped[6:].strip()
                continue
            content, _ = _parse_line(stripped, event_type)
            if content:
                yield content


class Noxus:
    __slots__ = (
        "http_client",
        "stream_client",
        "bot_id",
        "session_id",
        "conv_uuid",
        "nonce",
        "prev_response_id",
    )

    def __init__(self, model: str = "noxus/openai") -> None:
        self.http_client = make_client(headers=HEADERS)
        self.stream_client = make_client(headers=STREAM_HEADERS)
        self.bot_id = noxus_bot_ids.get(model, noxus_bot_ids["noxus/openai"])
        self.session_id = _gen_session_id()
        self.conv_uuid = _gen_conversation_uuid()
        self.nonce = ""
        self.prev_response_id: str | None = None

    def _refresh_nonce(self) -> str:
        self.nonce = _fetch_nonce(self.http_client, self.bot_id)
        return self.nonce

    def _post_message(self, user_input: str, images: list[Image] | None = None) -> dict:
        if not self.nonce:
            self._refresh_nonce()
        fields = {
            "action": "aipkit_cache_sse_message",
            "message": user_input,
            "_ajax_nonce": self.nonce,
            "bot_id": str(self.bot_id),
            "user_client_message_id": _gen_msg_id(self.bot_id),
        }
        if images:
            inputs = []
            for img in images:
                if img.mime_type in ALLOWED_IMAGE_MIMES:
                    inputs.append(
                        {"mime_type": img.mime_type, "base64_data": img.base64_data}
                    )
            if inputs:
                fields["image_inputs"] = json_encode.encode(inputs).decode()
        data = _post_multipart(self.http_client, fields)
        if isinstance(data, dict) and data.get("success") is False:
            self.nonce = ""
            self._refresh_nonce()
            fields["_ajax_nonce"] = self.nonce
            data = _post_multipart(self.http_client, fields)
        return data

    def _get_cache_key(self, post_data: dict) -> str:
        return (
            (
                post_data.get("data", {}).get("cache_key")
                if isinstance(post_data.get("data"), dict)
                else None
            )
            or post_data.get("cache_key", "")
            or _gen_cache_key()
        )

    def _build_stream_url(self, cache_key: str, web_search: bool = False) -> str:
        ts = int(time() * 1000)
        params: dict[str, str] = {
            "action": "aipkit_frontend_chat_stream",
            "cache_key": cache_key,
            "bot_id": str(self.bot_id),
            "session_id": self.session_id,
            "conversation_uuid": self.conv_uuid,
            "post_id": "6",
            "_ts": str(ts),
            "frontend_web_search_active": "true" if web_search else "false",
            "_ajax_nonce": self.nonce,
        }
        if self.prev_response_id:
            params["previous_openai_response_id"] = self.prev_response_id
        return f"{AJAX}?{urlencode(params)}"

    def chat(
        self,
        messages: tuple[NoxusMessage, ...],
        *,
        model: str = "noxus/openai",
        web_search: bool = False,
        stream: bool = False,
    ) -> NoxusResponse | object:
        new_bot_id = noxus_bot_ids.get(model, noxus_bot_ids["noxus/openai"])
        if new_bot_id != self.bot_id:
            self.bot_id = new_bot_id
            self.nonce = ""
        last_user = [m for m in messages if m.role == "user"][-1]
        images = [m.image for m in messages if m.role == "user" and m.image is not None]
        post_data = self._post_message(last_user.content, images=images or None)
        cache_key = self._get_cache_key(post_data)
        url = self._build_stream_url(cache_key, web_search=web_search)
        resp = self.stream_client.get(url, stream=True)
        if stream:
            return Stream(resp)
        text, response_id = _collect_lines(resp)
        if response_id:
            self.prev_response_id = response_id
        return NoxusResponse(
            content=text,
            conversation_uuid=self.conv_uuid,
            previous_response_id=self.prev_response_id,
        )

    def ask(
        self,
        prompt: str,
        *,
        model: str = "noxus/openai",
        web_search: bool = False,
        images: list[Image] | None = None,
    ) -> str:
        messages = (NoxusMessage(role="user", content=prompt),)
        resp = self.chat(messages, model=model, web_search=web_search)
        return resp.content

    async def ask_async(
        self,
        prompt: str,
        *,
        model: str = "noxus/openai",
        web_search: bool = False,
        images: list[Image] | None = None,
    ) -> str:
        return await asyncio.to_thread(
            self.ask,
            prompt,
            model=model,
            web_search=web_search,
            images=images,
        )

    async def chat_async(
        self,
        messages: tuple[NoxusMessage, ...],
        *,
        model: str = "noxus/openai",
        web_search: bool = False,
    ) -> NoxusResponse:
        return await asyncio.to_thread(
            self.chat,
            messages,
            model=model,
            web_search=web_search,
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
    "Noxus",
    "NoxusResponse",
    "NoxusMessage",
    "Image",
    "Stream",
]
