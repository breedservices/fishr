import base64
import os
import time
from typing import AsyncIterator
from urllib.parse import urlencode

from msgspec import Struct

from fishr.Http import make_client
from fishr.Loop import asyncio
from fishr.Utils import aiter_lines, json_decode, json_encode

# Endpoints
API_V2 = "https://composer.opera-api.com/api/v2/a-chat"
TOKEN_URL = "https://oauth2.opera-api.com/oauth2/v1/token/"
SIGNUP_URL = "https://auth.opera.com/account/v2/external/anonymous/signup"

# Models
OPERA_MODELS = ["aria"]
DEFAULT_MODEL = "aria"

UA = "Mozilla/5.0 (Linux; U; Android 14; Pixel 8 Pro Build/UQ1A.240205.004; wv) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/138.0.7204.179 Mobile Safari/537.36 OPR/99.0.2254.81922"

TOKEN_HEADERS = {
    "User-Agent": "okhttp/5.3.2",
    "Content-Type": "application/x-www-form-urlencoded",
    "x-requested-with": "XMLHttpRequest",
    "x-opera-client-cache": "1",
}

SIGNUP_HEADERS = {
    "User-Agent": "okhttp/5.3.2",
    "Accept": "application/json",
    "Content-Type": "application/json; charset=utf-8",
    "x-requested-with": "XMLHttpRequest",
    "x-opera-client-cache": "1",
}


def resolve_model(model: str) -> str:
    base = model.split("/", 1)[-1] if "/" in model else model
    if base in OPERA_MODELS:
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


# ==================== Authentication ====================


def _generate_refresh_token(client) -> str:
    """Generate refresh token via anonymous signup."""
    # Step 1: Get anonymous access token
    form1 = urlencode(
        {
            "client_id": "mini-client",
            "client_secret": "Pcc5NvlCrxl02pMw32kO6WrnhpS0pUZ95YrDP8XNKJJQvFht4wQDkFJ7v9x5hn7C",
            "grant_type": "client_credentials",
            "scope": "anonymous_account",
        }
    )
    resp = client.post(TOKEN_URL, content=form1.encode(), headers=TOKEN_HEADERS)
    anon_token = json_decode.decode(resp.text)["access_token"]

    # Step 2: Anonymous signup
    resp = client.post(
        SIGNUP_URL,
        headers={**SIGNUP_HEADERS, "Authorization": f"Bearer {anon_token}"},
        json={"client_id": "mini"},
    )
    auth_token = json_decode.decode(resp.text)["token"]

    # Step 3: Exchange auth_token for refresh token
    form3 = urlencode(
        {
            "auth_token": auth_token,
            "client_id": "mini",
            "grant_type": "auth_token",
            "scope": "shodan:aria",
        }
    )
    resp = client.post(TOKEN_URL, content=form3.encode(), headers=TOKEN_HEADERS)
    return json_decode.decode(resp.text)["refresh_token"]


def _refresh_access_token(client, refresh_token: str) -> tuple[str, int]:
    """Refresh the access token using a refresh token."""
    form = urlencode(
        {
            "client_id": "mini",
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "scope": "shodan:aria",
        }
    )
    resp = client.post(TOKEN_URL, content=form.encode(), headers=TOKEN_HEADERS)
    result = json_decode.decode(resp.text)
    return result["access_token"], result.get("expires_in", 3600)


# ==================== SSE Parsing ====================


def _parse_sse_line(line: str) -> tuple[str | None, dict | None]:
    """Parse an SSE line into (event_type, data)."""
    line = line.strip()
    if line.startswith("event:"):
        return line[6:].strip(), None
    if line.startswith("data:"):
        content = line[5:].strip()
        if content in ("[DONE]", "null", ""):
            return None, None
        try:
            return None, json_decode.decode(content)
        except Exception:
            return None, None
    return None, None


def _extract_content(data: dict) -> tuple[str | None, str | None, bool]:
    """Extract (text, image_url, is_thinking) from response data."""
    response = data.get("response", {})
    if not isinstance(response, dict):
        return None, None, False

    content_type = response.get("content_type")
    if content_type == "image":
        return None, response.get("image_url"), False

    is_thinking = content_type == "thinking"
    text = response.get("message") if isinstance(response.get("message"), str) else None
    return text, None, is_thinking


def _extract_conversation_id(data: dict) -> str | None:
    """Extract conversation ID from response data."""
    metadata = data.get("metadata", {})
    return metadata.get("conversation_id") if isinstance(metadata, dict) else None


def _parse_sse_text(raw: str) -> tuple[str, list[str], str | None, str]:
    """Parse SSE text into (content, image_urls, conversation_id, thinking)."""
    out: list[str] = []
    thinking_parts: list[str] = []
    image_urls: list[str] = []
    conv_id = None
    in_thinking = False

    for line in raw.splitlines():
        evt, data = _parse_sse_line(line)

        if evt:
            if evt == "thinking_status":
                in_thinking = True
            continue

        if data is None:
            continue

        content, image_url, is_thinking = _extract_content(data)

        if is_thinking:
            in_thinking = True
        elif in_thinking:
            in_thinking = False

        if image_url:
            image_urls.append(image_url)
        if content and in_thinking:
            thinking_parts.append(content)
        elif content:
            out.append(content)

        cid = _extract_conversation_id(data)
        if cid:
            conv_id = cid

    return "".join(out), image_urls, conv_id, "".join(thinking_parts)


# ==================== Headers & Payload ====================


def _build_chat_headers(access_token: str) -> dict:
    """Build request headers for the chat endpoint."""
    return {
        "Accept": "text/event-stream",
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "Origin": "https://composer.opera-api.com",
        "Referer": "https://composer.opera-api.com/assets/aria/index.html",
        "User-Agent": UA,
        "X-Opera-Timezone": "+02:00",
        "X-Opera-UI-Language": "en",
        "X-Requested-With": "com.opera.mini.native",
    }


def _build_payload(
    prompt: str,
    encryption_key: str,
    conversation_id: str | None,
    is_first: bool,
    think_harder: bool = False,
) -> bytes:
    """Build the request payload for the chat endpoint."""
    data = {
        "query": prompt,
        "sia": True,
        "think_harder": think_harder,
        "supported_features": [],
        "file_attachments": [],
        "encryption": {"key": encryption_key},
    }

    if not is_first and conversation_id:
        data["conversation_id"] = conversation_id

    return json_encode.encode(data)


class OperaAriaResponse(Struct, frozen=True):
    content: str
    model: str
    image_urls: tuple[str, ...] = ()
    thinking: str = ""


class OperaAriaStream:
    __slots__ = ("resp", "model", "image_urls", "_conv_id")

    def __init__(self, resp, model: str) -> None:
        self.resp = resp
        self.model = model
        self.image_urls: list[str] = []
        self._conv_id: str | None = None

    def __iter__(self):
        in_thinking = False
        for line in self.resp.iter_lines():
            text = line.decode(errors="ignore") if isinstance(line, bytes) else line
            evt, data = _parse_sse_line(text)
            if evt:
                if evt == "thinking_status":
                    in_thinking = True
                continue
            if data is None:
                continue
            content, image_url, is_thinking = _extract_content(data)
            if is_thinking:
                in_thinking = True
            elif in_thinking:
                in_thinking = False
            if image_url:
                self.image_urls.append(image_url)
            if content:
                yield content, in_thinking
            cid = _extract_conversation_id(data)
            if cid:
                self._conv_id = cid
        self.resp.close()

    async def __aiter__(self) -> AsyncIterator[tuple[str, bool]]:
        in_thinking = False
        async for line in aiter_lines(self.resp):
            evt, data = _parse_sse_line(line)
            if evt:
                if evt == "thinking_status":
                    in_thinking = True
                continue
            if data is None:
                continue
            content, image_url, is_thinking = _extract_content(data)
            if is_thinking:
                in_thinking = True
            elif in_thinking:
                in_thinking = False
            if image_url:
                self.image_urls.append(image_url)
            if content:
                yield content, in_thinking
            cid = _extract_conversation_id(data)
            if cid:
                self._conv_id = cid


class OperaAria:
    __slots__ = (
        "http_client",
        "stream_client",
        "refresh_token",
        "access_token",
        "expires_at",
        "encryption_key",
        "conversation_id",
        "is_first_request",
    )

    def __init__(self, model: str = "opera/aria") -> None:
        self.http_client = make_client()
        self.stream_client = make_client()
        self.refresh_token: str | None = None
        self.access_token: str | None = None
        self.expires_at: float = 0
        self.encryption_key = base64.b64encode(os.urandom(32)).decode("utf-8")
        self.conversation_id: str | None = None
        self.is_first_request = True

    def _ensure_access_token(self) -> str:
        """Get a valid access token, refreshing or generating as needed."""
        if not self.refresh_token:
            self.refresh_token = _generate_refresh_token(self.http_client)
        if self.access_token and time.time() < self.expires_at:
            return self.access_token
        token, expires_in = _refresh_access_token(self.http_client, self.refresh_token)
        self.access_token = token
        self.expires_at = time.time() + expires_in - 60
        return token

    def ask(
        self,
        prompt: str,
        *,
        model: str = "opera/aria",
        stream: bool = False,
        think_harder: bool = False,
    ) -> OperaAriaResponse | OperaAriaStream:
        messages = [{"role": "user", "content": prompt}]
        return self.chat(
            messages, model=model, stream=stream, think_harder=think_harder
        )

    async def ask_async(
        self,
        prompt: str,
        *,
        model: str = "opera/aria",
        stream: bool = False,
        think_harder: bool = False,
    ) -> OperaAriaResponse | OperaAriaStream:
        return await asyncio.to_thread(
            self.ask,
            prompt,
            model=model,
            stream=stream,
            think_harder=think_harder,
        )

    def chat(
        self,
        messages: list[dict],
        *,
        model: str = "opera/aria",
        stream: bool = False,
        think_harder: bool = False,
    ) -> OperaAriaResponse | OperaAriaStream:
        resolved = resolve_model(model)
        access_token = self._ensure_access_token()
        headers = _build_chat_headers(access_token)
        prompt = _format_prompt(messages)
        body = _build_payload(
            prompt,
            self.encryption_key,
            self.conversation_id,
            self.is_first_request,
            think_harder=think_harder,
        )

        if stream:
            resp = self.stream_client.post(
                API_V2, content=body, headers=headers, stream=True
            )
            return OperaAriaStream(resp, resolved)

        resp = self.http_client.post(API_V2, content=body, headers=headers)
        raw = resp.text
        if resp.status_code >= 400:
            raise RuntimeError(f"OperaAria API error {resp.status_code}: {raw[:500]}")
        content, image_urls, conv_id, thinking = _parse_sse_text(raw)
        if not content and not image_urls:
            # Fallback: try parsing the raw text as a single JSON blob
            try:
                data = json_decode.decode(raw)
                if isinstance(data, dict):
                    text, img, _ = _extract_content(data)
                    if text:
                        content = text
                    if img:
                        image_urls.append(img)
                    cid = _extract_conversation_id(data)
                    if cid:
                        conv_id = cid
            except Exception:
                pass
        if conv_id:
            self.conversation_id = conv_id
        self.is_first_request = False
        return OperaAriaResponse(
            content=content,
            model=resolved,
            image_urls=tuple(image_urls),
            thinking=thinking,
        )

    async def chat_async(
        self,
        messages: list[dict],
        *,
        model: str = "opera/aria",
        stream: bool = False,
        think_harder: bool = False,
    ) -> OperaAriaResponse | OperaAriaStream:
        return await asyncio.to_thread(
            self.chat,
            messages,
            model=model,
            stream=stream,
            think_harder=think_harder,
        )

    def new_chat(self) -> None:
        self.conversation_id = None
        self.is_first_request = True

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args) -> None:
        pass


__all__ = [
    "OperaAria",
    "OperaAriaResponse",
    "OperaAriaStream",
    "resolve_model",
]
