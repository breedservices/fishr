import mimetypes
from base64 import b64decode
from hashlib import md5
from random import random
from typing import AsyncIterator
from uuid import uuid4

from msgspec import Struct

from fishr.Http import make_client
from fishr.Loop import asyncio
from fishr.Utils import aiter_lines, build_multipart, json_decode, json_encode

USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36"

CHAT_URL = "https://api.deepai.org/hacking_is_a_serious_crime"
IMAGE_URL = "https://api.deepai.org/api/text2img"
UPLOAD_URL = "https://api.deepai.org/chat_attachments/upload"

DEEPAI_MODELS = [
    "standard",
    "online",
    "gemma-4",
    "gemini-2.5-flash-lite",
    "deepseek-v3.2",
    "image",
]

MODEL_ALIASES = {
    "gpt-4": "standard",
}

DEFAULT_MODEL = "standard"


def resolve_model(model: str) -> str:
    base = model.split("/", 1)[-1] if "/" in model else model
    if base in DEEPAI_MODELS:
        return base
    aliased = MODEL_ALIASES.get(base)
    if aliased:
        return aliased
    return DEFAULT_MODEL


def generate_api_key(user_agent: str) -> str:
    random_str = str(round(random() * 100000000000))

    def hash_fn(input_str: str) -> str:
        return md5(input_str.encode("utf-8")).hexdigest()[::-1]

    hash1 = hash_fn(
        user_agent
        + random_str
        + "hackers_become_a_little_stinkier_every_time_they_hack"
    )
    hash2 = hash_fn(user_agent + hash1)
    hash3 = hash_fn(user_agent + hash2)
    return f"tryit-{random_str}-{hash3}"


def _build_headers() -> dict[str, str]:
    api_key = generate_api_key(USER_AGENT)
    return {
        "api-key": api_key,
        "user-agent": USER_AGENT,
        "origin": "https://deepai.org",
        "referer": "https://deepai.org/chat",
    }


def _upload_file(client, headers: dict, file_data: bytes, filename: str) -> str:
    content_type = mimetypes.guess_type(filename)[0] or "image/png"
    boundary = "----WebKitFormBoundary" + "".join(
        __import__("random").choices(
            __import__("string").ascii_letters + __import__("string").digits,
            k=16,
        ),
    )
    parts: list[bytes] = []
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\nContent-Type: {content_type}\r\n\r\n'.encode(),
    )
    parts.append(file_data)
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    body = b"".join(parts)
    upload_headers = {
        k: v for k, v in headers.items() if k.lower() not in ("content-type",)
    }
    upload_headers["content-type"] = f"multipart/form-data; boundary={boundary}"
    resp = client.post(UPLOAD_URL, content=body, headers=upload_headers)
    result = json_decode.decode(resp.text)
    if isinstance(result, dict) and result.get("success"):
        return result["attachment"]["uuid"]
    raise RuntimeError(f"DeepAI upload failed: {resp.text}")


ALLOWED_IMAGE_MIMES = frozenset({"image/webp", "image/png", "image/jpeg", "image/jpg"})


def _extract_images(messages: list[dict]) -> list[tuple[bytes, str]]:
    images: list[tuple[bytes, str]] = []
    for m in messages:
        img = m.get("image")
        if isinstance(img, dict) and img.get("mime_type") in ALLOWED_IMAGE_MIMES:
            raw = img.get("base64_data", "")
            file_data = b64decode(raw)
            mime = img.get("mime_type", "image/png")
            ext = mime.split("/")[-1] if "/" in mime else "png"
            if ext == "jpeg":
                ext = "jpg"
            filename = f"image.{ext}"
            images.append((file_data, filename))
    return images


class DeepAIResponse(Struct, frozen=True):
    content: str
    model: str
    image_url: str | None = None


class DeepAIStream:
    __slots__ = ("resp", "model")

    def __init__(self, resp, model: str) -> None:
        self.resp = resp
        self.model = model

    def __iter__(self):
        buffer = ""
        for line in self.resp.iter_lines():
            text = line.decode(errors="ignore") if isinstance(line, bytes) else line
            if "\x1c" in text or "\x1c" in buffer:
                buffer += text
            else:
                yield text
        if buffer:
            yield from _drain_buffer(buffer)
        self.resp.close()

    async def __aiter__(self) -> AsyncIterator[str]:
        buffer = ""
        async for line in aiter_lines(self.resp):
            if "\x1c" in line or "\x1c" in buffer:
                buffer += line
            else:
                yield line
        if buffer:
            for chunk in _drain_buffer(buffer):
                yield chunk


def _drain_buffer(buffer: str):
    parts = buffer.split("\x1c")
    if parts[0].strip():
        yield parts[0]
    for part in parts[1:]:
        part = part.strip()
        if not part:
            continue
        try:
            data = json_decode.decode(part)
            if isinstance(data, dict) and data.get("type") == "generated_image":
                yield f"[Generated image: {data.get('prompt', '')}]"
                continue
        except Exception:
            pass
        yield part


class DeepAI:
    __slots__ = ("http_client",)

    def __init__(self) -> None:
        self.http_client = make_client()

    def _post_form(self, url: str, fields: dict[str, str], headers: dict[str, str]):
        body, boundary = build_multipart(fields)
        return self.http_client.post(
            url,
            content=body,
            headers={
                **headers,
                "content-type": f"multipart/form-data; boundary={boundary}",
            },
        )

    def _upload_attachments(
        self,
        headers: dict,
        messages: list[dict],
    ) -> list[str]:
        images = _extract_images(messages)
        uuids: list[str] = []
        for file_data, filename in images:
            file_uuid = _upload_file(self.http_client, headers, file_data, filename)
            uuids.append(file_uuid)
        return uuids

    def ask(
        self,
        prompt: str,
        *,
        model: str = "deepai/standard",
        web_search: bool = False,
        stream: bool = False,
    ) -> DeepAIResponse | DeepAIStream:
        resolved = resolve_model(model)
        headers = _build_headers()
        messages = [{"role": "user", "content": prompt}]

        if resolved == "image":
            return self._generate_image(prompt, resolved, headers)

        chat_model = "online" if (web_search and resolved != "online") else resolved

        form_data = {
            "chat_style": "chat",
            "chatHistory": json_encode.encode(messages).decode(),
            "model": chat_model,
            "session_uuid": str(uuid4()),
            "sensitivity_request_id": str(uuid4()),
            "hacker_is_stinky": "very_stinky",
            "enabled_tools": '["image_generator","image_editor"]',
        }

        resp = self._post_form(CHAT_URL, form_data, headers)

        if stream:
            return DeepAIStream(resp, resolved)

        text = resp.text
        content = _extract_text(text)
        return DeepAIResponse(content=content, model=resolved)

    async def ask_async(
        self,
        prompt: str,
        *,
        model: str = "deepai/standard",
        web_search: bool = False,
        stream: bool = False,
    ) -> DeepAIResponse | DeepAIStream:
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
        model: str = "deepai/standard",
        web_search: bool = False,
        stream: bool = False,
    ) -> DeepAIResponse | DeepAIStream:
        resolved = resolve_model(model)
        headers = _build_headers()

        last_user = [m for m in messages if m.get("role") == "user"][-1]
        prompt = last_user["content"]

        if resolved == "image":
            return self._generate_image(prompt, resolved, headers)

        chat_model = "online" if (web_search and resolved != "online") else resolved

        attachment_uuids = self._upload_attachments(headers, messages)

        cleaned = []
        for m in messages:
            clean = {"role": m["role"], "content": m["content"]}
            cleaned.append(clean)

        if attachment_uuids:
            for i in range(len(cleaned) - 1, -1, -1):
                if cleaned[i]["role"] == "user":
                    cleaned[i]["attachment_uuids"] = attachment_uuids
                    break

        form_data = {
            "chat_style": "chat",
            "chatHistory": json_encode.encode(cleaned).decode(),
            "model": chat_model,
            "session_uuid": str(uuid4()),
            "sensitivity_request_id": str(uuid4()),
            "hacker_is_stinky": "very_stinky",
            "enabled_tools": '["image_generator","image_editor"]',
        }

        if attachment_uuids:
            form_data["attachment_uuids"] = json_encode.encode(
                attachment_uuids
            ).decode()

        resp = self._post_form(CHAT_URL, form_data, headers)

        if stream:
            return DeepAIStream(resp, resolved)

        text = resp.text
        content = _extract_text(text)
        return DeepAIResponse(content=content, model=resolved)

    async def chat_async(
        self,
        messages: list[dict],
        *,
        model: str = "deepai/standard",
        web_search: bool = False,
        stream: bool = False,
    ) -> DeepAIResponse | DeepAIStream:
        return await asyncio.to_thread(
            self.chat,
            messages,
            model=model,
            web_search=web_search,
            stream=stream,
        )

    def _generate_image(
        self,
        prompt: str,
        model: str,
        headers: dict,
    ) -> DeepAIResponse:
        form_data = {
            "text": prompt,
            "generation_source": "chat",
            "width": "640",
            "height": "640",
            "image_generator_version": "hd",
            "quality": "true",
        }
        resp = self._post_form(IMAGE_URL, form_data, headers)
        try:
            result = json_decode.decode(resp.text)
            image_url = result.get("output_url", "")
        except Exception:
            image_url = ""
        return DeepAIResponse(
            content=f"[Generated image: {prompt}]",
            model=model,
            image_url=image_url,
        )


def _extract_text(raw: str) -> str:
    if "\x1c" not in raw:
        return raw.strip()
    parts = raw.split("\x1c")
    out = []
    if parts[0].strip():
        out.append(parts[0].strip())
    for part in parts[1:]:
        part = part.strip()
        if not part:
            continue
        try:
            data = json_decode.decode(part)
            if isinstance(data, list):
                continue
            if isinstance(data, dict) and data.get("type") == "generated_image":
                out.append(f"[Generated image: {data.get('prompt', '')}]")
                continue
        except Exception:
            pass
        out.append(part)
    return "".join(out)


__all__ = [
    "DeepAI",
    "DeepAIResponse",
    "DeepAIStream",
    "resolve_model",
]
