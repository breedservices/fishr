from __future__ import annotations

import logging
import os
import re
import traceback
from typing import AsyncIterator

from msgspec import Struct
from msgspec.json import Encoder as JsonEncoder
from msgspec.json import decode as json_decode

from fishr.client import AsyncClient
from fishr.Loop import asyncio
from fishr.Types import models as model_registry

json_encode = JsonEncoder()

log = logging.getLogger("fishr.chat")

_HERE = os.path.dirname(os.path.abspath(__file__))
_HTML_PATH = os.path.join(_HERE, "ui.html")

_html_cache: bytes | None = None
_client: AsyncClient | None = None


class ChatRequest(Struct, frozen=True):
    model: str = "noxus/openai"
    messages: list[dict] = []
    web_search: bool = False
    think_harder: bool = False
    stream: bool = True


class CompareRequest(Struct, frozen=True):
    models: list[str] = []
    messages: list[dict] = []
    web_search: bool = False
    think_harder: bool = False


class ImageRequest(Struct, frozen=True):
    model: str = "deepai/image"
    prompt: str = ""


class AudioRequest(Struct, frozen=True):
    model: str = "fm/coral"
    input: str = ""
    voice: str = ""
    instructions: str = ""
    format: str = ""


class ModelEntry(Struct, frozen=True):
    id: str
    web_search: bool
    image: bool
    file_attach: bool
    history: bool
    system: bool


def _get_client() -> AsyncClient:
    global _client
    if _client is None:
        _client = AsyncClient()
    return _client


def _load_html() -> bytes:
    global _html_cache
    if _html_cache is None:
        with open(_HTML_PATH, "rb") as fh:
            _html_cache = fh.read()
    return _html_cache


def _build_model_list() -> list[dict]:
    out: list[dict] = []
    for name, m in model_registry.items():
        out.append(
            ModelEntry(
                id=name,
                web_search=m.web_search,
                image=m.image,
                file_attach=m.file_attach,
                history=m.history,
                system=m.system,
            )
        )
    return out


_STATUS_REASONS = {200: "OK", 204: "No Content", 400: "Bad Request", 404: "Not Found"}


def _write_response(
    writer: asyncio.StreamWriter,
    status: int,
    content_type: str,
    body: bytes,
) -> None:
    reason = _STATUS_REASONS.get(status, "OK")
    header = (
        f"HTTP/1.1 {status} {reason}\r\n"
        f"Content-Type: {content_type}\r\n"
        f"Content-Length: {len(body)}\r\n"
        f"Access-Control-Allow-Origin: *\r\n"
        f"\r\n"
    )
    writer.write(header.encode() + body)


def _write_json(writer: asyncio.StreamWriter, status: int, data: object) -> None:
    body = json_encode.encode(data)
    _write_response(writer, status, "application/json", body)


async def _handle_chat(writer: asyncio.StreamWriter, body: bytes) -> None:
    try:
        req = json_decode(body, type=ChatRequest)
    except Exception:
        _write_json(writer, 400, {"error": "invalid request"})
        await writer.drain()
        return

    if not req.messages:
        _write_json(writer, 400, {"error": "messages is empty"})
        await writer.drain()
        return

    client = _get_client()

    if req.stream:
        writer.write(
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: text/event-stream\r\n"
            b"Cache-Control: no-cache\r\n"
            b"Connection: keep-alive\r\n"
            b"Access-Control-Allow-Origin: *\r\n\r\n"
        )
        await writer.drain()

        response = await client.chat.completions.create(
            model=req.model,
            messages=req.messages,
            web_search=req.web_search,
            stream=True,
            think_harder=req.think_harder,
        )

        async for chunk in response:
            if not chunk.choices:
                continue
            d = chunk.choices[0].delta
            if d.thinking:
                payload = json_encode.encode(
                    {"type": "thinking", "content": d.thinking}
                )
            else:
                payload = json_encode.encode({"type": "content", "content": d.content})
            writer.write(b"data: " + payload + b"\n\n")
            await writer.drain()

        writer.write(b"data: [DONE]\n\n")
        await writer.drain()
        writer.close()
    else:
        result = await client.chat.completions.create(
            model=req.model,
            messages=req.messages,
            web_search=req.web_search,
            stream=False,
            think_harder=req.think_harder,
        )
        content = result.text
        image_urls: list[str] = []
        if req.model.startswith("opera/"):
            if hasattr(result, "_raw") and hasattr(result._raw, "image_urls"):
                image_urls = list(result._raw.image_urls)
        _write_json(writer, 200, {"content": content, "image_urls": image_urls})
        await writer.drain()


async def _handle_images(writer: asyncio.StreamWriter, body: bytes) -> None:
    try:
        req = json_decode(body, type=ImageRequest)
    except Exception:
        _write_json(writer, 400, {"error": "invalid request"})
        await writer.drain()
        return

    if not req.prompt:
        _write_json(writer, 400, {"error": "prompt is empty"})
        await writer.drain()
        return

    client = _get_client()

    # DeepAI image generation
    if req.model.startswith("deepai/"):
        result = await client.images.generate(
            model=req.model,
            prompt=req.prompt,
        )
        urls = [img.url for img in result.data if img.url]
        _write_json(writer, 200, {"urls": urls})
        await writer.drain()
        return

    # Opera Aria image generation via chat
    if req.model.startswith("opera/"):
        result = await client.chat.completions.create(
            model=req.model,
            messages=[
                {"role": "user", "content": f"Generate an image of: {req.prompt}"}
            ],
            stream=False,
        )
        content = result.text
        img_urls: list[str] = []
        img_regex = re.compile(
            r"https?://[^\s]+\.(?:png|jpg|jpeg|gif|webp)", re.IGNORECASE
        )
        for match in img_regex.finditer(content):
            img_urls.append(match.group(0))
        _write_json(writer, 200, {"urls": img_urls, "content": content})
        await writer.drain()
        return

    _write_json(
        writer, 400, {"error": f"image generation not supported for {req.model}"}
    )
    await writer.drain()


async def _handle_compare(writer: asyncio.StreamWriter, body: bytes) -> None:
    try:
        req = json_decode(body, type=CompareRequest)
    except Exception:
        _write_json(writer, 400, {"error": "invalid request"})
        await writer.drain()
        return

    if not req.messages or not req.models:
        _write_json(writer, 400, {"error": "messages and models are required"})
        await writer.drain()
        return

    if len(req.models) > 4:
        _write_json(writer, 400, {"error": "maximum 4 models for comparison"})
        await writer.drain()
        return

    writer.write(
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Type: text/event-stream\r\n"
        b"Cache-Control: no-cache\r\n"
        b"Connection: keep-alive\r\n"
        b"Access-Control-Allow-Origin: *\r\n\r\n"
    )
    await writer.drain()

    client = _get_client()
    log.info("compare start: %s", req.models)

    async def _stream_model(model_id: str, idx: int) -> None:
        payload = json_encode.encode({"model": model_id, "idx": idx})
        writer.write(b"data: " + payload + b"\n\n")
        await writer.drain()

        try:
            response = await client.chat.completions.create(
                model=model_id,
                messages=req.messages,
                web_search=req.web_search,
                stream=True,
                think_harder=req.think_harder,
            )
            chunk_count = 0
            async for chunk in response:
                if not chunk.choices:
                    continue
                chunk_count += 1
                d = chunk.choices[0].delta
                if d.thinking:
                    payload = json_encode.encode(
                        {"idx": idx, "type": "thinking", "content": d.thinking}
                    )
                else:
                    payload = json_encode.encode(
                        {"idx": idx, "type": "content", "content": d.content}
                    )
                writer.write(b"data: " + payload + b"\n\n")
                await writer.drain()
            log.info("compare[%d] %s done: %d chunks", idx, model_id, chunk_count)
        except Exception as exc:
            log.error(
                "compare[%d] %s failed: %s\n%s",
                idx,
                model_id,
                exc,
                traceback.format_exc(),
            )
            payload = json_encode.encode(
                {"idx": idx, "type": "error", "content": f"{type(exc).__name__}: {exc}"}
            )
            writer.write(b"data: " + payload + b"\n\n")
            await writer.drain()

        payload = json_encode.encode({"idx": idx, "type": "done"})
        writer.write(b"data: " + payload + b"\n\n")
        await writer.drain()

    # Run all model streams concurrently
    try:
        await asyncio.gather(*[_stream_model(m, i) for i, m in enumerate(req.models)])
    except Exception:
        log.error("compare gather failed:\n%s", traceback.format_exc())

    writer.write(b"data: [DONE]\n\n")
    await writer.drain()
    writer.close()


async def _handle_audio(writer: asyncio.StreamWriter, body: bytes) -> None:
    try:
        req = json_decode(body, type=AudioRequest)
    except Exception:
        _write_json(writer, 400, {"error": "invalid request"})
        await writer.drain()
        return

    if not req.input:
        _write_json(writer, 400, {"error": "input is empty"})
        await writer.drain()
        return

    client = _get_client()

    kwargs: dict = {"model": req.model, "input": req.input}
    if req.voice:
        kwargs["voice"] = req.voice
    if req.instructions:
        kwargs["instructions"] = req.instructions
    if req.format:
        kwargs["format"] = req.format

    try:
        result = await client.audio.speech.create(**kwargs)
    except Exception as exc:
        log.error("audio failed: %s", exc)
        _write_json(writer, 400, {"error": f"{type(exc).__name__}: {exc}"})
        await writer.drain()
        return

    if not result.data:
        _write_json(writer, 500, {"error": "no audio returned"})
        await writer.drain()
        return

    item = result.data[0]
    _write_response(writer, 200, item.mime_type, item.audio)
    await writer.drain()
    writer.close()


async def _handle_models(writer: asyncio.StreamWriter) -> None:
    _write_json(writer, 200, _build_model_list())
    await writer.drain()


async def _handle_html(writer: asyncio.StreamWriter) -> None:
    _write_response(writer, 200, "text/html; charset=utf-8", _load_html())
    await writer.drain()
    writer.close()


async def _handle_request(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> None:
    try:
        request_line = await asyncio.wait_for(reader.readline(), timeout=30)
        if not request_line:
            writer.close()
            return

        parts = request_line.decode("utf-8", errors="replace").strip().split()
        if len(parts) < 2:
            writer.close()
            return

        method, path = parts[0], parts[1]

        headers: dict[str, str] = {}
        while True:
            header_line = await asyncio.wait_for(reader.readline(), timeout=10)
            if header_line in (b"\r\n", b"\n", b""):
                break
            decoded = header_line.decode("utf-8", errors="replace").strip()
            if ":" in decoded:
                k, v = decoded.split(":", 1)
                headers[k.strip().lower()] = v.strip()

        body = b""
        content_length = int(headers.get("content-length", 0))
        if content_length > 0:
            body = await reader.readexactly(content_length)

        if method == "OPTIONS":
            writer.write(
                b"HTTP/1.1 204 No Content\r\n"
                b"Access-Control-Allow-Origin: *\r\n"
                b"Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
                b"Access-Control-Allow-Headers: Content-Type\r\n\r\n"
            )
            await writer.drain()
            writer.close()
            return

        if path == "/api/chat" and method == "POST":
            await _handle_chat(writer, body)
        elif path == "/api/chat/compare" and method == "POST":
            await _handle_compare(writer, body)
        elif path == "/api/images" and method == "POST":
            await _handle_images(writer, body)
        elif path == "/api/audio" and method == "POST":
            await _handle_audio(writer, body)
        elif path == "/api/models" and method == "GET":
            await _handle_models(writer)
        elif path in ("/", "/index.html"):
            await _handle_html(writer)
        else:
            _write_json(writer, 404, {"error": "not found"})
            await writer.drain()
            writer.close()
    except Exception:
        try:
            writer.close()
        except Exception:
            pass


async def Run(host: str = "127.0.0.1", port: int = 8000) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    server = await asyncio.start_server(_handle_request, host, port)
    addr = server.sockets[0].getsockname()
    print(f"  fishr chat  ->  http://{addr[0]}:{addr[1]}")
    print(f"  Press Ctrl+C to stop\n")
    async with server:
        await server.serve_forever()
