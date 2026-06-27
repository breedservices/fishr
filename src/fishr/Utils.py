from random import choices
from string import ascii_letters, digits
from typing import AsyncIterator

from msgspec.json import Decoder as JsonDecoder
from msgspec.json import Encoder as JsonEncoder

json_encode = JsonEncoder()
json_decode = JsonDecoder()


def build_multipart(fields: dict[str, str]) -> tuple[bytes, str]:
    boundary = "----WebKitFormBoundary" + "".join(
        choices(ascii_letters + digits, k=16),
    )
    parts: list[bytes] = []
    for k, v in fields.items():
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode(),
        )
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), boundary


async def aiter_lines(resp) -> AsyncIterator[str]:
    from fishr.Loop import asyncio

    queue: asyncio.Queue[str | None] = asyncio.Queue()

    def sync_consume():
        try:
            for line in resp.iter_lines():
                decoded = (
                    line.decode(errors="replace") if isinstance(line, bytes) else line
                )
                queue.put_nowait(decoded)
        except Exception as exc:
            queue.put_nowait(exc)
        queue.put_nowait(None)

    loop = asyncio.get_running_loop()
    future = loop.run_in_executor(None, sync_consume)
    future.add_done_callback(lambda f: f.exception() if not f.cancelled() else None)

    while True:
        line = await queue.get()
        if line is None:
            break
        if isinstance(line, Exception):
            raise line
        yield line


async def aiter_bytes(resp, chunk_size: int = 8192) -> AsyncIterator[bytes]:
    from fishr.Loop import asyncio

    queue: asyncio.Queue[bytes | None | Exception] = asyncio.Queue()

    def sync_consume():
        try:
            if hasattr(resp, "iter_chunks"):
                for chunk in resp.iter_chunks():
                    data = chunk[0] if isinstance(chunk, tuple) else chunk
                    if data:
                        queue.put_nowait(data)
            elif hasattr(resp, "content"):
                body = resp.content or b""
                if body:
                    for i in range(0, len(body), chunk_size):
                        queue.put_nowait(body[i : i + chunk_size])
        except Exception as exc:
            queue.put_nowait(exc)
        finally:
            queue.put_nowait(None)

    loop = asyncio.get_running_loop()
    future = loop.run_in_executor(None, sync_consume)
    future.add_done_callback(lambda f: f.exception() if not f.cancelled() else None)

    while True:
        item = await queue.get()
        if item is None:
            break
        if isinstance(item, Exception):
            raise item
        yield item


__all__ = [
    "json_encode",
    "json_decode",
    "build_multipart",
    "aiter_lines",
    "aiter_bytes",
]
