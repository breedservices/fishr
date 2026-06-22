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
        for line in resp.iter_lines():
            decoded = line.decode(errors="replace") if isinstance(line, bytes) else line
            queue.put_nowait(decoded)
        queue.put_nowait(None)

    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, sync_consume)

    while True:
        line = await queue.get()
        if line is None:
            break
        yield line


__all__ = [
    "json_encode",
    "json_decode",
    "build_multipart",
    "aiter_lines",
]
