from time import time

from fishr.Base.DeepAI import DeepAI, DeepAIStream
from fishr.Types import ImageResponse, ImageUrl


class Images:
    """Generate images via ``client.images.generate(...)``.

    Uses DeepAI's text2img API under the hood.

    Usage::

        ```py
        from fishr import Client

        client = Client()
        result = client.images.generate(
            model="deepai/image",
            prompt="A cat riding a skateboard",
        )
        print(result.data[0].url)```
    """

    __slots__ = ("deepai",)

    def __init__(self, deepai: DeepAI) -> None:
        self.deepai = deepai

    def generate(
        self,
        *,
        model: str = "deepai/image",
        prompt: str,
    ) -> ImageResponse:
        result = self.deepai.ask(prompt, model=model)
        if isinstance(result, DeepAIStream):
            url = ""
        else:
            url = result.image_url or ""
        return ImageResponse(
            created=int(time()),
            data=(ImageUrl(url=url, alt=prompt),),
        )


class AsyncImages:
    """Async version of :class:`Images`.

    Usage::

        ```py
        from fishr import AsyncClient

        client = AsyncClient()
        result = await client.images.generate(
            model="deepai/image",
            prompt="A cat riding a skateboard",
        )
        print(result.data[0].url)```
    """

    __slots__ = ("deepai",)

    def __init__(self, deepai: DeepAI) -> None:
        self.deepai = deepai

    async def generate(
        self,
        *,
        model: str = "deepai/image",
        prompt: str,
    ) -> ImageResponse:
        from fishr.Loop import asyncio

        result = await asyncio.to_thread(
            self.deepai.ask,
            prompt,
            model=model,
        )
        if isinstance(result, DeepAIStream):
            url = ""
        else:
            url = result.image_url or ""
        return ImageResponse(
            created=int(time()),
            data=(ImageUrl(url=url, alt=prompt),),
        )


__all__ = [
    "Images",
    "AsyncImages",
]
