from time import time

from fishr.Base.DeepAI import DeepAI, DeepAIStream
from fishr.Base.Raphael import Raphael
from fishr.Types import ImageResponse, ImageUrl


class Images:
    """Generate images via ``client.images.generate(...)``.

    Routes ``deepai/*`` to DeepAI's text2img API and ``raphael/*`` to
    raphael.app's generate-image endpoint.

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

    __slots__ = ("deepai", "raphael")

    def __init__(self, deepai: DeepAI, raphael: Raphael) -> None:
        self.deepai = deepai
        self.raphael = raphael

    def generate(
        self,
        *,
        model: str = "deepai/image",
        prompt: str,
        number_of_images: int = 1,
        aspect: str = "1:1",
        resolution: str = "0.5k",
        quality: str = "low",
        fast_mode: bool = False,
        high_quality: bool = False,
        image: dict | None = None,
    ) -> ImageResponse:
        if model.startswith("raphael/"):
            result = self.raphael.generate(
                prompt,
                model=model,
                number_of_images=number_of_images,
                aspect=aspect,
                resolution=resolution,
                quality=quality,
                fast_mode=fast_mode,
                high_quality=high_quality,
                image=image,
            )
            data = tuple(ImageUrl(url=u, alt=prompt) for u in result.images)
            return ImageResponse(created=int(time()), data=data)
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

    __slots__ = ("deepai", "raphael")

    def __init__(self, deepai: DeepAI, raphael: Raphael) -> None:
        self.deepai = deepai
        self.raphael = raphael

    async def generate(
        self,
        *,
        model: str = "deepai/image",
        prompt: str,
        number_of_images: int = 1,
        aspect: str = "1:1",
        resolution: str = "0.5k",
        quality: str = "low",
        fast_mode: bool = False,
        high_quality: bool = False,
        image: dict | None = None,
    ) -> ImageResponse:
        from fishr.Loop import asyncio

        if model.startswith("raphael/"):
            result = await asyncio.to_thread(
                self.raphael.generate,
                prompt,
                model=model,
                number_of_images=number_of_images,
                aspect=aspect,
                resolution=resolution,
                quality=quality,
                fast_mode=fast_mode,
                high_quality=high_quality,
                image=image,
            )
            data = tuple(ImageUrl(url=u, alt=prompt) for u in result.images)
            return ImageResponse(created=int(time()), data=data)
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
