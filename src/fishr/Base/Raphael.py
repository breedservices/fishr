from __future__ import annotations

import logging
from uuid import uuid4

from msgspec import Struct

from fishr.Http import make_client
from fishr.Loop import asyncio
from fishr.Utils import json_decode, json_encode

Log = logging.getLogger("fishr.raphael")

IMAGE_URL = "https://raphael.app/api/generate-image"
Base = "https://raphael.app"

RAPHAEL_MODELS = {
    "raphael/image": "raphael-basic",
}
DEFAULT_MODEL = "raphael/image"

AspectRatios = ("1:1", "4:3", "3:4", "16:9", "9:16", "auto")
Resolutions = ("0.5k", "1k", "2k")
Qualities = ("low", "medium", "high")

AllowedImageMimes = frozenset(("image/webp", "image/png", "image/jpeg", "image/jpg"))

Headers = {
    "accept": "*/*",
    "accept-language": "en-GB,en-US;q=0.9,en;q=0.8",
    "content-type": "application/json",
    "origin": "https://raphael.app",
    "referer": "https://raphael.app/",
    "cache-control": "no-cache",
    "pragma": "no-cache",
    "dnt": "1",
}


def ResolveModel(Model: str) -> str:
    BaseName = Model.split("/", 1)[-1] if "/" in Model else Model
    Key = f"raphael/{BaseName}"
    if Key in RAPHAEL_MODELS:
        return Key
    return DEFAULT_MODEL


def _ToDataUri(Image: dict) -> str:
    Mime = Image.get("mime_type", "image/webp")
    Data = Image.get("base64_data", "")
    return f"data:{Mime};base64,{Data}"


def _BuildPayload(
    Prompt: str,
    Model: str,
    NumberOfImages: int,
    Aspect: str,
    Resolution: str,
    Quality: str,
    FastMode: bool,
    HighQuality: bool,
    InputImage: str | None = None,
) -> bytes:
    ModelId = RAPHAEL_MODELS.get(Model, RAPHAEL_MODELS[DEFAULT_MODEL])
    Action = "img2img" if InputImage else "generate"
    Payload = {
        "prompt": Prompt,
        "enhanced_prompt": Prompt,
        "action": Action,
        "aspect": Aspect,
        "size": Aspect,
        "autoTranslate": True,
        "client_request_id": str(uuid4()),
        "entry_type": "ai-image",
        "fastMode": FastMode,
        "highQuality": HighQuality,
        "isSafeContent": True,
        "model_id": ModelId,
        "number_of_images": NumberOfImages,
        "quality": Quality,
        "resolution": Resolution,
        "turnstileToken": None,
    }
    if InputImage:
        Payload["input_image"] = InputImage
        Payload["input_image_list"] = (InputImage,)
    return json_encode.encode(Payload)


def _ParseUrls(Body: bytes | str) -> tuple[str, ...]:
    Text = Body.decode(errors="ignore") if isinstance(Body, bytes) else Body
    Urls: list[str] = []
    for Line in Text.splitlines():
        Line = Line.strip()
        if not Line:
            continue
        try:
            Obj = json_decode.decode(Line)
        except Exception:
            continue
        if not isinstance(Obj, dict):
            continue
        Url = Obj.get("url") or ""
        if not Url:
            continue
        if Url.startswith("/"):
            Url = f"{Base}{Url}"
        Urls.append(Url)
    return tuple(Urls)


class RaphaelResponse(Struct, frozen=True):
    images: tuple[str, ...] = ()


class Raphael:
    """
    Usage::

        ```py
        from fishr import Raphael

        r = Raphael()
        result = r.generate("A cat riding a skateboard")
        print(result.images)```
    """

    __slots__ = ("HttpClient",)

    def __init__(self) -> None:
        self.HttpClient = make_client(headers=Headers)

    def generate(
        self,
        Prompt: str,
        *,
        model: str = DEFAULT_MODEL,
        number_of_images: int = 1,
        aspect: str = "1:1",
        resolution: str = "0.5k",
        quality: str = "low",
        fast_mode: bool = False,
        high_quality: bool = False,
        image: dict | None = None,
    ) -> RaphaelResponse:
        Resolved = ResolveModel(model)
        if aspect not in AspectRatios:
            aspect = "1:1"
        if resolution not in Resolutions:
            resolution = "0.5k"
        if quality not in Qualities:
            quality = "low"
        InputImage = None
        if isinstance(image, dict):
            Mime = image.get("mime_type")
            if Mime in AllowedImageMimes and image.get("base64_data"):
                InputImage = _ToDataUri(image)
        Body = _BuildPayload(
            Prompt,
            Resolved,
            number_of_images,
            aspect,
            resolution,
            quality,
            fast_mode,
            high_quality,
            InputImage,
        )
        Resp = self.HttpClient.post(
            IMAGE_URL, content=Body, headers=Headers, timeout=600
        )
        if Resp.status_code >= 400:
            Log.warning(
                "raphael request failed: %s %s",
                Resp.status_code,
                Resp.text[:200] if hasattr(Resp, "text") else b"",
            )
            return RaphaelResponse(images=())
        Raw = Resp.content if hasattr(Resp, "content") else b""
        Urls = _ParseUrls(Raw)
        return RaphaelResponse(images=Urls)

    async def generate_async(
        self,
        Prompt: str,
        *,
        model: str = DEFAULT_MODEL,
        number_of_images: int = 1,
        aspect: str = "1:1",
        resolution: str = "0.5k",
        quality: str = "low",
        fast_mode: bool = False,
        high_quality: bool = False,
        image: dict | None = None,
    ) -> RaphaelResponse:
        return await asyncio.to_thread(
            self.generate,
            Prompt,
            model=model,
            number_of_images=number_of_images,
            aspect=aspect,
            resolution=resolution,
            quality=quality,
            fast_mode=fast_mode,
            high_quality=high_quality,
            image=image,
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
    "Raphael",
    "RaphaelResponse",
    "ResolveModel",
    "RAPHAEL_MODELS",
    "DEFAULT_MODEL",
]
