from __future__ import annotations

import logging
from typing import AsyncIterator

from msgspec import Struct

from fishr.Http import make_client
from fishr.Loop import asyncio
from fishr.Utils import json_encode

Log = logging.getLogger("fishr.audio.make")

ApiUrl = "https://musicmake.ai/api/qwen3tts/generate"

# Public model ids. Each maps to an upstream voice; the real names are
# kept private in ``_VOICE_MAP`` and never exposed.
Voices = (
    "aura",
    "breeze",
    "cypress",
    "drift",
    "echo",
    "flare",
    "gem",
    "hazel",
    "ivy",
    "jazz",
    "kite",
    "lumen",
    "mist",
    "saffron",
    "solstice",
    "pearl",
    "quartz",
    "ripple",
    "cobalt",
    "tide",
    "vale",
    "wren",
    "ash",
    "brook",
    "cedar",
    "dawn",
    "fern",
    "glen",
    "harbor",
    "indigo",
    "juniper",
    "lotus",
    "maple",
    "nettle",
    "opal",
    "pine",
    "river",
    "slate",
    "willow",
)

_VOICE_MAP = {
    "aura": "Cherry",
    "breeze": "Serena",
    "cypress": "Ethan",
    "drift": "Chelsie",
    "echo": "Momo",
    "flare": "Vivian",
    "gem": "Moon",
    "hazel": "Maia",
    "ivy": "Kai",
    "jazz": "Nolish",
    "kite": "Bella",
    "lumen": "Jennifer",
    "mist": "Ryan",
    "saffron": "Katerina",
    "solstice": "Aiden",
    "pearl": "Eldric Sage",
    "quartz": "Mia",
    "ripple": "Mochi",
    "cobalt": "Bellona",
    "tide": "Vincent",
    "vale": "Bunny",
    "wren": "Neil",
    "ash": "Elias",
    "brook": "Arthur",
    "cedar": "Nini",
    "dawn": "Ebona",
    "fern": "Soren",
    "glen": "Pip",
    "harbor": "Stella",
    "indigo": "Bodega",
    "juniper": "Sonrisa",
    "lotus": "Alek",
    "maple": "Dolce",
    "nettle": "Suhee",
    "opal": "Ono Anna",
    "pine": "Lenn",
    "river": "Emilien",
    "slate": "Andre",
    "willow": "Radio Gol",
}

DefaultVoice = "aura"
DefaultMode = "system"

Headers = {
    "accept": "*/*",
    "content-type": "application/json",
    "origin": "https://musicmake.ai",
    "referer": "https://musicmake.ai/qwen3tts",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36",
}


def ResolveVoice(Model: str, Default: str = DefaultVoice) -> str:
    """Resolve a ``make/<id>`` model string to a public voice id."""
    Base = Model.split("/", 1)[-1] if "/" in Model else Model
    if Base in _VOICE_MAP:
        return Base
    return Default


def _upstream_voice(VoiceId: str) -> str:
    """Map a public voice id to the upstream voice name."""
    return _VOICE_MAP.get(VoiceId, _VOICE_MAP[DefaultVoice])


class MusicMakeResponse(Struct, frozen=True):
    voice: str
    model: str
    audio: bytes = b""
    mime_type: str = "audio/wav"


class MusicMakeStream:
    """Async-only binary stream of the raw audio response body."""

    __slots__ = ("Bytes", "Voice", "Model", "MimeType")

    def __init__(
        self, Bytes: bytes, Voice: str, Model: str, MimeType: str = "audio/wav"
    ) -> None:
        self.Bytes = Bytes
        self.Voice = Voice
        self.Model = Model
        self.MimeType = MimeType

    async def __aiter__(self) -> AsyncIterator[bytes]:
        if self.Bytes:
            yield self.Bytes

    async def aread(self) -> bytes:
        return self.Bytes


class MusicMake:
    """Qwen3 TTS provider.

    Models are specified as ``make/<voice>`` — 39 voices available:
    ``make/aura``, ``make/breeze``, ``make/cypress``, ... ``make/willow``.

    Usage::

        ```py
        from fishr import MusicMake

        mm = MusicMake()

        # default voice (aura)
        result = mm.speak("Hello world")
        print(result.voice, len(result.audio))

        # explicit voice
        result = mm.speak("Hello world", model="make/breeze")
        ```
    """

    __slots__ = ("HttpClient",)

    def __init__(self) -> None:
        self.HttpClient = make_client(headers=Headers)

    def speak(
        self,
        Prompt: str,
        *,
        model: str = "make/aura",
        voice: str | None = None,
        stream: bool = False,
    ) -> MusicMakeResponse | MusicMakeStream:
        VoiceId = ResolveVoice(model if voice is None else voice)
        Upstream = _upstream_voice(VoiceId)
        Payload = {
            "text": Prompt,
            "voice": Upstream,
            "mode": DefaultMode,
        }
        Body = json_encode.encode(Payload)

        Resp = self.HttpClient.post(ApiUrl, content=Body, headers=Headers, timeout=600)
        if Resp.status_code >= 400:
            Log.warning(
                "MusicMake request failed: %s %s",
                Resp.status_code,
                Resp.text[:200] if hasattr(Resp, "text") else b"",
            )
            if stream:
                return MusicMakeStream(b"", VoiceId, VoiceId)
            return MusicMakeResponse(voice=VoiceId, model=VoiceId, audio=b"")
        Audio = Resp.content if hasattr(Resp, "content") else b""
        if stream:
            return MusicMakeStream(Audio, VoiceId, VoiceId)
        return MusicMakeResponse(voice=VoiceId, model=VoiceId, audio=Audio)

    async def speak_async(
        self,
        Prompt: str,
        *,
        model: str = "make/aura",
        voice: str | None = None,
        stream: bool = False,
    ) -> MusicMakeResponse | MusicMakeStream:
        return await asyncio.to_thread(
            self.speak,
            Prompt,
            model=model,
            voice=voice,
            stream=stream,
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
    "MusicMake",
    "MusicMakeResponse",
    "MusicMakeStream",
    "Voices",
    "DefaultVoice",
    "ApiUrl",
    "ResolveVoice",
]
