from __future__ import annotations

import logging
from typing import AsyncIterator

from msgspec import Struct

from fishr.Http import make_client
from fishr.Loop import asyncio
from fishr.Utils import json_encode

Log = logging.getLogger("fishr.audio.make")

ApiUrl = "https://musicmake.ai/api/qwen3tts/generate"

# Voices confirmed working on the Qwen3 TTS endpoint.
Voices = (
    "Cherry",
    "Amber",
    "Bella",
    "Clara",
    "Daisy",
    "Emma",
    "Fiona",
    "Grace",
    "Hazel",
    "Ivy",
    "Jade",
    "Kai",
    "Lily",
    "Mia",
    "Nina",
    "Olive",
    "Pearl",
    "Quinn",
    "Rose",
    "Skye",
)

DefaultVoice = "Cherry"
DefaultMode = "system"

Headers = {
    "accept": "*/*",
    "content-type": "application/json",
    "origin": "https://musicmake.ai",
    "referer": "https://musicmake.ai/qwen3tts",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36",
}


def ResolveVoice(VoiceName: str, Default: str = DefaultVoice) -> str:
    """Resolve a ``make/<voice>`` model string to a voice name."""
    Base = VoiceName.split("/", 1)[-1] if "/" in VoiceName else VoiceName
    if Base in Voices:
        return Base
    return Base or Default


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
    """MusicMake.ai Qwen3 TTS provider.

    Models are specified as ``make/<voice>``:

    - voices: ``Cherry``, ``Amber``, ``Bella``, ``Clara``, ``Daisy``,
      ``Emma``, ``Fiona``, ``Grace``, ``Hazel``, ``Ivy``, ``Jade``,
      ``Kai``, ``Lily``, ``Mia``, ``Nina``, ``Olive``, ``Pearl``,
      ``Quinn``, ``Rose``, ``Skye``

    Usage::

        ```py
        from fishr import MusicMake

        mm = MusicMake()

        # default voice (Cherry)
        result = mm.speak("Hello world")
        print(result.voice, len(result.audio))

        # explicit voice
        result = mm.speak("Hello world", voice="Amber")

        # using model string
        result = mm.speak("Hello world", model="make/Amber")
        ```
    """

    __slots__ = ("HttpClient",)

    def __init__(self) -> None:
        self.HttpClient = make_client(headers=Headers)

    def speak(
        self,
        Prompt: str,
        *,
        model: str = "make/Cherry",
        voice: str | None = None,
        stream: bool = False,
    ) -> MusicMakeResponse | MusicMakeStream:
        Voice = ResolveVoice(model if voice is None else voice)
        Payload = {
            "text": Prompt,
            "voice": Voice,
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
                return MusicMakeStream(b"", Voice, Voice)
            return MusicMakeResponse(voice=Voice, model=Voice, audio=b"")
        Audio = Resp.content if hasattr(Resp, "content") else b""
        if stream:
            return MusicMakeStream(Audio, Voice, Voice)
        return MusicMakeResponse(voice=Voice, model=Voice, audio=Audio)

    async def speak_async(
        self,
        Prompt: str,
        *,
        model: str = "make/Cherry",
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
