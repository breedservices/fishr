from __future__ import annotations

import logging
from typing import AsyncIterator

from msgspec import Struct

from fishr.Http import make_client
from fishr.Loop import asyncio
from fishr.Utils import json_encode

Log = logging.getLogger("fishr.audio.ta")

TtsUrl = "https://telnyx.com/api/tts-demo"

TELNYX_TTS_MODELS = {
    "telnyx-tts/astra": "astra",
    "telnyx-tts/luna": "luna",
    "telnyx-tts/sol": "sol",
    "telnyx-tts/nova": "nova",
    "telnyx-tts/orion": "orion",
}
DEFAULT_MODEL = "telnyx-tts/astra"
DEFAULT_VOICE = "astra"
UpstreamPrefix = "Telnyx.NaturalHD"

# Voices confirmed working on the no-auth demo endpoint (all NaturalHD).
WorkingVoices = (
    "astra",
    "luna",
    "sol",
    "nova",
    "orion",
)

Headers = {
    "accept": "*/*",
    "content-type": "application/json",
    "origin": "https://telnyx.com",
    "referer": "https://telnyx.com/",
}


def ResolveModel(Model: str) -> str:
    if Model in TELNYX_TTS_MODELS:
        return Model
    if Model.startswith("telnyx-tts/"):
        Base = Model.split("/", 1)[1]
        if Base:
            return Model
    return DEFAULT_MODEL


def ResolveVoice(Model: str, Voice: str | None) -> str:
    Resolved = ResolveModel(Model)
    VoiceId = Voice or TELNYX_TTS_MODELS.get(Resolved) or Resolved.split("/", 1)[1]
    if VoiceId.startswith("Telnyx."):
        return VoiceId
    return f"{UpstreamPrefix}.{VoiceId}"


def _ContentType(Resp) -> str:
    Hs = getattr(Resp, "headers", None)
    if not Hs:
        return "audio/mpeg"
    for Key in ("content-type", "Content-Type"):
        Value = Hs.get(Key)
        if Value:
            return Value.split(";", 1)[0].strip()
    return "audio/mpeg"


class TelnyxAudioResponse(Struct, frozen=True):
    voice: str = ""
    model: str = ""
    audio: bytes = b""
    mime_type: str = "audio/mpeg"


class TelnyxAudioStream:
    """Async-only binary stream of the raw audio response body."""

    __slots__ = ("Bytes", "Voice", "Model", "MimeType")

    def __init__(
        self, Bytes: bytes, Voice: str, Model: str, MimeType: str = "audio/mpeg"
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


class TelnyxAudio:
    """
    Usage::

        ```py
        from fishr import TelnyxAudio

        ta = TelnyxAudio()

        # default model is telnyx-tts/astra (voice: astra)
        result = ta.speak("Hello world.")
        print(result.model, result.voice, len(result.audio))

        # pick a voice by passing its model id
        result = ta.speak("Hello world.", model="telnyx-tts/luna")

        # override the voice on top of any model id
        result = ta.speak("Hello world.", model="telnyx-tts/astra", voice="orion")```
    """

    __slots__ = ("HttpClient",)

    def __init__(self) -> None:
        self.HttpClient = make_client(headers=Headers)

    def speak(
        self,
        Prompt: str,
        *,
        model: str = DEFAULT_MODEL,
        voice: str | None = None,
        language: str | None = None,
        text_type: str | None = None,
        stream: bool = False,
    ) -> TelnyxAudioResponse | TelnyxAudioStream:
        Resolved = ResolveModel(model)
        FullVoice = ResolveVoice(Resolved, voice)
        UpstreamModel = UpstreamPrefix
        Payload = {
            "text": Prompt,
            "voice": FullVoice,
            "language": language or "en-US",
            "output_type": "binary_output",
        }
        if text_type:
            Payload["text_type"] = text_type
        Body = json_encode.encode(Payload)
        Resp = self.HttpClient.post(TtsUrl, content=Body, headers=Headers, timeout=600)
        if Resp.status_code >= 400:
            Log.warning(
                "telnyx tts request failed: %s %s",
                Resp.status_code,
                Resp.text[:200] if hasattr(Resp, "text") else b"",
            )
            Mime = "audio/mpeg"
            if stream:
                return TelnyxAudioStream(b"", FullVoice, UpstreamModel, Mime)
            return TelnyxAudioResponse(
                voice=FullVoice,
                model=UpstreamModel,
                audio=b"",
                mime_type=Mime,
            )
        Mime = _ContentType(Resp)
        Audio = Resp.content if hasattr(Resp, "content") else b""
        if stream:
            return TelnyxAudioStream(Audio, FullVoice, UpstreamModel, Mime)
        return TelnyxAudioResponse(
            voice=FullVoice,
            model=UpstreamModel,
            audio=Audio,
            mime_type=Mime,
        )

    async def speak_async(
        self,
        Prompt: str,
        *,
        model: str = DEFAULT_MODEL,
        voice: str | None = None,
        language: str = "en-US",
        text_type: str | None = None,
        stream: bool = False,
    ) -> TelnyxAudioResponse | TelnyxAudioStream:
        return await asyncio.to_thread(
            self.speak,
            Prompt,
            model=model,
            voice=voice,
            language=language,
            text_type=text_type,
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
    "TelnyxAudio",
    "TelnyxAudioResponse",
    "TelnyxAudioStream",
    "TELNYX_TTS_MODELS",
    "DEFAULT_MODEL",
    "DEFAULT_VOICE",
    "ResolveModel",
    "ResolveVoice",
]
