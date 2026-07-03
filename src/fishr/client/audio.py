from __future__ import annotations

from time import time

from fishr.audio.MusicMake import (
    MusicMake,
    MusicMakeResponse,
)
from fishr.audio.OpenAIFM import (
    OpenAIFM,
    OpenAIFMResponse,
)
from fishr.audio.TelnyxAudio import (
    TelnyxAudio,
    TelnyxAudioResponse,
)
from fishr.audio.Transcode import AudioFormat, Transcode
from fishr.Types import AudioData, AudioResponse


def _is_telnyx_tts(Model: str) -> bool:
    return Model.startswith("telnyx-tts/")


def _is_musicmake(Model: str) -> bool:
    return Model.startswith("make/")


class Speech:
    """Generate speech via ``client.audio.speech.create(...)``.

    Routes to:
    - OpenAI.fm for ``fm/*`` models
    - Telnyx no-auth TTS for ``telnyx-tts/*`` models
    - MusicMake for ``make/*`` models

    ``format`` is optional. OpenAI.fm emits MP3, Telnyx emits MP3, and
    MusicMake emits WAV — other formats are produced via the bundled
    ``static-ffmpeg`` binary.

    Usage::

        ```py
        from fishr import Client

        client = Client()
        result = client.audio.speech.create(
            model="fm/coral",
            input="Hello world",
            voice="coral",
        )
        print(result.data[0].voice)```
    """

    __slots__ = ("OpenAIFm", "Telnyx", "MusicMake")

    def __init__(
        self, OpenAIFm: OpenAIFM, Telnyx: TelnyxAudio, MusicMake: MusicMake
    ) -> None:
        self.OpenAIFm = OpenAIFm
        self.Telnyx = Telnyx
        self.MusicMake = MusicMake

    def create(
        self,
        *,
        model: str = "fm/coral",
        input: str,
        voice: str | None = None,
        instructions: str | None = None,
        format: AudioFormat | None = None,
    ) -> AudioResponse:
        if _is_telnyx_tts(model):
            Res = self.Telnyx.speak(input, model=model, voice=voice, stream=False)
            Audio = Res.audio if isinstance(Res, TelnyxAudioResponse) else b""
            Mime = (
                Res.mime_type if isinstance(Res, TelnyxAudioResponse) else "audio/mpeg"
            )
            Voice = Res.voice if isinstance(Res, TelnyxAudioResponse) else (voice or "")
            ModelName = Res.model if isinstance(Res, TelnyxAudioResponse) else model
            if format is not None and format != "mp3":
                Transcoded = Transcode(Audio, format, SourceMime=Mime)
                if Transcoded is not None:
                    Audio, Mime = Transcoded
        elif _is_musicmake(model):
            Res = self.MusicMake.speak(input, model=model, voice=voice, stream=False)
            Audio = Res.audio if isinstance(Res, MusicMakeResponse) else b""
            Mime = Res.mime_type if isinstance(Res, MusicMakeResponse) else "audio/wav"
            Voice = Res.voice if isinstance(Res, MusicMakeResponse) else (voice or "")
            ModelName = Res.model if isinstance(Res, MusicMakeResponse) else model
            if format is not None and format != "wav":
                Transcoded = Transcode(Audio, format, SourceMime=Mime)
                if Transcoded is not None:
                    Audio, Mime = Transcoded
        else:
            Res = self.OpenAIFm.speak(
                input,
                model=model,
                voice=voice,
                instructions=instructions,
                stream=False,
            )
            Audio = Res.audio if isinstance(Res, OpenAIFMResponse) else b""
            Mime = Res.mime_type if isinstance(Res, OpenAIFMResponse) else "audio/mpeg"
            Voice = Res.voice if isinstance(Res, OpenAIFMResponse) else (voice or "")
            ModelName = Res.model if isinstance(Res, OpenAIFMResponse) else model
            if format is not None and format != "mp3":
                Transcoded = Transcode(Audio, format, SourceMime=Mime)
                if Transcoded is not None:
                    Audio, Mime = Transcoded
        return AudioResponse(
            created=int(time()),
            data=(
                AudioData(
                    voice=Voice,
                    model=ModelName,
                    audio=Audio,
                    mime_type=Mime,
                ),
            ),
        )


class AsyncSpeech:
    """Async version of :class:`Speech`.

    Usage::

        ```py
        from fishr import AsyncClient

        client = AsyncClient()
        result = await client.audio.speech.create(
            model="fm/coral",
            input="Hello world",
            voice="coral",
        )
        print(result.data[0].voice)```
    """

    __slots__ = ("OpenAIFm", "Telnyx", "MusicMake")

    def __init__(
        self, OpenAIFm: OpenAIFM, Telnyx: TelnyxAudio, MusicMake: MusicMake
    ) -> None:
        self.OpenAIFm = OpenAIFm
        self.Telnyx = Telnyx
        self.MusicMake = MusicMake

    async def create(
        self,
        *,
        model: str = "fm/coral",
        input: str,
        voice: str | None = None,
        instructions: str | None = None,
        format: AudioFormat | None = None,
    ) -> AudioResponse:
        from fishr.Loop import asyncio

        if _is_telnyx_tts(model):
            Res = await asyncio.to_thread(
                self.Telnyx.speak,
                input,
                model=model,
                voice=voice,
                stream=False,
            )
            Audio = Res.audio if isinstance(Res, TelnyxAudioResponse) else b""
            Mime = (
                Res.mime_type if isinstance(Res, TelnyxAudioResponse) else "audio/mpeg"
            )
            Voice = Res.voice if isinstance(Res, TelnyxAudioResponse) else (voice or "")
            ModelName = Res.model if isinstance(Res, TelnyxAudioResponse) else model
            if format is not None and format != "mp3":
                Transcoded = await asyncio.to_thread(
                    Transcode, Audio, format, SourceMime=Mime
                )
                if Transcoded is not None:
                    Audio, Mime = Transcoded
        elif _is_musicmake(model):
            Res = await asyncio.to_thread(
                self.MusicMake.speak,
                input,
                model=model,
                voice=voice,
                stream=False,
            )
            Audio = Res.audio if isinstance(Res, MusicMakeResponse) else b""
            Mime = Res.mime_type if isinstance(Res, MusicMakeResponse) else "audio/wav"
            Voice = Res.voice if isinstance(Res, MusicMakeResponse) else (voice or "")
            ModelName = Res.model if isinstance(Res, MusicMakeResponse) else model
            if format is not None and format != "wav":
                Transcoded = await asyncio.to_thread(
                    Transcode, Audio, format, SourceMime=Mime
                )
                if Transcoded is not None:
                    Audio, Mime = Transcoded
        else:
            Res = await asyncio.to_thread(
                self.OpenAIFm.speak,
                input,
                model=model,
                voice=voice,
                instructions=instructions,
                stream=False,
            )
            Audio = Res.audio if isinstance(Res, OpenAIFMResponse) else b""
            Mime = Res.mime_type if isinstance(Res, OpenAIFMResponse) else "audio/mpeg"
            Voice = Res.voice if isinstance(Res, OpenAIFMResponse) else (voice or "")
            ModelName = Res.model if isinstance(Res, OpenAIFMResponse) else model
            if format is not None and format != "mp3":
                Transcoded = await asyncio.to_thread(
                    Transcode, Audio, format, SourceMime=Mime
                )
                if Transcoded is not None:
                    Audio, Mime = Transcoded
        return AudioResponse(
            created=int(time()),
            data=(
                AudioData(
                    voice=Voice,
                    model=ModelName,
                    audio=Audio,
                    mime_type=Mime,
                ),
            ),
        )


class Audio:
    """Audio namespace exposing :class:`Speech`.

    Usage::

        ```py
        from fishr import Client

        client = Client()
        result = client.audio.speech.create(
            model="fm/coral",
            input="Hello world",
        )
        print(result.data[0].voice)```
    """

    __slots__ = ("speech",)

    def __init__(
        self, OpenAIFm: OpenAIFM, Telnyx: TelnyxAudio, MusicMake: MusicMake
    ) -> None:
        self.speech = Speech(OpenAIFm, Telnyx, MusicMake)


class AsyncAudio:
    """Async version of :class:`Audio`."""

    __slots__ = ("speech",)

    def __init__(
        self, OpenAIFm: OpenAIFM, Telnyx: TelnyxAudio, MusicMake: MusicMake
    ) -> None:
        self.speech = AsyncSpeech(OpenAIFm, Telnyx, MusicMake)


__all__ = [
    "Audio",
    "AsyncAudio",
    "Speech",
    "AsyncSpeech",
]
