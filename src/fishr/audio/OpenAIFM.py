from __future__ import annotations

import logging
from typing import AsyncIterator

from msgspec import Struct

from fishr.Http import make_client
from fishr.Loop import asyncio

Log = logging.getLogger("fishr.audio.fm")

ApiUrl = "https://www.openai.fm/api/generate"

Voices = (
    "alloy",
    "ash",
    "ballad",
    "cedar",
    "coral",
    "fable",
    "marin",
    "nova",
    "onyx",
    "sage",
    "verse",
)

DefaultVoice = "coral"

Styles = (
    "friendly",
    "patient_teacher",
    "noir_detective",
    "cowboy",
    "calm",
    "scientific_style",
)

Headers = {
    "accept": "*/*",
    "origin": "https://www.openai.fm",
    "referer": "https://www.openai.fm/",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36",
}

StylePrompts = {
    "friendly": """Affect/personality: A cheerful guide

Tone: Friendly, clear, and reassuring, creating a calm atmosphere and making the listener feel confident and comfortable.

Pronunciation: Clear, articulate, and steady, ensuring each instruction is easily understood while maintaining a natural, conversational flow.

Pause: Brief, purposeful pauses after key instructions (e.g., "cross the street" and "turn right") to allow time for the listener to process the information and follow along.

Emotion: Warm and supportive, conveying empathy and care, ensuring the listener feels guided and safe throughout the journey.""",
    "patient_teacher": """Accent/Affect: Warm, refined, and gently instructive, reminiscent of a friendly art instructor.

Tone: Calm, encouraging, and articulate, clearly describing each step with patience.

Pacing: Slow and deliberate, pausing often to allow the listener to follow instructions comfortably.

Emotion: Cheerful, supportive, and pleasantly enthusiastic; convey genuine enjoyment and appreciation of art.

Pronunciation: Clearly articulate artistic terminology (e.g., "brushstrokes," "landscape," "palette") with gentle emphasis.

Personality Affect: Friendly and approachable with a hint of sophistication; speak confidently and reassuringly, guiding users through each painting step patiently and warmly.""",
    "noir_detective": """Affect: a mysterious noir detective

Tone: Cool, detached, but subtly reassuring—like they've seen it all and know how to handle a missing package like it's just another case.

Delivery: Slow and deliberate, with dramatic pauses to build suspense, as if every detail matters in this investigation.

Emotion: A mix of world-weariness and quiet determination, with just a hint of dry humor to keep things from getting too grim.

Punctuation: Short, punchy sentences with ellipses and dashes to create rhythm and tension, mimicking the inner monologue of a detective piecing together clues.""",
    "cowboy": """Voice: Warm, relaxed, and friendly, with a steady cowboy drawl that feels approachable.

Punctuation: Light and natural, with gentle pauses that create a conversational rhythm without feeling rushed.

Delivery: Smooth and easygoing, with a laid-back pace that reassures the listener while keeping things clear.

Phrasing: Simple, direct, and folksy, using casual, familiar language to make technical support feel more personable.

Tone: Lighthearted and welcoming, with a calm confidence that puts the caller at ease.""",
    "calm": """Voice Affect: Calm, composed, and reassuring; project quiet authority and confidence.

Tone: Sincere, empathetic, and gently authoritative—express genuine apology while conveying competence.

Pacing: Steady and moderate; unhurried enough to communicate care, yet efficient enough to demonstrate professionalism.

Emotion: Genuine empathy and understanding; speak with warmth, especially during apologies ("I'm very sorry for any disruption...").

Pronunciation: Clear and precise, emphasizing key reassurances ("smoothly," "quickly," "promptly") to reinforce confidence.

Pauses: Brief pauses after offering assistance or requesting details, highlighting willingness to listen and support.""",
    "scientific_style": """Voice: Authoritative and precise, with a measured, academic tone.

Tone: Formal and analytical, maintaining objectivity while conveying complex information.

Pacing: Moderate and deliberate, allowing time for complex concepts to be processed.

Pronunciation: Precise articulation of technical terms and scientific vocabulary.

Pauses: Strategic pauses after introducing new concepts to allow for comprehension.

Emotion: Restrained enthusiasm for discoveries and findings, conveying intellectual curiosity.""",
}


def ResolveVoice(Model: str) -> str:
    """Resolve a model string into a voice name.

    A style name resolves to its prompt being used as instructions and the
    default voice; a voice name resolves to itself. Unknown names are
    passed through unchanged so the upstream API is the source of truth.
    """
    Base = Model.split("/", 1)[-1] if "/" in Model else Model
    if Base in Styles:
        return DefaultVoice
    if Base in Voices:
        return Base
    return Base or DefaultVoice


def ResolveInstructions(Model: str) -> str:
    """Resolve a model string into TTS instructions.

    If the model is a style name, the style's prompt is used; otherwise an
    empty string is returned so the caller can supply its own instructions.
    """
    Base = Model.split("/", 1)[-1] if "/" in Model else Model
    return StylePrompts.get(Base, "")


def ResolveModel(Model: str) -> tuple[str, str]:
    """Resolve a model string into ``(voice, instructions)``."""
    return ResolveVoice(Model), ResolveInstructions(Model)


class OpenAIFMResponse(Struct, frozen=True):
    voice: str
    model: str
    audio: bytes = b""
    mime_type: str = "audio/mpeg"
    expires_at: float = 0.0


class OpenAIFMStream:
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


class OpenAIFM:
    """OpenAI.fm text-to-speech provider.

    Models are specified as ``fm/<voice-or-style>``:

    - voices: ``alloy``, ``ash``, ``ballad``, ``cedar``, ``coral`` (default),
      ``fable``, ``marin``, ``nova``, ``onyx``, ``sage``, ``verse``
    - styles: ``friendly``, ``patient_teacher``, ``noir_detective``,
      ``cowboy``, ``calm``, ``scientific_style`` (a style sets the
      instructions and falls back to the ``coral`` voice)

    Usage::

        ```py
        from fishr import OpenAIFM

        fm = OpenAIFM()

        # default voice (coral)
        result = fm.speak("Hello world")
        print(result.voice, len(result.audio))

        # explicit voice
        result = fm.speak("Hello world", voice="nova")

        # style (instructions come from the style)
        result = fm.speak("Hello world", model="fm/cowboy")

        # custom instructions
        result = fm.speak(
            "Hello world",
            voice="sage",
            instructions="Speak softly and slowly.",
        )

        # stream raw audio bytes
        async for chunk in await fm.speak_async("Hello world", stream=True):
            ...```
    """

    __slots__ = ("HttpClient",)

    def __init__(self) -> None:
        self.HttpClient = make_client(headers=Headers)

    def speak(
        self,
        Prompt: str,
        *,
        model: str = "fm/coral",
        voice: str | None = None,
        instructions: str | None = None,
        stream: bool = False,
    ) -> OpenAIFMResponse | OpenAIFMStream:
        ResolvedVoice, StyleInstructions = ResolveModel(model)
        Voice = voice or ResolvedVoice
        if instructions is None:
            instructions = StyleInstructions

        Params = {"input": Prompt, "voice": Voice}
        if instructions:
            Params["prompt"] = instructions

        Resp = self.HttpClient.get(ApiUrl, params=Params)
        if Resp.status_code >= 400:
            Log.warning(
                "OpenAI.fm request failed: %s %s",
                Resp.status_code,
                Resp.text[:200],
            )
            Mime = _ContentType(Resp)
            if stream:
                return OpenAIFMStream(b"", Voice, _StreamModelName(model), Mime)
            return OpenAIFMResponse(
                voice=Voice,
                model=_StreamModelName(model),
                audio=b"",
                mime_type=Mime,
            )
        Mime = _ContentType(Resp)
        Audio = Resp.content if hasattr(Resp, "content") else b""
        if stream:
            return OpenAIFMStream(Audio, Voice, _StreamModelName(model), Mime)
        return OpenAIFMResponse(
            voice=Voice,
            model=_StreamModelName(model),
            audio=Audio,
            mime_type=Mime,
        )

    async def speak_async(
        self,
        Prompt: str,
        *,
        model: str = "fm/coral",
        voice: str | None = None,
        instructions: str | None = None,
        stream: bool = False,
    ) -> OpenAIFMResponse | OpenAIFMStream:
        return await asyncio.to_thread(
            self.speak,
            Prompt,
            model=model,
            voice=voice,
            instructions=instructions,
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


def _ContentType(Resp) -> str:
    Headers = getattr(Resp, "headers", None)
    if not Headers:
        return "audio/mpeg"
    for Key in ("content-type", "Content-Type"):
        Value = Headers.get(Key)
        if Value:
            return Value.split(";", 1)[0].strip()
    return "audio/mpeg"


def _StreamModelName(Model: str) -> str:
    Base = Model.split("/", 1)[-1] if "/" in Model else Model
    return Base if Base in Voices or Base in Styles else "coral"


__all__ = [
    "OpenAIFM",
    "OpenAIFMResponse",
    "OpenAIFMStream",
    "Voices",
    "Styles",
    "DefaultVoice",
    "ApiUrl",
    "ResolveModel",
    "ResolveVoice",
    "ResolveInstructions",
]
