from msgspec import Struct


class ModelDef(Struct, frozen=True):
    web_search: bool
    image: bool
    file_attach: bool = False
    history: bool = True
    system: bool = True


class Message(Struct, frozen=True):
    role: str
    content: str


class Choice(Struct, frozen=True):
    index: int
    message: Message
    finish_reason: str = ""


class Usage(Struct, frozen=True):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletion(Struct, frozen=True):
    id: str = ""
    model: str = ""
    choices: tuple[Choice, ...] = ()
    usage: Usage = Usage()

    @property
    def text(self) -> str:
        """Shortcut for ``choices[0].message.content``."""
        if self.choices:
            return self.choices[0].message.content
        return ""


class Delta(Struct, frozen=True):
    role: str = ""
    content: str = ""
    thinking: str = ""


class ChunkChoice(Struct, frozen=True):
    index: int
    delta: Delta
    finish_reason: str = ""


class ChatCompletionChunk(Struct, frozen=True):
    id: str = ""
    model: str = ""
    choices: tuple[ChunkChoice, ...] = ()


class ImageUrl(Struct, frozen=True):
    url: str
    alt: str = ""


class ImageResponse(Struct, frozen=True):
    created: int = 0
    data: tuple[ImageUrl, ...] = ()


class AudioData(Struct, frozen=True):
    voice: str = ""
    model: str = ""
    audio: bytes = b""
    mime_type: str = "audio/mpeg"


class AudioResponse(Struct, frozen=True):
    created: int = 0
    data: tuple[AudioData, ...] = ()


noxus_bot_ids = {
    "noxus/openai": 25871,
    "noxus/google": 25874,
    "noxus/sonnet-4.6": 25873,
    "noxus/sonnet-3.5": 25875,
    "noxus/grok-4.3": 25872,
    "noxus/perplexity": 25876,
    "noxus/metaai": 25870,
    "noxus/qwen": 25869,
}

models = {
    # noxus
    "noxus/openai": ModelDef(
        web_search=True, image=False, file_attach=True, history=True, system=True
    ),
    "noxus/google": ModelDef(
        web_search=True, image=False, file_attach=True, history=True, system=True
    ),
    "noxus/sonnet-4.6": ModelDef(
        web_search=True, image=False, file_attach=True, history=True, system=True
    ),
    "noxus/sonnet-3.5": ModelDef(
        web_search=True, image=False, file_attach=True, history=True, system=True
    ),
    "noxus/grok-4.3": ModelDef(
        web_search=True, image=False, file_attach=True, history=True, system=True
    ),
    "noxus/perplexity": ModelDef(
        web_search=False, image=False, file_attach=True, history=True, system=True
    ),
    "noxus/metaai": ModelDef(web_search=False, image=False, history=True, system=True),
    "noxus/qwen": ModelDef(web_search=True, image=False, history=True, system=True),
    # deepai
    "deepai/standard": ModelDef(
        web_search=False, image=True, file_attach=True, history=True, system=True
    ),
    "deepai/online": ModelDef(
        web_search=True, image=True, file_attach=True, history=True, system=True
    ),
    "deepai/gemma-4": ModelDef(
        web_search=False, image=True, file_attach=True, history=True, system=True
    ),
    "deepai/gemini-2.5-flash-lite": ModelDef(
        web_search=False,
        image=True,
        file_attach=True,
        history=True,
        system=True,
    ),
    "deepai/deepseek-v3.2": ModelDef(
        web_search=False, image=True, file_attach=True, history=True, system=True
    ),
    "deepai/image": ModelDef(
        web_search=False, image=True, file_attach=False, history=True, system=False
    ),
    # quillbot
    "quillbot/quillbot": ModelDef(
        web_search=False, image=False, history=True, system=True
    ),
    "quillbot/quillbot-search": ModelDef(
        web_search=True, image=False, history=True, system=True
    ),
    # notrack
    "notrack/fast": ModelDef(web_search=False, image=False, history=True, system=False),
    "notrack/standard": ModelDef(
        web_search=False, image=False, history=True, system=False
    ),
    "notrack/reasoning": ModelDef(
        web_search=False, image=False, history=True, system=False
    ),
    # dphnai
    "dphnai/24b": ModelDef(web_search=False, image=False, history=True, system=True),
    "dphnai/6b": ModelDef(web_search=False, image=False, history=True, system=True),
    # yqcloud
    "yqcloud/gpt-4": ModelDef(web_search=False, image=False, history=True, system=True),
    # opera
    "opera/aria": ModelDef(
        web_search=False, image=True, file_attach=True, history=True, system=True
    ),
    # eris
    "eris/deepseek-v4-flash": ModelDef(
        web_search=False, image=False, history=True, system=True
    ),
    "eris/deepseek-v4-pro": ModelDef(
        web_search=False, image=False, history=True, system=True
    ),
    "eris/glm-5.1": ModelDef(web_search=False, image=False, history=True, system=True),
    "eris/minimax-m3": ModelDef(
        web_search=False, image=False, history=True, system=True
    ),
    "eris/kimi-k2.6": ModelDef(
        web_search=False, image=False, history=True, system=True
    ),
    # telnyx (anonymous fast-path inference)
    "telnyx/glm-5.2": ModelDef(
        web_search=False, image=False, history=True, system=True
    ),
    "telnyx/glm-5.1": ModelDef(
        web_search=False, image=False, history=True, system=True
    ),
    "telnyx/kimi-k2.6": ModelDef(
        web_search=False, image=False, history=True, system=True
    ),
    "telnyx/minimax-m3": ModelDef(
        web_search=False, image=False, history=True, system=True
    ),
    # fm (text-to-speech)
    "fm/coral": ModelDef(web_search=False, image=False, history=False, system=False),
    "fm/alloy": ModelDef(web_search=False, image=False, history=False, system=False),
    "fm/ash": ModelDef(web_search=False, image=False, history=False, system=False),
    "fm/ballad": ModelDef(web_search=False, image=False, history=False, system=False),
    "fm/cedar": ModelDef(web_search=False, image=False, history=False, system=False),
    "fm/marin": ModelDef(web_search=False, image=False, history=False, system=False),
    "fm/fable": ModelDef(web_search=False, image=False, history=False, system=False),
    "fm/onyx": ModelDef(web_search=False, image=False, history=False, system=False),
    "fm/nova": ModelDef(web_search=False, image=False, history=False, system=False),
    "fm/sage": ModelDef(web_search=False, image=False, history=False, system=False),
    "fm/verse": ModelDef(web_search=False, image=False, history=False, system=False),
    "fm/friendly": ModelDef(web_search=False, image=False, history=False, system=False),
    "fm/patient_teacher": ModelDef(
        web_search=False, image=False, history=False, system=False
    ),
    "fm/noir_detective": ModelDef(
        web_search=False, image=False, history=False, system=False
    ),
    "fm/cowboy": ModelDef(web_search=False, image=False, history=False, system=False),
    "fm/calm": ModelDef(web_search=False, image=False, history=False, system=False),
    "fm/scientific_style": ModelDef(
        web_search=False, image=False, history=False, system=False
    ),
    # telnyx-tts (no-auth fast-path TTS demo endpoint)
    # Each model id is a voice: telnyx-tts/<voice> -> Telnyx.NaturalHD.<voice>
    "telnyx-tts/astra": ModelDef(
        web_search=False, image=False, history=False, system=False
    ),
    "telnyx-tts/luna": ModelDef(
        web_search=False, image=False, history=False, system=False
    ),
    "telnyx-tts/sol": ModelDef(
        web_search=False, image=False, history=False, system=False
    ),
    "telnyx-tts/nova": ModelDef(
        web_search=False, image=False, history=False, system=False
    ),
    "telnyx-tts/orion": ModelDef(
        web_search=False, image=False, history=False, system=False
    ),
    # raphael (image generation)
    "raphael/image": ModelDef(
        web_search=False, image=True, file_attach=False, history=False, system=False
    ),
}

__all__ = [
    "ModelDef",
    "Message",
    "Choice",
    "Usage",
    "ChatCompletion",
    "Delta",
    "ChunkChoice",
    "ChatCompletionChunk",
    "ImageUrl",
    "ImageResponse",
    "AudioData",
    "AudioResponse",
    "noxus_bot_ids",
    "models",
]
