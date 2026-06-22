from msgspec import Struct


class ModelDef(Struct, frozen=True):
    web_search: bool
    image: bool
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
    "noxus/openai": ModelDef(web_search=True, image=True, history=True, system=True),
    "noxus/google": ModelDef(web_search=True, image=True, history=True, system=True),
    "noxus/sonnet-4.6": ModelDef(
        web_search=True, image=False, history=True, system=True
    ),
    "noxus/sonnet-3.5": ModelDef(
        web_search=True, image=True, history=True, system=True
    ),
    "noxus/grok-4.3": ModelDef(web_search=True, image=True, history=True, system=True),
    "noxus/perplexity": ModelDef(
        web_search=False, image=True, history=True, system=True
    ),
    "noxus/metaai": ModelDef(web_search=False, image=False, history=True, system=True),
    "noxus/qwen": ModelDef(web_search=True, image=False, history=True, system=True),
    # deepai
    "deepai/standard": ModelDef(
        web_search=False, image=True, history=True, system=True
    ),
    "deepai/online": ModelDef(web_search=True, image=True, history=True, system=True),
    "deepai/gemma-4": ModelDef(web_search=False, image=True, history=True, system=True),
    "deepai/gemini-2.5-flash-lite": ModelDef(
        web_search=False,
        image=True,
        history=True,
        system=True,
    ),
    "deepai/deepseek-v3.2": ModelDef(
        web_search=False, image=True, history=True, system=True
    ),
    "deepai/image": ModelDef(web_search=False, image=True, history=True, system=False),
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
    "opera/aria": ModelDef(web_search=False, image=True, history=True, system=True),
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
    "noxus_bot_ids",
    "models",
]
