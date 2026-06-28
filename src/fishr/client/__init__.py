from fishr.audio.OpenAIFM import OpenAIFM
from fishr.audio.TelnyxAudio import TelnyxAudio
from fishr.Base.Conversation import AsyncConversation, Conversation
from fishr.Base.DeepAI import DeepAI
from fishr.Base.DphnAI import DphnAI
from fishr.Base.Eris import Eris
from fishr.Base.NoTrack import NoTrack
from fishr.Base.Noxus import Noxus
from fishr.Base.OperaAria import OperaAria
from fishr.Base.Quillbot import Quillbot
from fishr.Base.Raphael import Raphael
from fishr.Base.Telnyx import Telnyx
from fishr.Base.Yqcloud import Yqcloud
from fishr.client.agents import AgentRun, AgentStep, AsyncAgentRun
from fishr.client.audio import AsyncAudio, Audio
from fishr.client.completions import (
    AsyncChat,
    AsyncCompletions,
    Chat,
    Completions,
)
from fishr.client.images import AsyncImages, Images
from fishr.client.routing import provider_of, resolve_model
from fishr.client.streams import (
    AsyncStream,
    DeepAIAsyncStream,
    DeepAISyncStream,
    ErisAsyncStream,
    ErisSyncStream,
    NoTrackAsyncStream,
    NoTrackSyncStream,
    OperaAriaAsyncStream,
    OperaAriaSyncStream,
    QuillbotAsyncStream,
    QuillbotSyncStream,
    SyncStream,
    TelnyxAsyncStream,
    TelnyxSyncStream,
    YqcloudAsyncStream,
    YqcloudSyncStream,
)


class Client:
    """Synchronous fishr client.

    Usage::

        from fishr import Client

        client = Client()

        # chat
        response = client.chat.completions.create(
            model="noxus/openai",
            messages=[{"role": "user", "content": "Hello!"}],
        )
        print(response.text)

        # streaming
        for chunk in client.chat.completions.create(
            model="noxus/openai",
            messages=[{"role": "user", "content": "Tell me a story"}],
            stream=True,
        ):
            print(chunk.choices[0].delta.content or "", end="")

        # image generation
        result = client.images.generate(
            model="deepai/image",
            prompt="A cat riding a skateboard",
        )
        print(result.data[0].url)

        # agent
        result = client.agents.run(
            model="noxus/openai",
            prompt="Research quantum computing",
            tools=[{"type": "web_search"}],
        )
        print(result.content)

        # multi-turn
        conv = client.conversation(model="noxus/openai")
        conv.ask("Remember: 42")
        conv.ask("What number did I say?")
    """

    __slots__ = (
        "chat",
        "images",
        "audio",
        "agents",
        "noxus",
        "deepai",
        "quillbot",
        "notrack",
        "dphnai",
        "yqcloud",
        "opera",
        "eris",
        "telnyx",
        "raphael",
        "openai_fm",
        "telnyx_audio",
    )

    def __init__(self) -> None:
        self.noxus = Noxus()
        self.deepai = DeepAI()
        self.quillbot = Quillbot()
        self.notrack = NoTrack()
        self.dphnai = DphnAI()
        self.yqcloud = Yqcloud()
        self.opera = OperaAria()
        self.eris = Eris()
        self.telnyx = Telnyx()
        self.raphael = Raphael()
        self.openai_fm = OpenAIFM()
        self.telnyx_audio = TelnyxAudio()
        self.chat = Chat(
            self.noxus,
            self.deepai,
            self.quillbot,
            self.notrack,
            self.dphnai,
            self.yqcloud,
            self.opera,
            self.eris,
            self.telnyx,
        )
        self.images = Images(self.deepai, self.raphael)
        self.audio = Audio(self.openai_fm, self.telnyx_audio)
        self.agents = AgentRun(self.noxus, self.deepai)

    def conversation(self, model: str = "noxus/openai") -> Conversation:
        provider = provider_of(model)
        if provider == "deepai":
            return Conversation(self.deepai, model=model)
        if provider == "quillbot":
            return Conversation(self.quillbot, model=model)
        if provider == "notrack":
            return Conversation(self.notrack, model=model)
        if provider == "dphnai":
            return Conversation(self.dphnai, model=model)
        if provider == "yqcloud":
            return Conversation(self.yqcloud, model=model)
        if provider == "opera":
            return Conversation(self.opera, model=model)
        if provider == "eris":
            return Conversation(self.eris, model=model)
        if provider == "telnyx":
            return Conversation(self.telnyx, model=model)
        return Conversation(self.noxus, model=model)

    def __repr__(self) -> str:
        return "Client()"


class AsyncClient:
    """Asynchronous fishr client.

    Usage::

        from fishr import AsyncClient

        client = AsyncClient()

        # chat
        response = await client.chat.completions.create(
            model="noxus/openai",
            messages=[{"role": "user", "content": "Hello!"}],
        )
        print(response.text)

        # image generation
        result = await client.images.generate(
            model="deepai/image",
            prompt="A cat riding a skateboard",
        )
        print(result.data[0].url)

        # multi-turn
        conv = client.conversation(model="noxus/openai")
        await conv.ask("Remember: 42")
        await conv.ask("What number did I say?")
    """

    __slots__ = (
        "chat",
        "images",
        "audio",
        "agents",
        "noxus",
        "deepai",
        "quillbot",
        "notrack",
        "dphnai",
        "yqcloud",
        "opera",
        "eris",
        "telnyx",
        "raphael",
        "openai_fm",
        "telnyx_audio",
    )

    def __init__(self) -> None:
        self.noxus = Noxus()
        self.deepai = DeepAI()
        self.quillbot = Quillbot()
        self.notrack = NoTrack()
        self.dphnai = DphnAI()
        self.yqcloud = Yqcloud()
        self.opera = OperaAria()
        self.eris = Eris()
        self.telnyx = Telnyx()
        self.raphael = Raphael()
        self.openai_fm = OpenAIFM()
        self.telnyx_audio = TelnyxAudio()
        self.chat = AsyncChat(
            self.noxus,
            self.deepai,
            self.quillbot,
            self.notrack,
            self.dphnai,
            self.yqcloud,
            self.opera,
            self.eris,
            self.telnyx,
        )
        self.images = AsyncImages(self.deepai, self.raphael)
        self.audio = AsyncAudio(self.openai_fm, self.telnyx_audio)
        self.agents = AsyncAgentRun(self.noxus, self.deepai)

    def conversation(self, model: str = "noxus/openai") -> AsyncConversation:
        provider = provider_of(model)
        if provider == "deepai":
            return AsyncConversation(self.deepai, model=model)
        if provider == "quillbot":
            return AsyncConversation(self.quillbot, model=model)
        if provider == "notrack":
            return AsyncConversation(self.notrack, model=model)
        if provider == "dphnai":
            return AsyncConversation(self.dphnai, model=model)
        if provider == "yqcloud":
            return AsyncConversation(self.yqcloud, model=model)
        if provider == "opera":
            return AsyncConversation(self.opera, model=model)
        if provider == "eris":
            return AsyncConversation(self.eris, model=model)
        if provider == "telnyx":
            return AsyncConversation(self.telnyx, model=model)
        return AsyncConversation(self.noxus, model=model)

    def __repr__(self) -> str:
        return "AsyncClient()"


__all__ = [
    "Client",
    "AsyncClient",
    "Chat",
    "AsyncChat",
    "Completions",
    "AsyncCompletions",
    "Images",
    "AsyncImages",
    "Audio",
    "AsyncAudio",
    "AgentRun",
    "AsyncAgentRun",
    "AgentStep",
    "SyncStream",
    "AsyncStream",
    "DeepAISyncStream",
    "DeepAIAsyncStream",
    "QuillbotSyncStream",
    "QuillbotAsyncStream",
    "NoTrackSyncStream",
    "NoTrackAsyncStream",
    "YqcloudSyncStream",
    "YqcloudAsyncStream",
    "OperaAriaSyncStream",
    "OperaAriaAsyncStream",
    "ErisSyncStream",
    "ErisAsyncStream",
    "TelnyxSyncStream",
    "TelnyxAsyncStream",
    "resolve_model",
    "provider_of",
]
