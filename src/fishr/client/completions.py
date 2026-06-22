from typing import AsyncIterator, Iterator

from fishr.Base.DeepAI import DeepAI, DeepAIStream
from fishr.Base.DphnAI import DphnAI, DphnAIStream
from fishr.Base.NoTrack import NoTrack, NoTrackStream
from fishr.Base.Noxus import Image, Noxus, NoxusMessage
from fishr.Base.OperaAria import OperaAria, OperaAriaStream
from fishr.Base.Quillbot import Quillbot, QuillbotStream
from fishr.Base.Yqcloud import Yqcloud, YqcloudStream
from fishr.client.routing import provider_of, resolve_model
from fishr.client.streams import (
    AsyncStream,
    DeepAIAsyncStream,
    DeepAISyncStream,
    DphnAIAsyncStream,
    DphnAISyncStream,
    NoTrackAsyncStream,
    NoTrackSyncStream,
    OperaAriaAsyncStream,
    OperaAriaSyncStream,
    QuillbotAsyncStream,
    QuillbotSyncStream,
    SyncStream,
    YqcloudAsyncStream,
    YqcloudSyncStream,
)
from fishr.Types import (
    ChatCompletion,
    ChatCompletionChunk,
    Choice,
    Delta,
    Message,
)


class Completions:
    """Create chat completions via ``client.chat.completions.create(...)``."""

    __slots__ = ("noxus", "deepai", "quillbot", "notrack", "dphnai", "yqcloud", "opera")

    def __init__(
        self,
        noxus: Noxus,
        deepai: DeepAI,
        quillbot: Quillbot,
        notrack: NoTrack,
        dphnai: DphnAI,
        yqcloud: Yqcloud,
        opera: OperaAria,
    ) -> None:
        self.noxus = noxus
        self.deepai = deepai
        self.quillbot = quillbot
        self.notrack = notrack
        self.dphnai = dphnai
        self.yqcloud = yqcloud
        self.opera = opera

    @staticmethod
    def _build_messages(messages: list[dict]) -> tuple[NoxusMessage, ...]:
        out: list[NoxusMessage] = []
        for m in messages:
            img = None
            if "image" in m and isinstance(m["image"], dict):
                img = Image(
                    mime_type=m["image"]["mime_type"],
                    base64_data=m["image"]["base64_data"],
                )
            out.append(NoxusMessage(role=m["role"], content=m["content"], image=img))
        return tuple(out)

    def _noxus_create(
        self,
        resolved: str,
        messages: list[dict],
        web_search: bool,
        stream: bool,
    ) -> ChatCompletion | Iterator[ChatCompletionChunk]:
        noxus_msgs = self._build_messages(messages)
        last_user = [m for m in messages if m.get("role") == "user"][-1]

        if stream:
            raw = self.noxus.chat(
                noxus_msgs,
                model=resolved,
                web_search=web_search,
                stream=True,
            )
            return SyncStream(raw, resolved)

        result = self.noxus.ask(
            last_user["content"],
            model=resolved,
            web_search=web_search,
        )
        msg = Message(role="assistant", content=result)
        choice = Choice(index=0, message=msg)
        return ChatCompletion(id="", model=resolved, choices=(choice,))

    def _deepai_create(
        self,
        resolved: str,
        messages: list[dict],
        web_search: bool,
        stream: bool,
    ) -> ChatCompletion | Iterator[ChatCompletionChunk]:
        if stream:
            raw = self.deepai.chat(
                messages,
                model=resolved,
                web_search=web_search,
                stream=True,
            )
            if isinstance(raw, DeepAIStream):
                return DeepAISyncStream(raw, resolved)
            return SyncStream(iter([raw.content]), resolved)

        has_images = any(m.get("image") for m in messages)
        if has_images:
            result = self.deepai.chat(
                messages,
                model=resolved,
                web_search=web_search,
            )
        else:
            last_user = [m for m in messages if m.get("role") == "user"][-1]
            result = self.deepai.ask(
                last_user["content"],
                model=resolved,
                web_search=web_search,
            )
        if isinstance(result, DeepAIStream):
            content = "".join(result)
        else:
            content = result.image_url if result.image_url else result.content
        msg = Message(role="assistant", content=content)
        choice = Choice(index=0, message=msg)
        return ChatCompletion(id="", model=resolved, choices=(choice,))

    def _quillbot_create(
        self,
        resolved: str,
        messages: list[dict],
        web_search: bool,
        stream: bool,
    ) -> ChatCompletion | Iterator[ChatCompletionChunk]:
        if stream:
            raw = self.quillbot.chat(
                messages,
                model=resolved,
                web_search=web_search,
                stream=True,
            )
            if isinstance(raw, QuillbotStream):
                return QuillbotSyncStream(raw, resolved)
            return SyncStream(iter([raw.content]), resolved)

        result = self.quillbot.chat(
            messages,
            model=resolved,
            web_search=web_search,
        )
        if isinstance(result, QuillbotStream):
            content = "".join(result)
        else:
            content = result.content
        msg = Message(role="assistant", content=content)
        choice = Choice(index=0, message=msg)
        return ChatCompletion(id="", model=resolved, choices=(choice,))

    def _notrack_create(
        self,
        resolved: str,
        messages: list[dict],
        web_search: bool,
        stream: bool,
    ) -> ChatCompletion | Iterator[ChatCompletionChunk]:
        if stream:
            raw = self.notrack.chat(
                messages,
                model=resolved,
                stream=True,
            )
            if isinstance(raw, NoTrackStream):
                return NoTrackSyncStream(raw, resolved)
            return SyncStream(iter([raw.content]), resolved)

        result = self.notrack.chat(
            messages,
            model=resolved,
        )
        if isinstance(result, NoTrackStream):
            content = "".join(result)
        else:
            content = result.content
        msg = Message(role="assistant", content=content)
        choice = Choice(index=0, message=msg)
        return ChatCompletion(id="", model=resolved, choices=(choice,))

    def _dphnai_create(
        self,
        resolved: str,
        messages: list[dict],
        web_search: bool,
        stream: bool,
    ) -> ChatCompletion | Iterator[ChatCompletionChunk]:
        if stream:
            raw = self.dphnai.chat(
                messages,
                model=resolved,
                stream=True,
            )
            if isinstance(raw, DphnAIStream):
                return DphnAISyncStream(raw, resolved)
            return SyncStream(iter([raw.content]), resolved)

        result = self.dphnai.chat(
            messages,
            model=resolved,
        )
        if isinstance(result, DphnAIStream):
            content = "".join(result)
        else:
            content = result.content
        msg = Message(role="assistant", content=content)
        choice = Choice(index=0, message=msg)
        return ChatCompletion(id="", model=resolved, choices=(choice,))

    def _yqcloud_create(
        self,
        resolved: str,
        messages: list[dict],
        web_search: bool,
        stream: bool,
    ) -> ChatCompletion | Iterator[ChatCompletionChunk]:
        if stream:
            raw = self.yqcloud.chat(
                messages,
                model=resolved,
                stream=True,
            )
            if isinstance(raw, YqcloudStream):
                return YqcloudSyncStream(raw, resolved)
            return SyncStream(iter([raw.content]), resolved)

        result = self.yqcloud.chat(
            messages,
            model=resolved,
        )
        if isinstance(result, YqcloudStream):
            content = "".join(result)
        else:
            content = result.content
        msg = Message(role="assistant", content=content)
        choice = Choice(index=0, message=msg)
        return ChatCompletion(id="", model=resolved, choices=(choice,))

    def _opera_create(
        self,
        resolved: str,
        messages: list[dict],
        web_search: bool,
        stream: bool,
        think_harder: bool = False,
    ) -> ChatCompletion | Iterator[ChatCompletionChunk]:
        if stream:
            raw = self.opera.chat(
                messages,
                model=resolved,
                stream=True,
                think_harder=think_harder,
            )
            if isinstance(raw, OperaAriaStream):
                return OperaAriaSyncStream(raw, resolved)
            return SyncStream(iter([raw.content]), resolved)

        result = self.opera.chat(
            messages,
            model=resolved,
            think_harder=think_harder,
        )
        if isinstance(result, OperaAriaStream):
            content = "".join(result)
        else:
            content = result.content
            if result.image_urls:
                content += "\n" + "\n".join(result.image_urls)
        msg = Message(role="assistant", content=content)
        choice = Choice(index=0, message=msg)
        return ChatCompletion(id="", model=resolved, choices=(choice,))

    def create(
        self,
        *,
        model: str,
        messages: list[dict],
        web_search: bool = False,
        stream: bool = False,
        think_harder: bool = False,
    ) -> ChatCompletion | Iterator[ChatCompletionChunk]:
        resolved = resolve_model(model)
        provider = provider_of(resolved)

        if provider == "deepai":
            return self._deepai_create(resolved, messages, web_search, stream)
        if provider == "quillbot":
            return self._quillbot_create(resolved, messages, web_search, stream)
        if provider == "notrack":
            return self._notrack_create(resolved, messages, web_search, stream)
        if provider == "dphnai":
            return self._dphnai_create(resolved, messages, web_search, stream)
        if provider == "yqcloud":
            return self._yqcloud_create(resolved, messages, web_search, stream)
        if provider == "opera":
            return self._opera_create(
                resolved, messages, web_search, stream, think_harder=think_harder
            )

        return self._noxus_create(resolved, messages, web_search, stream)


class AsyncCompletions:
    """Async version of :class:`Completions`."""

    __slots__ = ("noxus", "deepai", "quillbot", "notrack", "dphnai", "yqcloud", "opera")

    def __init__(
        self,
        noxus: Noxus,
        deepai: DeepAI,
        quillbot: Quillbot,
        notrack: NoTrack,
        dphnai: DphnAI,
        yqcloud: Yqcloud,
        opera: OperaAria,
    ) -> None:
        self.noxus = noxus
        self.deepai = deepai
        self.quillbot = quillbot
        self.notrack = notrack
        self.dphnai = dphnai
        self.yqcloud = yqcloud
        self.opera = opera

    async def _noxus_create(
        self,
        resolved: str,
        messages: list[dict],
        web_search: bool,
        stream: bool,
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        from fishr.Loop import asyncio

        noxus_msgs = Completions._build_messages(messages)
        last_user = [m for m in messages if m.get("role") == "user"][-1]

        if stream:
            raw = await asyncio.to_thread(
                self.noxus.chat,
                noxus_msgs,
                model=resolved,
                web_search=web_search,
                stream=True,
            )
            return AsyncStream(raw, resolved)

        result = await self.noxus.ask_async(
            last_user["content"],
            model=resolved,
            web_search=web_search,
        )
        msg = Message(role="assistant", content=result)
        choice = Choice(index=0, message=msg)
        return ChatCompletion(id="", model=resolved, choices=(choice,))

    async def _deepai_create(
        self,
        resolved: str,
        messages: list[dict],
        web_search: bool,
        stream: bool,
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        from fishr.Loop import asyncio

        if stream:
            raw = await asyncio.to_thread(
                self.deepai.chat,
                messages,
                model=resolved,
                web_search=web_search,
                stream=True,
            )
            if isinstance(raw, DeepAIStream):
                return DeepAIAsyncStream(raw, resolved)
            return AsyncStream(iter([raw.content]), resolved)

        has_images = any(m.get("image") for m in messages)
        if has_images:
            result = await asyncio.to_thread(
                self.deepai.chat,
                messages,
                model=resolved,
                web_search=web_search,
            )
        else:
            last_user = [m for m in messages if m.get("role") == "user"][-1]
            result = await asyncio.to_thread(
                self.deepai.ask,
                last_user["content"],
                model=resolved,
                web_search=web_search,
            )
        if isinstance(result, DeepAIStream):
            content = "".join(result)
        else:
            content = result.image_url if result.image_url else result.content
        msg = Message(role="assistant", content=content)
        choice = Choice(index=0, message=msg)
        return ChatCompletion(id="", model=resolved, choices=(choice,))

    async def _quillbot_create(
        self,
        resolved: str,
        messages: list[dict],
        web_search: bool,
        stream: bool,
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        from fishr.Loop import asyncio

        if stream:
            raw = await asyncio.to_thread(
                self.quillbot.chat,
                messages,
                model=resolved,
                web_search=web_search,
                stream=True,
            )
            if isinstance(raw, QuillbotStream):
                return QuillbotAsyncStream(raw, resolved)
            return AsyncStream(iter([raw.content]), resolved)

        result = await asyncio.to_thread(
            self.quillbot.chat,
            messages,
            model=resolved,
            web_search=web_search,
        )
        if isinstance(result, QuillbotStream):
            content = "".join(result)
        else:
            content = result.content
        msg = Message(role="assistant", content=content)
        choice = Choice(index=0, message=msg)
        return ChatCompletion(id="", model=resolved, choices=(choice,))

    async def _notrack_create(
        self,
        resolved: str,
        messages: list[dict],
        web_search: bool,
        stream: bool,
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        from fishr.Loop import asyncio

        if stream:
            raw = await asyncio.to_thread(
                self.notrack.chat,
                messages,
                model=resolved,
                stream=True,
            )
            if isinstance(raw, NoTrackStream):
                return NoTrackAsyncStream(raw, resolved)
            return AsyncStream(iter([raw.content]), resolved)

        result = await asyncio.to_thread(
            self.notrack.chat,
            messages,
            model=resolved,
        )
        if isinstance(result, NoTrackStream):
            content = "".join(result)
        else:
            content = result.content
        msg = Message(role="assistant", content=content)
        choice = Choice(index=0, message=msg)
        return ChatCompletion(id="", model=resolved, choices=(choice,))

    async def _dphnai_create(
        self,
        resolved: str,
        messages: list[dict],
        web_search: bool,
        stream: bool,
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        from fishr.Loop import asyncio

        if stream:
            raw = await asyncio.to_thread(
                self.dphnai.chat,
                messages,
                model=resolved,
                stream=True,
            )
            if isinstance(raw, DphnAIStream):
                return DphnAIAsyncStream(raw, resolved)
            return AsyncStream(iter([raw.content]), resolved)

        result = await asyncio.to_thread(
            self.dphnai.chat,
            messages,
            model=resolved,
        )
        if isinstance(result, DphnAIStream):
            content = "".join(result)
        else:
            content = result.content
        msg = Message(role="assistant", content=content)
        choice = Choice(index=0, message=msg)
        return ChatCompletion(id="", model=resolved, choices=(choice,))

    async def _yqcloud_create(
        self,
        resolved: str,
        messages: list[dict],
        web_search: bool,
        stream: bool,
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        from fishr.Loop import asyncio

        if stream:
            raw = await asyncio.to_thread(
                self.yqcloud.chat,
                messages,
                model=resolved,
                stream=True,
            )
            if isinstance(raw, YqcloudStream):
                return YqcloudAsyncStream(raw, resolved)
            return AsyncStream(iter([raw.content]), resolved)

        result = await asyncio.to_thread(
            self.yqcloud.chat,
            messages,
            model=resolved,
        )
        if isinstance(result, YqcloudStream):
            content = "".join(result)
        else:
            content = result.content
        msg = Message(role="assistant", content=content)
        choice = Choice(index=0, message=msg)
        return ChatCompletion(id="", model=resolved, choices=(choice,))

    async def _opera_create(
        self,
        resolved: str,
        messages: list[dict],
        web_search: bool,
        stream: bool,
        think_harder: bool = False,
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        from fishr.Loop import asyncio

        if stream:
            raw = await asyncio.to_thread(
                self.opera.chat,
                messages,
                model=resolved,
                stream=True,
                think_harder=think_harder,
            )
            if isinstance(raw, OperaAriaStream):
                return OperaAriaAsyncStream(raw, resolved)
            return AsyncStream(iter([raw.content]), resolved)

        result = await asyncio.to_thread(
            self.opera.chat,
            messages,
            model=resolved,
            think_harder=think_harder,
        )
        if isinstance(result, OperaAriaStream):
            content = "".join(result)
        else:
            content = result.content
            if result.image_urls:
                content += "\n" + "\n".join(result.image_urls)
        msg = Message(role="assistant", content=content)
        choice = Choice(index=0, message=msg)
        return ChatCompletion(id="", model=resolved, choices=(choice,))

    async def create(
        self,
        *,
        model: str,
        messages: list[dict],
        web_search: bool = False,
        stream: bool = False,
        think_harder: bool = False,
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        resolved = resolve_model(model)
        provider = provider_of(resolved)

        if provider == "deepai":
            return await self._deepai_create(resolved, messages, web_search, stream)
        if provider == "quillbot":
            return await self._quillbot_create(resolved, messages, web_search, stream)
        if provider == "notrack":
            return await self._notrack_create(resolved, messages, web_search, stream)
        if provider == "dphnai":
            return await self._dphnai_create(resolved, messages, web_search, stream)
        if provider == "yqcloud":
            return await self._yqcloud_create(resolved, messages, web_search, stream)
        if provider == "opera":
            return await self._opera_create(
                resolved, messages, web_search, stream, think_harder=think_harder
            )

        return await self._noxus_create(resolved, messages, web_search, stream)


class Chat:
    __slots__ = ("completions",)

    def __init__(
        self,
        noxus: Noxus,
        deepai: DeepAI,
        quillbot: Quillbot,
        notrack: NoTrack,
        dphnai: DphnAI,
        yqcloud: Yqcloud,
        opera: OperaAria,
    ) -> None:
        self.completions = Completions(
            noxus, deepai, quillbot, notrack, dphnai, yqcloud, opera
        )


class AsyncChat:
    __slots__ = ("completions",)

    def __init__(
        self,
        noxus: Noxus,
        deepai: DeepAI,
        quillbot: Quillbot,
        notrack: NoTrack,
        dphnai: DphnAI,
        yqcloud: Yqcloud,
        opera: OperaAria,
    ) -> None:
        self.completions = AsyncCompletions(
            noxus, deepai, quillbot, notrack, dphnai, yqcloud, opera
        )


__all__ = [
    "Completions",
    "AsyncCompletions",
    "Chat",
    "AsyncChat",
]
