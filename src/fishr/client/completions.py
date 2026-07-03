from typing import AsyncIterator, Iterator

from fishr.Base.DeepAI import DeepAI, DeepAIStream
from fishr.Base.DphnAI import DphnAI, DphnAIStream
from fishr.Base.Eris import Eris, ErisStream
from fishr.Base.Kai import Kai, KaiStream
from fishr.Base.NoTrack import NoTrack, NoTrackStream
from fishr.Base.Noxus import Image, Noxus, NoxusMessage, NoxusResponse
from fishr.Base.OperaAria import OperaAria, OperaAriaStream
from fishr.Base.Quillbot import Quillbot, QuillbotStream
from fishr.Base.Telnyx import Telnyx, TelnyxStream
from fishr.Base.Yqcloud import Yqcloud, YqcloudStream
from fishr.client.routing import provider_of, resolve_model
from fishr.client.streams import (
    AsyncStream,
    DeepAIAsyncStream,
    DeepAISyncStream,
    ErisAsyncStream,
    ErisSyncStream,
    KaiAsyncStream,
    KaiSyncStream,
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
from fishr.Types import (
    ChatCompletion,
    ChatCompletionChunk,
    Choice,
    Delta,
    FunctionCall,
    Message,
    ToolCall,
    ToolCallDelta,
)

_ALLOWED_IMAGE_MIMES = frozenset({"image/webp", "image/png", "image/jpeg", "image/jpg"})


def _normalize_messages(messages: list[dict]) -> list[dict]:
    """Normalize OpenAI-style content arrays into the internal ``image`` dict format.

    Accepts both:

    - ``{"content": "text", "image": {"mime_type": ..., "base64_data": ...}}``
    - ``{"content": [{"type": "text", ...}, {"type": "image_url", ...}]}``

    and always produces the first form so providers don't need to handle arrays.

    Only ``image/webp``, ``image/png``, ``image/jpeg``, and ``image/jpg`` are
    accepted — ``image/gif`` and others are silently skipped.
    """
    out: list[dict] = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            text_parts: list[str] = []
            image = None
            for part in content:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type")
                if ptype == "text":
                    text_parts.append(part.get("text", ""))
                elif ptype == "image_url":
                    url = part.get("image_url", {}).get("url", "")
                    if url.startswith("data:"):
                        header, _, data = url.partition(",")
                        mime = header[5:]  # strip "data:"
                        if ";" in mime:
                            mime = mime.split(";", 1)[0]
                        if mime in _ALLOWED_IMAGE_MIMES:
                            image = {"mime_type": mime, "base64_data": data}
            new_m: dict = {"role": m["role"], "content": "\n".join(text_parts)}
            if image:
                new_m["image"] = image
            if "tool_calls" in m and m["tool_calls"]:
                new_m["tool_calls"] = m["tool_calls"]
            if m.get("tool_call_id"):
                new_m["tool_call_id"] = m["tool_call_id"]
            out.append(new_m)
        else:
            # validate the legacy image dict format — strip unsupported mimes
            img = m.get("image")
            if (
                isinstance(img, dict)
                and img.get("mime_type") not in _ALLOWED_IMAGE_MIMES
            ):
                m = {k: v for k, v in m.items() if k != "image"}
            out.append(m)
    return out


class Completions:
    """Create chat completions via ``client.chat.completions.create(...)``."""

    __slots__ = (
        "noxus",
        "deepai",
        "quillbot",
        "notrack",
        "dphnai",
        "yqcloud",
        "opera",
        "eris",
        "telnyx",
        "kai",
    )

    def __init__(
        self,
        noxus: Noxus,
        deepai: DeepAI,
        quillbot: Quillbot,
        notrack: NoTrack,
        dphnai: DphnAI,
        yqcloud: Yqcloud,
        opera: OperaAria,
        eris: Eris,
        telnyx: Telnyx,
        kai: Kai,
    ) -> None:
        self.noxus = noxus
        self.deepai = deepai
        self.quillbot = quillbot
        self.notrack = notrack
        self.dphnai = dphnai
        self.yqcloud = yqcloud
        self.opera = opera
        self.eris = eris
        self.telnyx = telnyx
        self.kai = kai

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

        if stream:
            raw = self.noxus.chat(
                noxus_msgs,
                model=resolved,
                web_search=web_search,
                stream=True,
            )
            return SyncStream(raw, resolved)

        # non-streaming: use chat() so images are passed through
        result = self.noxus.chat(
            noxus_msgs,
            model=resolved,
            web_search=web_search,
            stream=False,
        )
        content = result.content if isinstance(result, NoxusResponse) else str(result)
        msg = Message(role="assistant", content=content)
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

    def _eris_create(
        self,
        resolved: str,
        messages: list[dict],
        web_search: bool,
        stream: bool,
        think_harder: bool = False,
    ) -> ChatCompletion | Iterator[ChatCompletionChunk]:
        if stream:
            raw = self.eris.chat(
                messages,
                model=resolved,
                stream=True,
                enable_thinking=think_harder,
            )
            if isinstance(raw, ErisStream):
                return ErisSyncStream(raw, resolved)
            return SyncStream(iter([raw.content]), resolved)

        result = self.eris.chat(
            messages,
            model=resolved,
            enable_thinking=think_harder,
        )
        if isinstance(result, ErisStream):
            content = "".join(c for c, _ in result if not isinstance(c, tuple))
        else:
            content = result.content
        msg = Message(role="assistant", content=content)
        choice = Choice(index=0, message=msg)
        return ChatCompletion(id="", model=resolved, choices=(choice,))

    def _telnyx_create(
        self,
        resolved: str,
        messages: list[dict],
        web_search: bool,
        stream: bool,
        think_harder: bool = False,
        max_tokens: int | None = None,
    ) -> ChatCompletion | Iterator[ChatCompletionChunk]:
        if stream:
            raw = self.telnyx.chat(
                messages,
                model=resolved,
                stream=True,
                enable_thinking=think_harder,
                max_tokens=max_tokens,
            )
            if isinstance(raw, TelnyxStream):
                return TelnyxSyncStream(raw, resolved)
            return SyncStream(iter([raw.content]), resolved)

        result = self.telnyx.chat(
            messages,
            model=resolved,
            enable_thinking=think_harder,
            max_tokens=max_tokens,
        )
        if isinstance(result, TelnyxStream):
            content = "".join(c for c, _ in result if not isinstance(c, tuple))
        else:
            content = result.content
        msg = Message(role="assistant", content=content)
        choice = Choice(index=0, message=msg)
        return ChatCompletion(id="", model=resolved, choices=(choice,))

    def _kai_create(
        self,
        resolved: str,
        messages: list[dict],
        stream: bool,
        tools: list[dict] | None = None,
        tool_choice: object = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> ChatCompletion | Iterator[ChatCompletionChunk]:
        if stream:
            raw = self.kai.chat(
                messages,
                model=resolved,
                stream=True,
                tools=tools,
                tool_choice=tool_choice,
                max_tokens=max_tokens or 16384,
                temperature=0.7 if temperature is None else temperature,
            )
            if isinstance(raw, KaiStream):
                return KaiSyncStream(raw, resolved)
            return SyncStream(iter([raw.content]), resolved)

        result = self.kai.chat(
            messages,
            model=resolved,
            tools=tools,
            tool_choice=tool_choice,
            max_tokens=max_tokens or 16384,
            temperature=0.7 if temperature is None else temperature,
        )
        tool_call_objs = tuple(
            ToolCall(
                id=tc.get("id", ""),
                type=tc.get("type", "function"),
                function=FunctionCall(
                    name=tc.get("name", ""),
                    arguments=tc.get("arguments", ""),
                ),
            )
            for tc in (result.tool_calls or ())
            if isinstance(tc, dict)
        )
        msg = Message(
            role="assistant",
            content=result.content,
            tool_calls=tool_call_objs,
        )
        finish = result.finish_reason or ("tool_calls" if tool_call_objs else "stop")
        choice = Choice(index=0, message=msg, finish_reason=finish)
        return ChatCompletion(id="", model=resolved, choices=(choice,))

    def create(
        self,
        *,
        model: str,
        messages: list[dict],
        web_search: bool = False,
        stream: bool = False,
        think_harder: bool = False,
        tools: list[dict] | None = None,
        tool_choice: object = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> ChatCompletion | Iterator[ChatCompletionChunk]:
        resolved = resolve_model(model)
        messages = _normalize_messages(messages)
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
        if provider == "eris":
            return self._eris_create(
                resolved, messages, web_search, stream, think_harder=think_harder
            )
        if provider == "telnyx":
            return self._telnyx_create(
                resolved,
                messages,
                web_search,
                stream,
                think_harder=think_harder,
                max_tokens=max_tokens,
            )
        if provider == "kai":
            return self._kai_create(
                resolved,
                messages,
                stream,
                tools=tools,
                tool_choice=tool_choice,
                max_tokens=max_tokens,
                temperature=temperature,
            )

        return self._noxus_create(resolved, messages, web_search, stream)


class AsyncCompletions:
    """Async version of :class:`Completions`."""

    __slots__ = (
        "noxus",
        "deepai",
        "quillbot",
        "notrack",
        "dphnai",
        "yqcloud",
        "opera",
        "eris",
        "telnyx",
        "kai",
    )

    def __init__(
        self,
        noxus: Noxus,
        deepai: DeepAI,
        quillbot: Quillbot,
        notrack: NoTrack,
        dphnai: DphnAI,
        yqcloud: Yqcloud,
        opera: OperaAria,
        eris: Eris,
        telnyx: Telnyx,
        kai: Kai,
    ) -> None:
        self.noxus = noxus
        self.deepai = deepai
        self.quillbot = quillbot
        self.notrack = notrack
        self.dphnai = dphnai
        self.yqcloud = yqcloud
        self.opera = opera
        self.eris = eris
        self.telnyx = telnyx
        self.kai = kai

    async def _noxus_create(
        self,
        resolved: str,
        messages: list[dict],
        web_search: bool,
        stream: bool,
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        from fishr.Loop import asyncio

        noxus_msgs = Completions._build_messages(messages)

        if stream:
            raw = await asyncio.to_thread(
                self.noxus.chat,
                noxus_msgs,
                model=resolved,
                web_search=web_search,
                stream=True,
            )
            return AsyncStream(raw, resolved)

        # non-streaming: use chat() so images are passed through
        result = await asyncio.to_thread(
            self.noxus.chat,
            noxus_msgs,
            model=resolved,
            web_search=web_search,
            stream=False,
        )
        content = result.content if isinstance(result, NoxusResponse) else str(result)
        msg = Message(role="assistant", content=content)
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

    async def _eris_create(
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
                self.eris.chat,
                messages,
                model=resolved,
                stream=True,
                enable_thinking=think_harder,
            )
            if isinstance(raw, ErisStream):
                return ErisAsyncStream(raw, resolved)
            return AsyncStream(iter([raw.content]), resolved)

        result = await asyncio.to_thread(
            self.eris.chat,
            messages,
            model=resolved,
            enable_thinking=think_harder,
        )
        if isinstance(result, ErisStream):
            content = "".join(c for c, _ in result if not isinstance(c, tuple))
        else:
            content = result.content
        msg = Message(role="assistant", content=content)
        choice = Choice(index=0, message=msg)
        return ChatCompletion(id="", model=resolved, choices=(choice,))

    async def _telnyx_create(
        self,
        resolved: str,
        messages: list[dict],
        web_search: bool,
        stream: bool,
        think_harder: bool = False,
        max_tokens: int | None = None,
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        from fishr.Loop import asyncio

        if stream:
            raw = await asyncio.to_thread(
                self.telnyx.chat,
                messages,
                model=resolved,
                stream=True,
                enable_thinking=think_harder,
                max_tokens=max_tokens,
            )
            if isinstance(raw, TelnyxStream):
                return TelnyxAsyncStream(raw, resolved)
            return AsyncStream(iter([raw.content]), resolved)

        result = await asyncio.to_thread(
            self.telnyx.chat,
            messages,
            model=resolved,
            enable_thinking=think_harder,
            max_tokens=max_tokens,
        )
        if isinstance(result, TelnyxStream):
            content = "".join(c for c, _ in result if not isinstance(c, tuple))
        else:
            content = result.content
        msg = Message(role="assistant", content=content)
        choice = Choice(index=0, message=msg)
        return ChatCompletion(id="", model=resolved, choices=(choice,))

    async def _kai_create(
        self,
        resolved: str,
        messages: list[dict],
        stream: bool,
        tools: list[dict] | None = None,
        tool_choice: object = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        from fishr.Loop import asyncio

        if stream:
            raw = await asyncio.to_thread(
                self.kai.chat,
                messages,
                model=resolved,
                stream=True,
                tools=tools,
                tool_choice=tool_choice,
                max_tokens=max_tokens or 16384,
                temperature=0.7 if temperature is None else temperature,
            )
            if isinstance(raw, KaiStream):
                return KaiAsyncStream(raw, resolved)
            return AsyncStream(iter([raw.content]), resolved)

        result = await asyncio.to_thread(
            self.kai.chat,
            messages,
            model=resolved,
            tools=tools,
            tool_choice=tool_choice,
            max_tokens=max_tokens or 16384,
            temperature=0.7 if temperature is None else temperature,
        )
        tool_call_objs = tuple(
            ToolCall(
                id=tc.get("id", ""),
                type=tc.get("type", "function"),
                function=FunctionCall(
                    name=tc.get("name", ""),
                    arguments=tc.get("arguments", ""),
                ),
            )
            for tc in (result.tool_calls or ())
            if isinstance(tc, dict)
        )
        msg = Message(
            role="assistant",
            content=result.content,
            tool_calls=tool_call_objs,
        )
        finish = result.finish_reason or ("tool_calls" if tool_call_objs else "stop")
        choice = Choice(index=0, message=msg, finish_reason=finish)
        return ChatCompletion(id="", model=resolved, choices=(choice,))

    async def create(
        self,
        *,
        model: str,
        messages: list[dict],
        web_search: bool = False,
        stream: bool = False,
        think_harder: bool = False,
        tools: list[dict] | None = None,
        tool_choice: object = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        resolved = resolve_model(model)
        messages = _normalize_messages(messages)
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
        if provider == "eris":
            return await self._eris_create(
                resolved, messages, web_search, stream, think_harder=think_harder
            )
        if provider == "telnyx":
            return await self._telnyx_create(
                resolved,
                messages,
                web_search,
                stream,
                think_harder=think_harder,
                max_tokens=max_tokens,
            )
        if provider == "kai":
            return await self._kai_create(
                resolved,
                messages,
                stream,
                tools=tools,
                tool_choice=tool_choice,
                max_tokens=max_tokens,
                temperature=temperature,
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
        eris: Eris,
        telnyx: Telnyx,
        kai: Kai,
    ) -> None:
        self.completions = Completions(
            noxus, deepai, quillbot, notrack, dphnai, yqcloud, opera, eris, telnyx, kai
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
        eris: Eris,
        telnyx: Telnyx,
        kai: Kai,
    ) -> None:
        self.completions = AsyncCompletions(
            noxus, deepai, quillbot, notrack, dphnai, yqcloud, opera, eris, telnyx, kai
        )


__all__ = [
    "Completions",
    "AsyncCompletions",
    "Chat",
    "AsyncChat",
]
