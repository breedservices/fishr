from typing import AsyncIterator, Iterator

from fishr.Base.DeepAI import DeepAIStream
from fishr.Base.DphnAI import DphnAIStream
from fishr.Base.Eris import ErisStream
from fishr.Base.Kai import KaiStream
from fishr.Base.NoTrack import NoTrackStream
from fishr.Base.OperaAria import OperaAriaStream
from fishr.Base.Quillbot import QuillbotStream
from fishr.Base.Telnyx import TelnyxStream
from fishr.Base.Yqcloud import YqcloudStream
from fishr.Types import (
    ChatCompletionChunk,
    ChunkChoice,
    Delta,
    FunctionCall,
    ToolCallDelta,
)


class SyncStream:
    __slots__ = ("inner", "model")

    def __init__(self, raw, model: str) -> None:
        self.inner = iter(raw)
        self.model = model

    def __iter__(self) -> Iterator[ChatCompletionChunk]:
        return self

    def __next__(self) -> ChatCompletionChunk:
        content = next(self.inner)
        delta = Delta(role="assistant", content=content)
        choice = ChunkChoice(index=0, delta=delta)
        return ChatCompletionChunk(
            id="",
            model=self.model,
            choices=(choice,),
        )

    def __del__(self) -> None:
        if hasattr(self.inner, "close"):
            self.inner.close()


class AsyncStream:
    __slots__ = ("inner", "model")

    def __init__(self, raw, model: str) -> None:
        self.inner = raw.__aiter__() if hasattr(raw, "__aiter__") else raw
        self.model = model

    def __aiter__(self) -> AsyncIterator[ChatCompletionChunk]:
        return self

    async def __anext__(self) -> ChatCompletionChunk:
        try:
            content = await self.inner.__anext__()
        except StopAsyncIteration:
            raise
        delta = Delta(role="assistant", content=content)
        choice = ChunkChoice(index=0, delta=delta)
        return ChatCompletionChunk(
            id="",
            model=self.model,
            choices=(choice,),
        )


class DeepAISyncStream:
    __slots__ = ("inner", "model")

    def __init__(self, raw: DeepAIStream, model: str) -> None:
        self.inner = iter(raw)
        self.model = model

    def __iter__(self) -> Iterator[ChatCompletionChunk]:
        return self

    def __next__(self) -> ChatCompletionChunk:
        content = next(self.inner)
        delta = Delta(role="assistant", content=content)
        choice = ChunkChoice(index=0, delta=delta)
        return ChatCompletionChunk(
            id="",
            model=self.model,
            choices=(choice,),
        )


class DeepAIAsyncStream:
    __slots__ = ("inner", "model")

    def __init__(self, raw: DeepAIStream, model: str) -> None:
        self.inner = raw.__aiter__()
        self.model = model

    def __aiter__(self) -> AsyncIterator[ChatCompletionChunk]:
        return self

    async def __anext__(self) -> ChatCompletionChunk:
        content = await self.inner.__anext__()
        delta = Delta(role="assistant", content=content)
        choice = ChunkChoice(index=0, delta=delta)
        return ChatCompletionChunk(
            id="",
            model=self.model,
            choices=(choice,),
        )


class QuillbotSyncStream:
    __slots__ = ("inner", "model")

    def __init__(self, raw: QuillbotStream, model: str) -> None:
        self.inner = iter(raw)
        self.model = model

    def __iter__(self) -> Iterator[ChatCompletionChunk]:
        return self

    def __next__(self) -> ChatCompletionChunk:
        content = next(self.inner)
        delta = Delta(role="assistant", content=content)
        choice = ChunkChoice(index=0, delta=delta)
        return ChatCompletionChunk(
            id="",
            model=self.model,
            choices=(choice,),
        )


class QuillbotAsyncStream:
    __slots__ = ("inner", "model")

    def __init__(self, raw: QuillbotStream, model: str) -> None:
        self.inner = raw.__aiter__() if hasattr(raw, "__aiter__") else raw
        self.model = model

    def __aiter__(self) -> AsyncIterator[ChatCompletionChunk]:
        return self

    async def __anext__(self) -> ChatCompletionChunk:
        try:
            content = await self.inner.__anext__()
        except StopAsyncIteration:
            raise
        delta = Delta(role="assistant", content=content)
        choice = ChunkChoice(index=0, delta=delta)
        return ChatCompletionChunk(
            id="",
            model=self.model,
            choices=(choice,),
        )


class NoTrackSyncStream:
    __slots__ = ("inner", "model")

    def __init__(self, raw: NoTrackStream, model: str) -> None:
        self.inner = iter(raw)
        self.model = model

    def __iter__(self) -> Iterator[ChatCompletionChunk]:
        return self

    def __next__(self) -> ChatCompletionChunk:
        content = next(self.inner)
        delta = Delta(role="assistant", content=content)
        choice = ChunkChoice(index=0, delta=delta)
        return ChatCompletionChunk(
            id="",
            model=self.model,
            choices=(choice,),
        )


class NoTrackAsyncStream:
    __slots__ = ("inner", "model")

    def __init__(self, raw: NoTrackStream, model: str) -> None:
        self.inner = raw.__aiter__() if hasattr(raw, "__aiter__") else raw
        self.model = model

    def __aiter__(self) -> AsyncIterator[ChatCompletionChunk]:
        return self

    async def __anext__(self) -> ChatCompletionChunk:
        try:
            content = await self.inner.__anext__()
        except StopAsyncIteration:
            raise
        delta = Delta(role="assistant", content=content)
        choice = ChunkChoice(index=0, delta=delta)
        return ChatCompletionChunk(
            id="",
            model=self.model,
            choices=(choice,),
        )


class DphnAISyncStream:
    __slots__ = ("inner", "model")

    def __init__(self, raw: DphnAIStream, model: str) -> None:
        self.inner = iter(raw)
        self.model = model

    def __iter__(self) -> Iterator[ChatCompletionChunk]:
        return self

    def __next__(self) -> ChatCompletionChunk:
        content = next(self.inner)
        delta = Delta(role="assistant", content=content)
        choice = ChunkChoice(index=0, delta=delta)
        return ChatCompletionChunk(
            id="",
            model=self.model,
            choices=(choice,),
        )


class DphnAIAsyncStream:
    __slots__ = ("inner", "model")

    def __init__(self, raw: DphnAIStream, model: str) -> None:
        self.inner = raw.__aiter__()
        self.model = model

    def __aiter__(self) -> AsyncIterator[ChatCompletionChunk]:
        return self

    async def __anext__(self) -> ChatCompletionChunk:
        content = await self.inner.__anext__()
        delta = Delta(role="assistant", content=content)
        choice = ChunkChoice(index=0, delta=delta)
        return ChatCompletionChunk(
            id="",
            model=self.model,
            choices=(choice,),
        )


class YqcloudSyncStream:
    __slots__ = ("inner", "model")

    def __init__(self, raw: YqcloudStream, model: str) -> None:
        self.inner = iter(raw)
        self.model = model

    def __iter__(self) -> Iterator[ChatCompletionChunk]:
        return self

    def __next__(self) -> ChatCompletionChunk:
        content = next(self.inner)
        delta = Delta(role="assistant", content=content)
        choice = ChunkChoice(index=0, delta=delta)
        return ChatCompletionChunk(
            id="",
            model=self.model,
            choices=(choice,),
        )


class YqcloudAsyncStream:
    __slots__ = ("inner", "model")

    def __init__(self, raw: YqcloudStream, model: str) -> None:
        self.inner = raw.__aiter__() if hasattr(raw, "__aiter__") else raw
        self.model = model

    def __aiter__(self) -> AsyncIterator[ChatCompletionChunk]:
        return self

    async def __anext__(self) -> ChatCompletionChunk:
        try:
            content = await self.inner.__anext__()
        except StopAsyncIteration:
            raise
        delta = Delta(role="assistant", content=content)
        choice = ChunkChoice(index=0, delta=delta)
        return ChatCompletionChunk(
            id="",
            model=self.model,
            choices=(choice,),
        )


class OperaAriaSyncStream:
    __slots__ = ("inner", "model")

    def __init__(self, raw: OperaAriaStream, model: str) -> None:
        self.inner = iter(raw)
        self.model = model

    def __iter__(self) -> Iterator[ChatCompletionChunk]:
        return self

    def __next__(self) -> ChatCompletionChunk:
        item = next(self.inner)
        if isinstance(item, tuple):
            content, is_thinking = item
        else:
            content, is_thinking = item, False
        if is_thinking:
            delta = Delta(role="assistant", thinking=content)
        else:
            delta = Delta(role="assistant", content=content)
        choice = ChunkChoice(index=0, delta=delta)
        return ChatCompletionChunk(
            id="",
            model=self.model,
            choices=(choice,),
        )


class OperaAriaAsyncStream:
    __slots__ = ("inner", "model")

    def __init__(self, raw: OperaAriaStream, model: str) -> None:
        self.inner = raw.__aiter__() if hasattr(raw, "__aiter__") else raw
        self.model = model

    def __aiter__(self) -> AsyncIterator[ChatCompletionChunk]:
        return self

    async def __anext__(self) -> ChatCompletionChunk:
        try:
            item = await self.inner.__anext__()
        except StopAsyncIteration:
            raise
        if isinstance(item, tuple):
            content, is_thinking = item
        else:
            content, is_thinking = item, False
        if is_thinking:
            delta = Delta(role="assistant", thinking=content)
        else:
            delta = Delta(role="assistant", content=content)
        choice = ChunkChoice(index=0, delta=delta)
        return ChatCompletionChunk(
            id="",
            model=self.model,
            choices=(choice,),
        )


class ErisSyncStream:
    __slots__ = ("inner", "model")

    def __init__(self, raw: ErisStream, model: str) -> None:
        self.inner = iter(raw)
        self.model = model

    def __iter__(self) -> Iterator[ChatCompletionChunk]:
        return self

    def __next__(self) -> ChatCompletionChunk:
        item = next(self.inner)
        if isinstance(item, tuple):
            content, is_thinking = item
        else:
            content, is_thinking = item, False
        if is_thinking:
            delta = Delta(role="assistant", thinking=content)
        else:
            delta = Delta(role="assistant", content=content)
        choice = ChunkChoice(index=0, delta=delta)
        return ChatCompletionChunk(
            id="",
            model=self.model,
            choices=(choice,),
        )


class TelnyxSyncStream:
    __slots__ = ("inner", "model")

    def __init__(self, raw: TelnyxStream, model: str) -> None:
        self.inner = iter(raw)
        self.model = model

    def __iter__(self) -> Iterator[ChatCompletionChunk]:
        return self

    def __next__(self) -> ChatCompletionChunk:
        item = next(self.inner)
        if isinstance(item, tuple):
            content, is_thinking = item
        else:
            content, is_thinking = item, False
        if is_thinking:
            delta = Delta(role="assistant", thinking=content)
        else:
            delta = Delta(role="assistant", content=content)
        choice = ChunkChoice(index=0, delta=delta)
        return ChatCompletionChunk(
            id="",
            model=self.model,
            choices=(choice,),
        )


class ErisAsyncStream:
    __slots__ = ("inner", "model")

    def __init__(self, raw: ErisStream, model: str) -> None:
        self.inner = raw.__aiter__() if hasattr(raw, "__aiter__") else raw
        self.model = model

    def __aiter__(self) -> AsyncIterator[ChatCompletionChunk]:
        return self

    async def __anext__(self) -> ChatCompletionChunk:
        try:
            item = await self.inner.__anext__()
        except StopAsyncIteration:
            raise
        if isinstance(item, tuple):
            content, is_thinking = item
        else:
            content, is_thinking = item, False
        if is_thinking:
            delta = Delta(role="assistant", thinking=content)
        else:
            delta = Delta(role="assistant", content=content)
        choice = ChunkChoice(index=0, delta=delta)
        return ChatCompletionChunk(
            id="",
            model=self.model,
            choices=(choice,),
        )


class TelnyxAsyncStream:
    __slots__ = ("inner", "model")

    def __init__(self, raw: TelnyxStream, model: str) -> None:
        self.inner = raw.__aiter__() if hasattr(raw, "__aiter__") else raw
        self.model = model

    def __aiter__(self) -> AsyncIterator[ChatCompletionChunk]:
        return self

    async def __anext__(self) -> ChatCompletionChunk:
        try:
            item = await self.inner.__anext__()
        except StopAsyncIteration:
            raise
        if isinstance(item, tuple):
            content, is_thinking = item
        else:
            content, is_thinking = item, False
        if is_thinking:
            delta = Delta(role="assistant", thinking=content)
        else:
            delta = Delta(role="assistant", content=content)
        choice = ChunkChoice(index=0, delta=delta)
        return ChatCompletionChunk(
            id="",
            model=self.model,
            choices=(choice,),
        )


class KaiSyncStream:
    __slots__ = ("inner", "model")

    def __init__(self, raw: KaiStream, model: str) -> None:
        self.inner = iter(raw)
        self.model = model

    def __iter__(self) -> Iterator[ChatCompletionChunk]:
        return self

    def __next__(self) -> ChatCompletionChunk:
        item = next(self.inner)
        if isinstance(item, tuple) and len(item) == 2 and item[1] == "tool_calls":
            tcs, _ = item
            deltas = tuple(
                ToolCallDelta(
                    index=tc.get("index", 0),
                    id=tc.get("id", ""),
                    type=tc.get("type", "function"),
                    function=FunctionCall(
                        name=tc.get("name", ""),
                        arguments=tc.get("arguments", ""),
                    ),
                )
                for tc in tcs
                if isinstance(tc, dict)
            )
            delta = Delta(role="assistant", tool_calls=deltas)
        elif isinstance(item, tuple):
            content, is_thinking = item
            if is_thinking:
                delta = Delta(role="assistant", thinking=content)
            else:
                delta = Delta(role="assistant", content=content)
        else:
            delta = Delta(role="assistant", content=item)
        choice = ChunkChoice(index=0, delta=delta)
        return ChatCompletionChunk(
            id="",
            model=self.model,
            choices=(choice,),
        )


class KaiAsyncStream:
    __slots__ = ("inner", "model")

    def __init__(self, raw: KaiStream, model: str) -> None:
        self.inner = raw.__aiter__() if hasattr(raw, "__aiter__") else raw
        self.model = model

    def __aiter__(self) -> AsyncIterator[ChatCompletionChunk]:
        return self

    async def __anext__(self) -> ChatCompletionChunk:
        try:
            item = await self.inner.__anext__()
        except StopAsyncIteration:
            raise
        if isinstance(item, tuple) and len(item) == 2 and item[1] == "tool_calls":
            tcs, _ = item
            deltas = tuple(
                ToolCallDelta(
                    index=tc.get("index", 0),
                    id=tc.get("id", ""),
                    type=tc.get("type", "function"),
                    function=FunctionCall(
                        name=tc.get("name", ""),
                        arguments=tc.get("arguments", ""),
                    ),
                )
                for tc in tcs
                if isinstance(tc, dict)
            )
            delta = Delta(role="assistant", tool_calls=deltas)
        elif isinstance(item, tuple):
            content, is_thinking = item
            if is_thinking:
                delta = Delta(role="assistant", thinking=content)
            else:
                delta = Delta(role="assistant", content=content)
        else:
            delta = Delta(role="assistant", content=item)
        choice = ChunkChoice(index=0, delta=delta)
        return ChatCompletionChunk(
            id="",
            model=self.model,
            choices=(choice,),
        )


__all__ = [
    "SyncStream",
    "AsyncStream",
    "DeepAISyncStream",
    "DeepAIAsyncStream",
    "QuillbotSyncStream",
    "QuillbotAsyncStream",
    "NoTrackSyncStream",
    "NoTrackAsyncStream",
    "DphnAISyncStream",
    "DphnAIAsyncStream",
    "YqcloudSyncStream",
    "YqcloudAsyncStream",
    "OperaAriaSyncStream",
    "OperaAriaAsyncStream",
    "ErisSyncStream",
    "ErisAsyncStream",
    "TelnyxSyncStream",
    "TelnyxAsyncStream",
    "KaiSyncStream",
    "KaiAsyncStream",
]
