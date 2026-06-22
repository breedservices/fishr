from typing import AsyncIterator, Iterator

from fishr.Base.DeepAI import DeepAIStream
from fishr.Base.DphnAI import DphnAIStream
from fishr.Base.NoTrack import NoTrackStream
from fishr.Base.OperaAria import OperaAriaStream
from fishr.Base.Quillbot import QuillbotStream
from fishr.Base.Yqcloud import YqcloudStream
from fishr.Types import (
    ChatCompletionChunk,
    ChunkChoice,
    Delta,
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
        content = next(self.inner)
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
]
