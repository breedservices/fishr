from fishr.Base.DeepAI import DeepAI
from fishr.Base.DphnAI import DphnAI
from fishr.Base.NoTrack import NoTrack
from fishr.Base.Noxus import NoxusMessage
from fishr.Base.OperaAria import OperaAria
from fishr.Base.Quillbot import Quillbot
from fishr.Base.Yqcloud import Yqcloud
from fishr.Types import models as model_registry


class Conversation:
    """Sync multi-turn conversation with history.

    Usage::

        from fishr import Client

        client = Client()
        conv = client.conversation(model="noxus/openai")

        conv.system("You are a helpful assistant.")
        conv.ask("What is Python?")
        conv.ask("Tell me more about that.")

        # works with all providers
        conv = client.conversation(model="deepai/standard")
        conv.ask("Hello!")

        conv = client.conversation(model="quillbot/quillbot")
        conv.ask("Hello!")
    """

    __slots__ = (
        "provider",
        "provider_type",
        "history",
        "model",
        "supports_history",
    )

    def __init__(self, provider, model: str = "noxus/openai") -> None:
        self.provider = provider
        if isinstance(provider, DeepAI):
            self.provider_type = "deepai"
        elif isinstance(provider, DphnAI):
            self.provider_type = "dphnai"
        elif isinstance(provider, Quillbot):
            self.provider_type = "quillbot"
        elif isinstance(provider, NoTrack):
            self.provider_type = "notrack"
        elif isinstance(provider, Yqcloud):
            self.provider_type = "yqcloud"
        elif isinstance(provider, OperaAria):
            self.provider_type = "opera"
        else:
            self.provider_type = "noxus"
        self.model = model
        self.history: list[NoxusMessage] = []
        model_def = model_registry.get(model)
        self.supports_history = model_def.history if model_def else True

    def ask(self, prompt: str, **kwargs) -> str:
        if self.provider_type == "deepai":
            return self._ask_deepai(prompt, **kwargs)
        if self.provider_type == "dphnai":
            return self._ask_dphnai(prompt, **kwargs)
        if self.provider_type == "quillbot":
            return self._ask_quillbot(prompt, **kwargs)
        if self.provider_type == "notrack":
            return self._ask_notrack(prompt, **kwargs)
        if self.provider_type == "yqcloud":
            return self._ask_yqcloud(prompt, **kwargs)
        if self.provider_type == "opera":
            return self._ask_opera(prompt, **kwargs)
        return self._ask_noxus(prompt, **kwargs)

    def _ask_noxus(self, prompt: str, **kwargs) -> str:
        user_msg = NoxusMessage(role="user", content=prompt)
        self.history.append(user_msg)
        if self.supports_history:
            resp = self.provider.chat(tuple(self.history), model=self.model, **kwargs)
        else:
            resp = self.provider.chat((user_msg,), model=self.model, **kwargs)
        assistant_msg = NoxusMessage(role="assistant", content=resp.content)
        self.history.append(assistant_msg)
        return resp.content

    def _ask_deepai(self, prompt: str, **kwargs) -> str:
        result = self.provider.ask(prompt, model=self.model, **kwargs)
        content = result.image_url if result.image_url else result.content
        self.history.append(NoxusMessage(role="user", content=prompt))
        self.history.append(NoxusMessage(role="assistant", content=content))
        return content

    def _ask_quillbot(self, prompt: str, **kwargs) -> str:
        messages = [{"role": m.role, "content": m.content} for m in self.history]
        messages.append({"role": "user", "content": prompt})
        result = self.provider.chat(messages, model=self.model, **kwargs)
        content = result.content
        self.history.append(NoxusMessage(role="user", content=prompt))
        self.history.append(NoxusMessage(role="assistant", content=content))
        return content

    def _ask_notrack(self, prompt: str, **kwargs) -> str:
        messages = [{"role": m.role, "content": m.content} for m in self.history]
        messages.append({"role": "user", "content": prompt})
        result = self.provider.chat(messages, model=self.model, **kwargs)
        content = result.content
        self.history.append(NoxusMessage(role="user", content=prompt))
        self.history.append(NoxusMessage(role="assistant", content=content))
        return content

    def _ask_dphnai(self, prompt: str, **kwargs) -> str:
        messages = [{"role": m.role, "content": m.content} for m in self.history]
        messages.append({"role": "user", "content": prompt})
        result = self.provider.chat(messages, model=self.model, **kwargs)
        content = result.content
        self.history.append(NoxusMessage(role="user", content=prompt))
        self.history.append(NoxusMessage(role="assistant", content=content))
        return content

    def _ask_yqcloud(self, prompt: str, **kwargs) -> str:
        messages = [{"role": m.role, "content": m.content} for m in self.history]
        messages.append({"role": "user", "content": prompt})
        result = self.provider.chat(messages, model=self.model, **kwargs)
        content = result.content
        self.history.append(NoxusMessage(role="user", content=prompt))
        self.history.append(NoxusMessage(role="assistant", content=content))
        return content

    def _ask_opera(self, prompt: str, **kwargs) -> str:
        messages = [{"role": m.role, "content": m.content} for m in self.history]
        messages.append({"role": "user", "content": prompt})
        result = self.provider.chat(messages, model=self.model, **kwargs)
        content = result.content
        if result.image_urls:
            content += "\n" + "\n".join(result.image_urls)
        self.history.append(NoxusMessage(role="user", content=prompt))
        self.history.append(NoxusMessage(role="assistant", content=content))
        return content

    def system(self, content: str) -> None:
        self.history.append(NoxusMessage(role="system", content=content))

    def clear(self) -> None:
        self.history.clear()
        if hasattr(self.provider, "new_chat"):
            self.provider.new_chat()

    def __len__(self) -> int:
        return len(self.history)

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        pass


class AsyncConversation:
    """Async multi-turn conversation with history.

    Usage::

        ```py
        from fishr import AsyncClient

        client = AsyncClient()
        conv = client.conversation(model="noxus/openai")

        await conv.ask("What is Python?")
        await conv.ask("Tell me more about that.")```
    """

    __slots__ = (
        "provider",
        "provider_type",
        "history",
        "model",
        "supports_history",
    )

    def __init__(self, provider, model: str = "noxus/openai") -> None:
        self.provider = provider
        if isinstance(provider, DeepAI):
            self.provider_type = "deepai"
        elif isinstance(provider, DphnAI):
            self.provider_type = "dphnai"
        elif isinstance(provider, Quillbot):
            self.provider_type = "quillbot"
        elif isinstance(provider, NoTrack):
            self.provider_type = "notrack"
        elif isinstance(provider, Yqcloud):
            self.provider_type = "yqcloud"
        elif isinstance(provider, OperaAria):
            self.provider_type = "opera"
        else:
            self.provider_type = "noxus"
        self.model = model
        self.history: list[NoxusMessage] = []
        model_def = model_registry.get(model)
        self.supports_history = model_def.history if model_def else True

    async def ask(self, prompt: str, **kwargs) -> str:
        if self.provider_type == "deepai":
            return await self._ask_deepai(prompt, **kwargs)
        if self.provider_type == "dphnai":
            return await self._ask_dphnai(prompt, **kwargs)
        if self.provider_type == "quillbot":
            return await self._ask_quillbot(prompt, **kwargs)
        if self.provider_type == "notrack":
            return await self._ask_notrack(prompt, **kwargs)
        if self.provider_type == "yqcloud":
            return await self._ask_yqcloud(prompt, **kwargs)
        if self.provider_type == "opera":
            return await self._ask_opera(prompt, **kwargs)
        return await self._ask_noxus(prompt, **kwargs)

    async def _ask_noxus(self, prompt: str, **kwargs) -> str:
        user_msg = NoxusMessage(role="user", content=prompt)
        self.history.append(user_msg)
        if self.supports_history:
            resp = await self.provider.chat_async(
                tuple(self.history),
                model=self.model,
                **kwargs,
            )
        else:
            resp = await self.provider.chat_async(
                (user_msg,),
                model=self.model,
                **kwargs,
            )
        assistant_msg = NoxusMessage(role="assistant", content=resp.content)
        self.history.append(assistant_msg)
        return resp.content

    async def _ask_deepai(self, prompt: str, **kwargs) -> str:
        result = await self.provider.ask_async(prompt, model=self.model, **kwargs)
        content = result.image_url if result.image_url else result.content
        self.history.append(NoxusMessage(role="user", content=prompt))
        self.history.append(NoxusMessage(role="assistant", content=content))
        return content

    async def _ask_quillbot(self, prompt: str, **kwargs) -> str:
        messages = [{"role": m.role, "content": m.content} for m in self.history]
        messages.append({"role": "user", "content": prompt})
        result = await self.provider.chat_async(
            messages,
            model=self.model,
            **kwargs,
        )
        content = result.content
        self.history.append(NoxusMessage(role="user", content=prompt))
        self.history.append(NoxusMessage(role="assistant", content=content))
        return content

    async def _ask_notrack(self, prompt: str, **kwargs) -> str:
        messages = [{"role": m.role, "content": m.content} for m in self.history]
        messages.append({"role": "user", "content": prompt})
        result = await self.provider.chat_async(
            messages,
            model=self.model,
            **kwargs,
        )
        content = result.content
        self.history.append(NoxusMessage(role="user", content=prompt))
        self.history.append(NoxusMessage(role="assistant", content=content))
        return content

    async def _ask_dphnai(self, prompt: str, **kwargs) -> str:
        messages = [{"role": m.role, "content": m.content} for m in self.history]
        messages.append({"role": "user", "content": prompt})
        result = await self.provider.chat_async(
            messages,
            model=self.model,
            **kwargs,
        )
        content = result.content
        self.history.append(NoxusMessage(role="user", content=prompt))
        self.history.append(NoxusMessage(role="assistant", content=content))
        return content

    async def _ask_yqcloud(self, prompt: str, **kwargs) -> str:
        messages = [{"role": m.role, "content": m.content} for m in self.history]
        messages.append({"role": "user", "content": prompt})
        result = await self.provider.chat_async(
            messages,
            model=self.model,
            **kwargs,
        )
        content = result.content
        self.history.append(NoxusMessage(role="user", content=prompt))
        self.history.append(NoxusMessage(role="assistant", content=content))
        return content

    async def _ask_opera(self, prompt: str, **kwargs) -> str:
        messages = [{"role": m.role, "content": m.content} for m in self.history]
        messages.append({"role": "user", "content": prompt})
        result = await self.provider.chat_async(
            messages,
            model=self.model,
            **kwargs,
        )
        content = result.content
        if result.image_urls:
            content += "\n" + "\n".join(result.image_urls)
        self.history.append(NoxusMessage(role="user", content=prompt))
        self.history.append(NoxusMessage(role="assistant", content=content))
        return content

    def system(self, content: str) -> None:
        self.history.append(NoxusMessage(role="system", content=content))

    def clear(self) -> None:
        self.history.clear()
        if hasattr(self.provider, "new_chat"):
            self.provider.new_chat()

    def __len__(self) -> int:
        return len(self.history)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args) -> None:
        pass


__all__ = [
    "Conversation",
    "AsyncConversation",
]
