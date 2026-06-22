from msgspec import Struct

from fishr.Base.DeepAI import DeepAI, DeepAIStream
from fishr.Base.Noxus import Noxus
from fishr.client.routing import provider_of, resolve_model


class AgentStep(Struct, frozen=True):
    role: str
    content: str
    tool: str = ""
    tool_input: str = ""


class AgentRun:
    """Run a multi-step agent loop via ``client.agents.run(...)``.

    The agent will iteratively call the LLM with tool results until
    it produces a final answer or reaches the max step limit.

    Usage::

        ```py
        from fishr import Client

        client = Client()
        result = client.agents.run(
            model="noxus/openai",
            instructions="You are a helpful research assistant.",
            prompt="What are the latest developments in quantum computing?",
            tools=[{"type": "web_search"}],
            max_steps=5,
        )
        print(result.content)```
    """

    __slots__ = (
        "noxus",
        "deepai",
    )

    def __init__(self, noxus: Noxus, deepai: DeepAI) -> None:
        self.noxus = noxus
        self.deepai = deepai

    def run(
        self,
        *,
        model: str = "noxus/openai",
        instructions: str = "",
        prompt: str,
        tools: list[dict] | None = None,
        max_steps: int = 5,
    ) -> AgentStep:
        resolved = resolve_model(model)
        provider = provider_of(resolved)
        use_web = bool(tools and any(t.get("type") == "web_search" for t in tools))
        history: list[dict] = []
        if instructions:
            history.append({"role": "system", "content": instructions})
        history.append({"role": "user", "content": prompt})

        for step in range(max_steps):
            if provider == "deepai":
                result = self.deepai.ask(
                    history[-1]["content"],
                    model=resolved,
                    web_search=use_web,
                )
                if isinstance(result, DeepAIStream):
                    content = "".join(result)
                else:
                    content = result.image_url if result.image_url else result.content
            else:
                content = self.noxus.ask(
                    history[-1]["content"],
                    model=resolved,
                    web_search=use_web,
                )

            history.append({"role": "assistant", "content": content})

            tool_calls = self._extract_tool_calls(content)
            if not tool_calls:
                return AgentStep(role="assistant", content=content)

            for call in tool_calls:
                history.append(
                    {
                        "role": "tool",
                        "content": call.get("result", ""),
                    }
                )

        last = history[-1]
        return AgentStep(
            role="assistant",
            content=last.get("content", ""),
        )

    @staticmethod
    def _extract_tool_calls(content: str) -> list[dict]:
        if "tool_call" not in content:
            return []
        return [{"result": content}]


class AsyncAgentRun:
    """Async version of :class:`AgentRun`.

    Usage::

        ```py
        from fishr import AsyncClient

        client = AsyncClient()
        result = await client.agents.run(
            model="noxus/openai",
            instructions="You are a helpful research assistant.",
            prompt="What are the latest developments in quantum computing?",
            tools=[{"type": "web_search"}],
            max_steps=5,
        )
        print(result.content)```
    """

    __slots__ = (
        "noxus",
        "deepai",
    )

    def __init__(self, noxus: Noxus, deepai: DeepAI) -> None:
        self.noxus = noxus
        self.deepai = deepai

    async def run(
        self,
        *,
        model: str = "noxus/openai",
        instructions: str = "",
        prompt: str,
        tools: list[dict] | None = None,
        max_steps: int = 5,
    ) -> AgentStep:
        from fishr.Loop import asyncio

        resolved = resolve_model(model)
        provider = provider_of(resolved)
        use_web = bool(tools and any(t.get("type") == "web_search" for t in tools))
        history: list[dict] = []
        if instructions:
            history.append({"role": "system", "content": instructions})
        history.append({"role": "user", "content": prompt})

        for step in range(max_steps):
            if provider == "deepai":
                result = await asyncio.to_thread(
                    self.deepai.ask,
                    history[-1]["content"],
                    model=resolved,
                    web_search=use_web,
                )
                if isinstance(result, DeepAIStream):
                    content = "".join(result)
                else:
                    content = result.image_url if result.image_url else result.content
            else:
                content = await self.noxus.ask_async(
                    history[-1]["content"],
                    model=resolved,
                    web_search=use_web,
                )

            history.append({"role": "assistant", "content": content})

            tool_calls = AgentRun._extract_tool_calls(content)
            if not tool_calls:
                return AgentStep(role="assistant", content=content)

            for call in tool_calls:
                history.append(
                    {
                        "role": "tool",
                        "content": call.get("result", ""),
                    }
                )

        last = history[-1]
        return AgentStep(
            role="assistant",
            content=last.get("content", ""),
        )


__all__ = [
    "AgentRun",
    "AsyncAgentRun",
    "AgentStep",
]
