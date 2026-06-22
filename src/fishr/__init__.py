"""fishr — free LLM access from Python.

Usage::

    ```py
    from fishr import Client

    client = Client()

    # single completion
    response = client.chat.completions.create(
        model="noxus/openai",
        messages=[{"role": "user", "content": "Hello!"}],
    )
    print(response.text)

    # streaming
    response = client.chat.completions.create(
        model="noxus/openai",
        messages=[{"role": "user", "content": "Tell me a story"}],
        stream=True,
    )
    for chunk in response:
        print(chunk.choices[0].delta.content or "", end="")

    # web search
    response = client.chat.completions.create(
        model="noxus/grok-4.3",
        messages=[{"role": "user", "content": "Latest news on SpaceX"}],
        web_search=True,
    )

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

    # async
    from fishr import AsyncClient

    client = AsyncClient()
    response = await client.chat.completions.create(
        model="deepai/standard",
        messages=[{"role": "user", "content": "Hello!"}],
    )

    # multi-turn conversation
    conv = client.conversation(model="noxus/openai")
    conv.ask("Remember: 42")
    conv.ask("What number did I say?")```

Models are specified as ``provider/name``:

    Noxus:
    - noxus/openai, noxus/google, noxus/sonnet-4.6, noxus/sonnet-3.5
    - noxus/grok-4.3, noxus/perplexity, noxus/metaai, noxus/qwen

    DeepAI:
    - deepai/standard, deepai/online, deepai/gemma-4
    - deepai/gemini-2.5-flash-lite, deepai/deepseek-v3.2, deepai/image

    Quillbot:
    - quillbot/quillbot, quillbot/quillbot-search

    NoTrack:
    - notrack/fast, notrack/standard, notrack/reasoning

    DphnAI:
    - dphnai/24b, dphnai/6b

    Yqcloud:
    - yqcloud/gpt-4

    Opera Aria:
    - opera/aria

The provider prefix is optional for noxus (default).
"""

from fishr.Base.Conversation import AsyncConversation, Conversation
from fishr.Base.DeepAI import DeepAI, DeepAIResponse, DeepAIStream
from fishr.Base.DphnAI import DphnAI, DphnAIResponse, DphnAIStream
from fishr.Base.NoTrack import NoTrack, NoTrackResponse, NoTrackStream
from fishr.Base.Noxus import Noxus, NoxusMessage, NoxusResponse
from fishr.Base.OperaAria import OperaAria, OperaAriaResponse, OperaAriaStream
from fishr.Base.Quillbot import Quillbot, QuillbotResponse, QuillbotStream
from fishr.Base.Yqcloud import Yqcloud, YqcloudResponse, YqcloudStream
from fishr.client import (
    AgentRun,
    AgentStep,
    AsyncAgentRun,
    AsyncClient,
    AsyncCompletions,
    AsyncImages,
    Client,
    Completions,
    Images,
)
from fishr.Types import (
    ChatCompletion,
    ChatCompletionChunk,
    Choice,
    ChunkChoice,
    Delta,
    ImageResponse,
    ImageUrl,
    Message,
    ModelDef,
    Usage,
    models,
)

__all__ = [
    "Client",
    "AsyncClient",
    "Conversation",
    "AsyncConversation",
    "Completions",
    "AsyncCompletions",
    "Images",
    "AsyncImages",
    "AgentRun",
    "AsyncAgentRun",
    "AgentStep",
    "Noxus",
    "NoxusResponse",
    "NoxusMessage",
    "DeepAI",
    "DeepAIResponse",
    "DeepAIStream",
    "DphnAI",
    "DphnAIResponse",
    "DphnAIStream",
    "Quillbot",
    "QuillbotResponse",
    "QuillbotStream",
    "NoTrack",
    "NoTrackResponse",
    "NoTrackStream",
    "Yqcloud",
    "YqcloudResponse",
    "YqcloudStream",
    "OperaAria",
    "OperaAriaResponse",
    "OperaAriaStream",
    "ChatCompletion",
    "ChatCompletionChunk",
    "Choice",
    "ChunkChoice",
    "Delta",
    "Message",
    "ModelDef",
    "Usage",
    "ImageUrl",
    "ImageResponse",
    "models",
]
