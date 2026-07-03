# fishr

FREE LLM ACCESS FROM PYTHON. NO KEYS! NO AUTH! NO LIMITS!

## Install

```bash
pip install fishr
```

## Quick Start

```python
from fishr import Client

client = Client()

# Chat completion
response = client.chat.completions.create(
    model="noxus/openai",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.text)

# Streaming
response = client.chat.completions.create(
    model="noxus/openai",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True,
)
for chunk in response:
    print(chunk.choices[0].delta.content or "", end="")

# Web search
response = client.chat.completions.create(
    model="noxus/grok-4.3",
    messages=[{"role": "user", "content": "Latest news on SpaceX"}],
    web_search=True,
)
print(response.text)

# Image generation
result = client.images.generate(
    model="deepai/image",
    prompt="A cat riding a skateboard",
)
print(result.data[0].url)

# Agent
result = client.agents.run(
    model="noxus/openai",
    prompt="Research quantum computing",
    tools=[{"type": "web_search"}],
)
print(result.content)

# Tool calling (kai/* models support native OpenAI-style tools)
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a location",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    },
}]
resp = client.chat.completions.create(
    model="kai/auto",
    messages=[{"role": "user", "content": "What is the weather in Tokyo?"}],
    tools=tools,
)
for tc in resp.choices[0].message.tool_calls:
    print(tc.function.name, tc.function.arguments)

# Multi-turn conversation
conv = client.conversation(model="noxus/openai")
conv.system("You are a helpful assistant.")
conv.ask("Remember: 42")
conv.ask("What number did I say?")
```

## Async

```python
from fishr import AsyncClient

client = AsyncClient()

response = await client.chat.completions.create(
    model="deepai/standard",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.text)

# Async streaming
response = await client.chat.completions.create(
    model="opera/aria",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True,
)
async for chunk in response:
    print(chunk.choices[0].delta.content or "", end="")

# Async conversation
conv = client.conversation(model="noxus/openai")
await conv.ask("Remember: 42")
await conv.ask("What number did I say?")
```

## Direct Provider Access

```python
from fishr import OperaAria, Yqcloud

# Use a provider directly
opera = OperaAria()
result = opera.ask("Hello!", model="opera/aria")
print(result.content)

yqcloud = Yqcloud()
result = yqcloud.chat([{"role": "user", "content": "Hi!"}])
print(result.content)
```

## Models

Models are specified as `provider/name`:

| Provider | Models | Features |
|----------|--------|----------|
| **Noxus** | `noxus/openai`, `noxus/google`, `noxus/sonnet-4.6`, `noxus/sonnet-3.5`, `noxus/grok-4.3`, `noxus/perplexity`, `noxus/metaai`, `noxus/qwen` | Web search, images, history, system messages |
| **DeepAI** | `deepai/standard`, `deepai/online`, `deepai/gemma-4`, `deepai/gemini-2.5-flash-lite`, `deepai/deepseek-v3.2`, `deepai/image` | Web search (online), images, history, system messages |
| **Quillbot** | `quillbot/quillbot`, `quillbot/quillbot-search` | Web search (quillbot-search), history, system messages |
| **NoTrack** | `notrack/fast`, `notrack/standard`, `notrack/reasoning` | History |
| **DphnAI** | `dphnai/24b`, `dphnai/6b` | History, system messages |
| **Yqcloud** | `yqcloud/gpt-4` | History, system messages |
| **Opera Aria** | `opera/aria` | Images, history, system messages |
| **Kai** | `kai/auto`, `kai/m.1`, `kai/xs-2.1`, `kai/xs.2`, `kai/north-mini`, `kai/nemo3-ultra`, `kai/nemo3-super`, `kai/nemo3-nemo`, `kai/3.7-flash`, `kai/openfree` | **Tool calling**, history, system messages (some support images) |

The provider prefix is optional for Noxus (default).

## Requirements

- Python >= 3.11
