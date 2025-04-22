# any-llm-client

A unified and lightweight asynchronous Python API for communicating with LLMs.

Supports multiple providers, including OpenAI Chat Completions API (and any OpenAI-compatible API, such as Ollama and vLLM) and YandexGPT API.

## How To Use

Before starting using any-llm-client, make sure you have it installed:

```sh
uv add any-llm-client
poetry add any-llm-client
```

### Response API

Here's a full example that uses Ollama and Qwen2.5-Coder:

```python
import asyncio

import any_llm_client


config = any_llm_client.OpenAIConfig(
    url="http://127.0.0.1:11434/v1/chat/completions",
    model_name="qwen2.5-coder:1.5b",
    request_extra={"best_of": 3}
)


async def main() -> None:
    async with any_llm_client.get_client(config) as client:
        print(await client.request_llm_message("Кек, чо как вообще на нарах?"))


asyncio.run(main())
```

To use `YandexGPT`, replace the config:

```python
config = any_llm_client.YandexGPTConfig(
    auth_header=os.environ["YANDEX_AUTH_HEADER"], folder_id=os.environ["YANDEX_FOLDER_ID"], model_name="yandexgpt"
)
```

### Streaming API

LLMs often take long time to respond fully. Here's an example of streaming API usage:

```python
import asyncio

import any_llm_client


config = any_llm_client.OpenAIConfig(
    url="http://127.0.0.1:11434/v1/chat/completions",
    model_name="qwen2.5-coder:1.5b",
    request_extra={"best_of": 3}
)


async def main() -> None:
    async with (
        any_llm_client.get_client(config) as client,
        client.stream_llm_message_chunks("Кек, чо как вообще на нарах?") as message_chunks,
    ):
        async for chunk in message_chunks:
            print(chunk, end="", flush=True)


asyncio.run(main())
```

### Passing chat history and temperature

You can pass list of messages instead of `str` as the first argument, and set `temperature`:

```python
async with (
    any_llm_client.get_client(config) as client,
    client.stream_llm_message_chunks(
        messages=[
            any_llm_client.SystemMessage("Ты — опытный ассистент"),
            any_llm_client.UserMessage("Кек, чо как вообще на нарах?"),
        ],
        temperature=1.0,
    ) as message_chunks,
):
    ...
```

### Reasoning models

Today you can access openapi-like reasoning models and retrieve their reasoning content:

```python
async def main() -> None:
    async with any_llm_client.get_client(config) as client:
        llm_response = await client.request_llm_message("Кек, чо как вообще на нарах?")
        print(f"Just a regular LLM response content: {llm_response.content}")
        print(f"LLM reasoning response content: {llm_response.reasoning_content}")

    ...
```

### Other

#### Mock client

You can use a mock client for testing:

```python
config = any_llm_client.MockLLMConfig(
    response_message=...,
    stream_messages=["Hi!"],
)

async with any_llm_client.get_client(config, ...) as client:
    ...
```

#### Configuration with environment variables

##### Credentials

Instead of passing credentials directly, you can set corresponding environment variables:

- OpenAI: `ANY_LLM_CLIENT_OPENAI_AUTH_TOKEN`,
- YandexGPT: `ANY_LLM_CLIENT_YANDEXGPT_AUTH_HEADER`, `ANY_LLM_CLIENT_YANDEXGPT_FOLDER_ID`.

##### LLM model config (with [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/))

```python
import os

import pydantic_settings

import any_llm_client


class Settings(pydantic_settings.BaseSettings):
    llm_model: any_llm_client.AnyLLMConfig


os.environ["LLM_MODEL"] = """{
    "api_type": "openai",
    "url": "http://127.0.0.1:11434/v1/chat/completions",
    "model_name": "qwen2.5-coder:1.5b",
    "request_extra": {"best_of": 3}
}"""
settings = Settings()

async with any_llm_client.get_client(settings.llm_model, ...) as client:
    ...
```

Combining with environment variables from previous section, you can keep LLM model configuration and secrets separate.

#### Using clients directly

The recommended way to get LLM client is to call `any_llm_client.get_client()`. This way you can easily swap LLM models. If you prefer, you can use `any_llm_client.OpenAIClient` or `any_llm_client.YandexGPTClient` directly:

```python
config = any_llm_client.OpenAIConfig(
    url=pydantic.HttpUrl("https://api.openai.com/v1/chat/completions"),
    auth_token=os.environ["OPENAI_API_KEY"],
    model_name="gpt-4o-mini",
    request_extra={"best_of": 3}
)

async with any_llm_client.OpenAIClient(config, ...) as client:
    ...
```

#### Errors

`any_llm_client.LLMClient.request_llm_message()` and `any_llm_client.LLMClient.stream_llm_message_chunks()` will raise:

- `any_llm_client.LLMError` or `any_llm_client.OutOfTokensOrSymbolsError` when the LLM API responds with a failed HTTP status,
- `any_llm_client.LLMRequestValidationError` when images are passed to YandexGPT client.
- `any_llm_client.LLMResponseValidationError` when invalid response come from LLM API (reraised from `pydantic.ValidationError`).

All these exceptions inherit from the base class `any_llm_client.AnyLLMClientError`.

#### Timeouts, proxy & other HTTP settings

Pass custom [HTTPX](https://www.python-httpx.org) kwargs to `any_llm_client.get_client()`:

```python
import httpx

import any_llm_client


async with any_llm_client.get_client(
    ...,
    mounts={"https://api.openai.com": httpx.AsyncHTTPTransport(proxy="http://localhost:8030")},
    timeout=httpx.Timeout(None, connect=5.0),
) as client:
    ...
```

Default timeout is `httpx.Timeout(None, connect=5.0)` (5 seconds on connect, unlimited on read, write or pool).

#### Retries

By default, requests are retried 3 times on HTTP status errors. You can change the retry behaviour by supplying `request_retry` parameter:

```python
async with any_llm_client.get_client(..., request_retry=any_llm_client.RequestRetryConfig(attempts=5, ...)) as client:
    ...
```

#### Passing extra data to LLM

```python
await client.request_llm_message("Кек, чо как вообще на нарах?", extra={"best_of": 3})
```

The `extra` parameter is united with `request_extra` in OpenAIConfig

#### Passing images

You can pass images to OpenAI client (YandexGPT doesn't support images yet):

```python
await client.request_llm_message(
    messages=[
        any_llm_client.TextContentItem("What's on the image?"),
        any_llm_client.ImageContentItem("https://upload.wikimedia.org/wikipedia/commons/a/a9/Example.jpg"),
    ]
)
```

You can also pass a data url with base64-encoded image:

```python
await client.request_llm_message(
    messages=[
        any_llm_client.TextContentItem("What's on the image?"),
        any_llm_client.ImageContentItem(
            f"data:image/jpeg;base64,{base64.b64encode(image_content_bytes).decode('utf-8')}"
        ),
    ]
)
```
