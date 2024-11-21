# any-llm-client

A unified and lightweight asynchronous Python API for communicating with LLMs. It supports multiple providers, including OpenAI Chat Completions API (and any OpenAI-compatible API, such as Ollama and vLLM) and YandexGPT API.

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

import httpx
import pydantic

import any_llm_client


config = any_llm_client.OpenAIConfig(
    url=pydantic.HttpUrl("http://127.0.0.1:11434/v1/chat/completions"),
    model_name="qwen2.5-coder:1.5b",
)


async def main() -> None:
    async with httpx.AsyncClient() as httpx_client:
        response = await any_llm_client.get_client(config, httpx_client=httpx_client).request_llm_response(
            messages=[
                any_llm_client.Message(role="system", text="Ты — опытный ассистент"),
                any_llm_client.Message(role="user", text="Привет!"),
            ],
            temperature=0.1,
        )
        print(response)  # type(response) is str


asyncio.run(main())
```

To use `YandexGPT`, replace the config:

```python
config = any_llm_client.YandexGPTConfig(
    auth_header=os.environ["YANDEX_AUTH_HEADER"],
    folder_id=os.environ["YANDEX_FOLDER_ID"],
    model_name="yandexgpt",
)
```

### Streaming API

LLMs often take long time to respond fully. Here's an example of streaming API usage:

```python
import asyncio
import sys

import httpx
import pydantic

import any_llm_client


config = any_llm_client.OpenAIConfig(
    url=pydantic.HttpUrl("http://127.0.0.1:11434/v1/chat/completions"),
    model_name="qwen2.5-coder:1.5b",
)


async def main() -> None:
    async with (
        httpx.AsyncClient() as httpx_client,
        any_llm_client.get_client(config, httpx_client=httpx_client).stream_llm_partial_responses(
            messages=[
                any_llm_client.Message(role="system", text="Ты — опытный ассистент"),
                any_llm_client.Message(role="user", text="Привет!"),
            ],
            temperature=0.1,
        ) as partial_messages,
    ):
        async for one_message in partial_messages:  # type(one_message) is str
            sys.stdout.write(f"\r{one_message}")
            sys.stdout.flush()


asyncio.run(main())
```

Note that this will yield partial growing message, not message chunks, for example: "Hi", "Hi there!", "Hi there! How can I help you?".

### Other

#### Mock client

You can use a mock client for testing:

```python
config = any_llm_client.MockLLMConfig(
    response_message=...,
    stream_messages=["Hi!"],
)
llm_client = any_llm_client.get_client(config, ...)
```

#### Using dynamic LLM config from environment with [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)

```python
import os

import pydantic_settings

import any_llm_client


class Settings(pydantic_settings.BaseSettings):
    llm_model: any_llm_client.AnyLLMConfig


os.environ["LLM_MODEL"] = """{
    "api_type": "openai",
    "url": "http://127.0.0.1:11434/v1/chat/completions",
    "model_name": "qwen2.5-coder:1.5b"
}"""
settings = Settings()
client = any_llm_client.get_client(settings.llm_model, ...)
```

#### Using clients directly

The recommended way to get LLM client is to call `any_llm_client.get_client()`. This way you can easily swap LLM models. If you prefer, you can use `any_llm_client.OpenAIClient` or `any_llm_client.YandexGPTClient` directly:

```python
config = any_llm_client.OpenAIConfig(
    url=pydantic.HttpUrl("https://api.openai.com/v1/chat/completions"),
    auth_token=os.environ["OPENAI_API_KEY"],
    model_name="gpt-4o-mini",
)
llm_client = any_llm_client.OpenAIClient(config, ...)
```

#### Errors

`any_llm_client.LLMClient.request_llm_response()` and `any_llm_client.LLMClient.stream_llm_partial_responses()` will raise `any_llm_client.LLMError` or `any_llm_client.OutOfTokensOrSymbolsError` when the LLM API responds with a failed HTTP status.

#### Retries

By default, requests are retried 3 times on HTTP status errors. You can change the retry behaviour by supplying `request_retry` parameter:

```python
llm_client = any_llm_client.get_client(..., request_retry=any_llm_client.RequestRetryConfig(attempts=5, ...))
```

#### Timeouts and proxy

Configure timeouts or proxy directly in `httpx.AsyncClient()`:

```python
import httpx

import any_llm_client


async with httpx.AsyncClient(
    proxies={
        "https://api.openai.com": httpx.HTTPTransport(proxy="http://localhost:8030"),
    },
    timeout=httpx.Timeout(None, connect=5.0),
) as httpx_client:
    llm_client = any_llm_client.get_client(..., httpx_client=httpx_client)
```
