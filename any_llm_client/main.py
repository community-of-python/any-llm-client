import functools
import typing

import httpx

from any_llm_client.abc import LLMClient
from any_llm_client.clients.mock import MockLLMClient, MockLLMConfig
from any_llm_client.clients.openai import OpenAIClient, OpenAIConfig
from any_llm_client.clients.yandexgpt import YandexGPTClient, YandexGPTConfig


AnyLLMConfig = YandexGPTConfig | OpenAIConfig | MockLLMConfig


if typing.TYPE_CHECKING:

    def get_client(config: AnyLLMConfig, *, httpx_client: httpx.AsyncClient) -> LLMClient: ...  # pragma: no cover
else:

    @functools.singledispatch
    def get_client(config: typing.Any, *, httpx_client: httpx.AsyncClient) -> LLMClient:  # noqa: ANN401, ARG001
        raise AssertionError("unknown LLM config type")

    @get_client.register
    def _(config: YandexGPTConfig, *, httpx_client: httpx.AsyncClient) -> LLMClient:
        return YandexGPTClient(config=config, httpx_client=httpx_client)

    @get_client.register
    def _(config: OpenAIConfig, *, httpx_client: httpx.AsyncClient) -> LLMClient:
        return OpenAIClient(config=config, httpx_client=httpx_client)

    @get_client.register
    def _(config: MockLLMConfig, *, httpx_client: httpx.AsyncClient) -> LLMClient:  # noqa: ARG001
        return MockLLMClient(config=config)
