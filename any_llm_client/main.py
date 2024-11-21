import functools
import typing

import httpx

from any_llm_client.clients.mock import MockLLMClient, MockLLMConfig
from any_llm_client.clients.openai import OpenAIClient, OpenAIConfig
from any_llm_client.clients.yandexgpt import YandexGPTClient, YandexGPTConfig
from any_llm_client.core import LLMClient
from any_llm_client.retry import RequestRetryConfig


AnyLLMConfig = YandexGPTConfig | OpenAIConfig | MockLLMConfig


if typing.TYPE_CHECKING:

    def get_client(
        config: AnyLLMConfig,
        *,
        httpx_client: httpx.AsyncClient | None = None,
        request_retry: RequestRetryConfig | None = None,
    ) -> LLMClient: ...  # pragma: no cover
else:

    @functools.singledispatch
    def get_client(
        config: typing.Any,  # noqa: ANN401, ARG001
        *,
        httpx_client: httpx.AsyncClient | None = None,  # noqa: ARG001
        request_retry: RequestRetryConfig | None = None,  # noqa: ARG001
    ) -> LLMClient:
        raise AssertionError("unknown LLM config type")

    @get_client.register
    def _(
        config: YandexGPTConfig,
        *,
        httpx_client: httpx.AsyncClient | None = None,
        request_retry: RequestRetryConfig | None = None,
    ) -> LLMClient:
        return YandexGPTClient(config=config, httpx_client=httpx_client, request_retry=request_retry)

    @get_client.register
    def _(
        config: OpenAIConfig,
        *,
        httpx_client: httpx.AsyncClient | None = None,
        request_retry: RequestRetryConfig | None = None,
    ) -> LLMClient:
        return OpenAIClient(config=config, httpx_client=httpx_client, request_retry=request_retry)

    @get_client.register
    def _(
        config: MockLLMConfig,
        *,
        httpx_client: httpx.AsyncClient | None = None,  # noqa: ARG001
        request_retry: RequestRetryConfig | None = None,  # noqa: ARG001
    ) -> LLMClient:
        return MockLLMClient(config=config)
