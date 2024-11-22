import functools
import typing

import pydantic

from any_llm_client.clients.mock import MockLLMClient, MockLLMConfig
from any_llm_client.clients.openai import OpenAIClient, OpenAIConfig
from any_llm_client.clients.yandexgpt import YandexGPTClient, YandexGPTConfig
from any_llm_client.core import LLMClient
from any_llm_client.retry import RequestRetryConfig


AnyLLMConfig = typing.Annotated[YandexGPTConfig | OpenAIConfig | MockLLMConfig, pydantic.Discriminator("api_type")]


if typing.TYPE_CHECKING:

    def get_client(
        config: AnyLLMConfig,
        *,
        request_retry: RequestRetryConfig | None = None,
        **httpx_kwargs: typing.Any,  # noqa: ANN401
    ) -> LLMClient: ...
else:

    @functools.singledispatch
    def get_client(
        config: typing.Any,  # noqa: ANN401, ARG001
        *,
        request_retry: RequestRetryConfig | None = None,  # noqa: ARG001
        **httpx_kwargs: typing.Any,  # noqa: ANN401, ARG001
    ) -> LLMClient:
        raise AssertionError("unknown LLM config type")

    @get_client.register
    def _(
        config: YandexGPTConfig,
        *,
        request_retry: RequestRetryConfig | None = None,
        **httpx_kwargs: typing.Any,  # noqa: ANN401
    ) -> LLMClient:
        return YandexGPTClient(config=config, request_retry=request_retry, **httpx_kwargs)

    @get_client.register
    def _(
        config: OpenAIConfig,
        *,
        request_retry: RequestRetryConfig | None = None,
        **httpx_kwargs: typing.Any,  # noqa: ANN401
    ) -> LLMClient:
        return OpenAIClient(config=config, request_retry=request_retry, **httpx_kwargs)

    @get_client.register
    def _(
        config: MockLLMConfig,
        *,
        request_retry: RequestRetryConfig | None = None,  # noqa: ARG001
        **httpx_kwargs: typing.Any,  # noqa: ANN401, ARG001
    ) -> LLMClient:
        return MockLLMClient(config=config)
