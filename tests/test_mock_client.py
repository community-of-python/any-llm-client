import typing
from unittest import mock

from polyfactory.factories.pydantic_factory import ModelFactory

import any_llm_client
from tests.conftest import LLMFuncRequestFactory, consume_llm_partial_responses


class MockLLMConfigFactory(ModelFactory[any_llm_client.MockLLMConfig]): ...


async def test_mock_client_request_llm_response_returns_config_value() -> None:
    config: typing.Final = MockLLMConfigFactory.build()
    response: typing.Final = await any_llm_client.get_client(config, httpx_client=mock.Mock()).request_llm_response(
        **LLMFuncRequestFactory.build()
    )
    assert response == config.response_message


async def test_mock_client_request_llm_partial_responses_returns_config_value() -> None:
    config: typing.Final = MockLLMConfigFactory.build()
    response: typing.Final = await consume_llm_partial_responses(
        any_llm_client.get_client(config, httpx_client=mock.Mock()).stream_llm_partial_responses(
            **LLMFuncRequestFactory.build()
        )
    )
    assert response == config.stream_messages
