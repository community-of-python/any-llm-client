import typing

import pytest
from polyfactory.factories.pydantic_factory import ModelFactory

import any_llm_client
from tests.conftest import LLMFuncRequest, LLMFuncRequestFactory, consume_llm_message_chunks


class MockLLMConfigFactory(ModelFactory[any_llm_client.MockLLMConfig]): ...


@pytest.mark.parametrize("func_request", LLMFuncRequestFactory.coverage())
async def test_mock_client_request_llm_message_returns_config_value(func_request: LLMFuncRequest) -> None:
    config: typing.Final = MockLLMConfigFactory.build()
    response: typing.Final = await any_llm_client.get_client(config).request_llm_message(**func_request)
    assert response == config.response_message


@pytest.mark.parametrize("func_request", LLMFuncRequestFactory.coverage())
async def test_mock_client_stream_llm_message_chunks_returns_config_value(func_request: LLMFuncRequest) -> None:
    config: typing.Final = MockLLMConfigFactory.build()
    response: typing.Final = await consume_llm_message_chunks(
        any_llm_client.get_client(config).stream_llm_message_chunks(**func_request),
    )
    assert response == config.stream_messages
