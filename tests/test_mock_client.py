import typing

from polyfactory.factories.pydantic_factory import ModelFactory

import any_llm_client
from tests.conftest import LLMFuncRequestFactory, consume_llm_message_chunks


class MockLLMConfigFactory(ModelFactory[any_llm_client.MockLLMConfig]): ...


async def test_mock_client_request_llm_message_returns_config_value() -> None:
    config: typing.Final = MockLLMConfigFactory.build()
    response: typing.Final = await any_llm_client.get_client(config).request_llm_message(
        **LLMFuncRequestFactory.build()
    )
    assert response == config.response_message


async def test_mock_client_stream_llm_message_chunks_returns_config_value() -> None:
    config: typing.Final = MockLLMConfigFactory.build()
    response: typing.Final = await consume_llm_message_chunks(
        any_llm_client.get_client(config).stream_llm_message_chunks(**LLMFuncRequestFactory.build())
    )
    assert response == config.stream_messages
