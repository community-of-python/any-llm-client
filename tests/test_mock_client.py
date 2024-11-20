from unittest import mock

from polyfactory.factories.pydantic_factory import ModelFactory

import any_llm
from tests.conftest import consume_llm_partial_responses
from tests.factories import LLMRequestFactory


class MockLLMConfigFactory(ModelFactory[any_llm.MockLLMConfig]): ...


async def test_mock_client_request_llm_response_returns_config_value() -> None:
    config = MockLLMConfigFactory.build()
    response = await any_llm.get_model(config, httpx_client=mock.Mock()).request_llm_response(
        **LLMRequestFactory.build()
    )
    assert response == config.response_message


async def test_mock_client_request_llm_partial_responses_returns_config_value() -> None:
    config = MockLLMConfigFactory.build()
    response = await consume_llm_partial_responses(
        any_llm.get_model(config, httpx_client=mock.Mock()).request_llm_partial_responses(**LLMRequestFactory.build())
    )
    assert response == config.stream_messages
