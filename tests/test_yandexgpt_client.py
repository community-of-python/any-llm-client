import typing
from unittest import mock

import faker
import pydantic
import pytest
from polyfactory.factories.pydantic_factory import ModelFactory

import any_llm_client
from any_llm_client.clients.yandexgpt import YandexGPTAlternative, YandexGPTResponse, YandexGPTResult
from any_llm_client.http import HttpStatusError
from tests.conftest import LLMFuncRequestFactory, consume_llm_partial_responses, mock_http_client


class YandexGPTConfigFactory(ModelFactory[any_llm_client.YandexGPTConfig]): ...


class TestYandexGPTRequestLLMResponse:
    async def test_ok(self, faker: faker.Faker) -> None:
        expected_result: typing.Final = faker.pystr()
        response: typing.Final = YandexGPTResponse(
            result=YandexGPTResult(
                alternatives=[YandexGPTAlternative(message=any_llm_client.AssistantMessage(expected_result))]
            )
        ).model_dump_json()
        client: typing.Final = mock_http_client(
            any_llm_client.get_client(YandexGPTConfigFactory.build()), mock.AsyncMock(return_value=response)
        )

        result: typing.Final = await client.request_llm_message(**LLMFuncRequestFactory.build())

        assert result == expected_result

    async def test_fails_without_alternatives(self) -> None:
        response: typing.Final = YandexGPTResponse(
            result=YandexGPTResult.model_construct(alternatives=[])
        ).model_dump_json()
        client: typing.Final = mock_http_client(
            any_llm_client.get_client(YandexGPTConfigFactory.build()), mock.AsyncMock(return_value=response)
        )

        with pytest.raises(pydantic.ValidationError):
            await client.request_llm_message(**LLMFuncRequestFactory.build())


class TestYandexGPTRequestLLMPartialResponses:
    async def test_ok(self, faker: faker.Faker) -> None:
        expected_result: typing.Final = faker.pylist(value_types=[str])
        func_request: typing.Final = LLMFuncRequestFactory.build()
        response: typing.Final = (
            "\n".join(
                YandexGPTResponse(
                    result=YandexGPTResult(
                        alternatives=[YandexGPTAlternative(message=any_llm_client.AssistantMessage(one_text))]
                    )
                ).model_dump_json()
                for one_text in expected_result
            )
            + "\n"
        )
        client: typing.Final = mock_http_client(
            any_llm_client.get_client(YandexGPTConfigFactory.build()), mock.AsyncMock(return_value=response)
        )

        result: typing.Final = await consume_llm_partial_responses(client.stream_llm_partial_messages(**func_request))

        assert result == expected_result

    async def test_fails_without_alternatives(self) -> None:
        response: typing.Final = (
            YandexGPTResponse(result=YandexGPTResult.model_construct(alternatives=[])).model_dump_json() + "\n"
        )
        client: typing.Final = mock_http_client(
            any_llm_client.get_client(YandexGPTConfigFactory.build()), mock.AsyncMock(return_value=response)
        )

        with pytest.raises(pydantic.ValidationError):
            await consume_llm_partial_responses(client.stream_llm_partial_messages(**LLMFuncRequestFactory.build()))


class TestYandexGPTLLMErrors:
    @pytest.mark.parametrize("stream", [True, False])
    @pytest.mark.parametrize("status_code", [400, 500])
    async def test_fails_with_unknown_error(self, faker: faker.Faker, stream: bool, status_code: int) -> None:
        client: typing.Final = mock_http_client(
            any_llm_client.get_client(YandexGPTConfigFactory.build()),
            mock.AsyncMock(side_effect=HttpStatusError(status_code=status_code, content=faker.pystr().encode())),
        )

        coroutine: typing.Final = (
            consume_llm_partial_responses(client.stream_llm_partial_messages(**LLMFuncRequestFactory.build()))
            if stream
            else client.request_llm_message(**LLMFuncRequestFactory.build())
        )

        with pytest.raises(any_llm_client.LLMError) as exc_info:
            await coroutine
        assert type(exc_info.value) is any_llm_client.LLMError

    @pytest.mark.parametrize("stream", [True, False])
    @pytest.mark.parametrize(
        "content",
        [
            b"...folder_id=1111: number of input tokens must be no more than 8192, got 28498...",
            b"...folder_id=1111: text length is 349354, which is outside the range (0, 100000]...",
        ],
    )
    async def test_fails_with_out_of_tokens_error(self, stream: bool, content: bytes) -> None:
        client: typing.Final = mock_http_client(
            any_llm_client.get_client(YandexGPTConfigFactory.build()),
            mock.AsyncMock(side_effect=HttpStatusError(status_code=400, content=content)),
        )

        coroutine: typing.Final = (
            consume_llm_partial_responses(client.stream_llm_partial_messages(**LLMFuncRequestFactory.build()))
            if stream
            else client.request_llm_message(**LLMFuncRequestFactory.build())
        )

        with pytest.raises(any_llm_client.OutOfTokensOrSymbolsError):
            await coroutine
