import typing
from unittest import mock

import faker
import httpx
import pydantic
import pytest
from polyfactory.factories.pydantic_factory import ModelFactory

import any_llm
from any_llm.abc import Message
from any_llm.clients.openai import (
    ChatCompletionsMessage,
    ChatCompletionsNotStreamingResponse,
    ChatCompletionsStreamingEvent,
    OneNotStreamingChoice,
    OneStreamingChoice,
    OneStreamingChoiceDelta,
    OpenAIClient,
)
from tests.conftest import consume_llm_partial_responses
from tests.factories import LLMRequestFactory


class OpenAILLMConfigFactory(ModelFactory[any_llm.OpenAIConfig]): ...


class TestOpenAIRequestLLMResponse:
    async def test_ok(self, faker: faker.Faker) -> None:
        expected_result = faker.pystr()
        response = httpx.Response(
            200,
            json=ChatCompletionsNotStreamingResponse(
                choices=[
                    OneNotStreamingChoice(message=ChatCompletionsMessage(role="assistant", content=expected_result))
                ]
            ).model_dump(mode="json"),
        )

        result: typing.Final = await any_llm.get_model(
            OpenAILLMConfigFactory.build(),
            httpx_client=httpx.AsyncClient(transport=httpx.MockTransport(lambda _: response)),
        ).request_llm_response(**LLMRequestFactory.build())

        assert result == expected_result

    async def test_fails_without_alternatives(self) -> None:
        response = httpx.Response(
            200,
            json=ChatCompletionsNotStreamingResponse.model_construct(choices=[]).model_dump(mode="json"),
        )
        client = any_llm.get_model(
            OpenAILLMConfigFactory.build(),
            httpx_client=httpx.AsyncClient(transport=httpx.MockTransport(lambda _: response)),
        )

        with pytest.raises(pydantic.ValidationError):
            await client.request_llm_response(**LLMRequestFactory.build())


class TestOpenAIRequestLLMPartialResponses:
    async def test_ok(self, faker: faker.Faker) -> None:
        generated_messages: typing.Final = [
            OneStreamingChoiceDelta(role="assistant"),
            OneStreamingChoiceDelta(content="H"),
            OneStreamingChoiceDelta(content="i"),
            OneStreamingChoiceDelta(content=" t"),
            OneStreamingChoiceDelta(role="assistant", content="here"),
            OneStreamingChoiceDelta(),
            OneStreamingChoiceDelta(content=". How is you"),
            OneStreamingChoiceDelta(content="r day?"),
            OneStreamingChoiceDelta(),
        ]
        expected_result: typing.Final = [
            "H",
            "Hi",
            "Hi t",
            "Hi there",
            "Hi there. How is you",
            "Hi there. How is your day?",
        ]
        config = OpenAILLMConfigFactory.build()
        func_request = LLMRequestFactory.build()
        response_content = (
            "\n\n".join(
                "data: "
                + ChatCompletionsStreamingEvent(choices=[OneStreamingChoice(delta=one_message)]).model_dump_json()
                for one_message in generated_messages
            )
            + f"\n\ndata: [DONE]\n\ndata: {faker.pystr()}\n\n"
        )
        response = httpx.Response(200, headers={"Content-Type": "text/event-stream"}, content=response_content)
        client = any_llm.get_model(
            config,
            httpx_client=httpx.AsyncClient(transport=httpx.MockTransport(lambda _: response)),
        )

        result: typing.Final = await consume_llm_partial_responses(client.request_llm_partial_responses(**func_request))

        assert result == expected_result

    async def test_fails_without_alternatives(self) -> None:
        response_content = f"data: {ChatCompletionsStreamingEvent.model_construct(choices=[]).model_dump_json()}\n\n"
        response = httpx.Response(200, headers={"Content-Type": "text/event-stream"}, content=response_content)
        client = any_llm.get_model(
            OpenAILLMConfigFactory.build(),
            httpx_client=httpx.AsyncClient(transport=httpx.MockTransport(lambda _: response)),
        )

        with pytest.raises(pydantic.ValidationError):
            await consume_llm_partial_responses(client.request_llm_partial_responses(**LLMRequestFactory.build()))


class TestOpenAILLMErrors:
    @pytest.mark.parametrize("stream", [True, False])
    @pytest.mark.parametrize("status_code", [400, 500])
    async def test_fails_with_unknown_error(self, stream: bool, status_code: int) -> None:
        client = any_llm.get_model(
            OpenAILLMConfigFactory.build(),
            httpx_client=httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(status_code))),
        )

        coroutine = (
            consume_llm_partial_responses(client.request_llm_partial_responses(**LLMRequestFactory.build()))
            if stream
            else client.request_llm_response(**LLMRequestFactory.build())
        )

        with pytest.raises(any_llm.LLMError) as exc_info:
            await coroutine
        assert type(exc_info.value) is any_llm.LLMError

    @pytest.mark.parametrize("stream", [True, False])
    @pytest.mark.parametrize(
        "content",
        [
            b'{"object":"error","message":"This model\'s maximum context length is 4096 tokens. However, you requested 5253 tokens in the messages, Please reduce the length of the messages.","type":"BadRequestError","param":null,"code":400}',  # noqa: E501
            b'{"object":"error","message":"This model\'s maximum context length is 16384 tokens. However, you requested 100000 tokens in the messages, Please reduce the length of the messages.","type":"BadRequestError","param":null,"code":400}',  # noqa: E501
        ],
    )
    async def test_fails_with_out_of_tokens_error(self, stream: bool, content: bytes | None) -> None:
        response = httpx.Response(400, content=content)
        client = any_llm.get_model(
            OpenAILLMConfigFactory.build(),
            httpx_client=httpx.AsyncClient(transport=httpx.MockTransport(lambda _: response)),
        )

        coroutine = (
            consume_llm_partial_responses(client.request_llm_partial_responses(**LLMRequestFactory.build()))
            if stream
            else client.request_llm_response(**LLMRequestFactory.build())
        )

        with pytest.raises(any_llm.OutOfTokensOrSymbolsError):
            await coroutine


class TestOpenAIMessageAlternation:
    @pytest.mark.parametrize(
        ("messages", "expected_result"),
        [
            ([], []),
            ([Message(role="system", text="")], []),
            ([Message(role="system", text=" ")], []),
            ([Message(role="user", text="")], []),
            ([Message(role="assistant", text="")], []),
            ([Message(role="system", text=""), Message(role="user", text="")], []),
            ([Message(role="system", text=""), Message(role="assistant", text="")], []),
            (
                [
                    Message(role="system", text=""),
                    Message(role="user", text=""),
                    Message(role="assistant", text=""),
                    Message(role="assistant", text=""),
                    Message(role="user", text=""),
                    Message(role="assistant", text=""),
                ],
                [],
            ),
            ([Message(role="system", text="Be nice")], [ChatCompletionsMessage(role="user", content="Be nice")]),
            (
                [Message(role="user", text="Hi there"), Message(role="assistant", text="Hi! How can I help you?")],
                [
                    ChatCompletionsMessage(role="user", content="Hi there"),
                    ChatCompletionsMessage(role="assistant", content="Hi! How can I help you?"),
                ],
            ),
            (
                [
                    Message(role="system", text=""),
                    Message(role="user", text="Hi there"),
                    Message(role="assistant", text="Hi! How can I help you?"),
                ],
                [
                    ChatCompletionsMessage(role="user", content="Hi there"),
                    ChatCompletionsMessage(role="assistant", content="Hi! How can I help you?"),
                ],
            ),
            (
                [Message(role="system", text="Be nice"), Message(role="user", text="Hi there")],
                [ChatCompletionsMessage(role="user", content="Be nice\n\nHi there")],
            ),
            (
                [
                    Message(role="system", text="Be nice"),
                    Message(role="assistant", text="Hi!"),
                    Message(role="assistant", text="I'm your answer to everything."),
                    Message(role="assistant", text="How can I help you?"),
                    Message(role="user", text="Hi there"),
                    Message(role="user", text=""),
                    Message(role="user", text="Why is the sky blue?"),
                    Message(role="assistant", text=" "),
                    Message(role="assistant", text="Well..."),
                    Message(role="assistant", text=""),
                    Message(role="assistant", text=" \n "),
                    Message(role="user", text="Hmmm..."),
                ],
                [
                    ChatCompletionsMessage(role="user", content="Be nice"),
                    ChatCompletionsMessage(
                        role="assistant",
                        content="Hi!\n\nI'm your answer to everything.\n\nHow can I help you?",
                    ),
                    ChatCompletionsMessage(role="user", content="Hi there\n\nWhy is the sky blue?"),
                    ChatCompletionsMessage(role="assistant", content="Well..."),
                    ChatCompletionsMessage(role="user", content="Hmmm..."),
                ],
            ),
        ],
    )
    def test_with_alternation(self, messages: list[Message], expected_result: list[ChatCompletionsMessage]) -> None:
        client = OpenAIClient(
            config=OpenAILLMConfigFactory.build(force_user_assistant_message_alternation=True), httpx_client=mock.Mock()
        )
        assert client._prepare_messages(messages) == expected_result  # noqa: SLF001

    def test_without_alternation(self) -> None:
        client = OpenAIClient(
            config=OpenAILLMConfigFactory.build(force_user_assistant_message_alternation=False),
            httpx_client=mock.Mock(),
        )
        assert client._prepare_messages(  # noqa: SLF001
            [Message(role="system", text="Be nice"), Message(role="user", text="Hi there")]
        ) == [
            ChatCompletionsMessage(role="system", content="Be nice"),
            ChatCompletionsMessage(role="user", content="Hi there"),
        ]
