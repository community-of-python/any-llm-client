import typing

import faker
import httpx
import pydantic
import pytest
from polyfactory.factories.pydantic_factory import ModelFactory

import any_llm_client
from any_llm_client.clients.openai import (
    ChatCompletionsMessage,
    ChatCompletionsNotStreamingResponse,
    ChatCompletionsStreamingEvent,
    OneNotStreamingChoice,
    OneStreamingChoice,
    OneStreamingChoiceDelta,
)
from tests.conftest import LLMFuncRequestFactory, consume_llm_partial_responses


class OpenAIConfigFactory(ModelFactory[any_llm_client.OpenAIConfig]): ...


class TestOpenAIRequestLLMResponse:
    async def test_ok(self, faker: faker.Faker) -> None:
        expected_result: typing.Final = faker.pystr()
        response: typing.Final = httpx.Response(
            200,
            json=ChatCompletionsNotStreamingResponse(
                choices=[
                    OneNotStreamingChoice(
                        message=ChatCompletionsMessage(
                            role=any_llm_client.MessageRole.assistant, content=expected_result
                        )
                    )
                ]
            ).model_dump(mode="json"),
        )

        result: typing.Final = await any_llm_client.get_client(
            OpenAIConfigFactory.build(), transport=httpx.MockTransport(lambda _: response)
        ).request_llm_message(**LLMFuncRequestFactory.build())

        assert result == expected_result

    async def test_fails_without_alternatives(self) -> None:
        response: typing.Final = httpx.Response(
            200,
            json=ChatCompletionsNotStreamingResponse.model_construct(choices=[]).model_dump(mode="json"),
        )
        client: typing.Final = any_llm_client.get_client(
            OpenAIConfigFactory.build(), transport=httpx.MockTransport(lambda _: response)
        )

        with pytest.raises(pydantic.ValidationError):
            await client.request_llm_message(**LLMFuncRequestFactory.build())


class TestOpenAIRequestLLMPartialResponses:
    async def test_ok(self, faker: faker.Faker) -> None:
        generated_messages: typing.Final = [
            OneStreamingChoiceDelta(role=any_llm_client.MessageRole.assistant),
            OneStreamingChoiceDelta(content="H"),
            OneStreamingChoiceDelta(content="i"),
            OneStreamingChoiceDelta(content=" t"),
            OneStreamingChoiceDelta(role=any_llm_client.MessageRole.assistant, content="here"),
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
        config: typing.Final = OpenAIConfigFactory.build()
        func_request: typing.Final = LLMFuncRequestFactory.build()
        response_content: typing.Final = (
            "\n\n".join(
                "data: "
                + ChatCompletionsStreamingEvent(choices=[OneStreamingChoice(delta=one_message)]).model_dump_json()
                for one_message in generated_messages
            )
            + f"\n\ndata: [DONE]\n\ndata: {faker.pystr()}\n\n"
        )
        response: typing.Final = httpx.Response(
            200, headers={"Content-Type": "text/event-stream"}, content=response_content
        )
        client: typing.Final = any_llm_client.get_client(config, transport=httpx.MockTransport(lambda _: response))

        result: typing.Final = await consume_llm_partial_responses(client.stream_llm_partial_messages(**func_request))

        assert result == expected_result

    async def test_fails_without_alternatives(self) -> None:
        response_content: typing.Final = (
            f"data: {ChatCompletionsStreamingEvent.model_construct(choices=[]).model_dump_json()}\n\n"
        )
        response: typing.Final = httpx.Response(
            200, headers={"Content-Type": "text/event-stream"}, content=response_content
        )
        client: typing.Final = any_llm_client.get_client(
            OpenAIConfigFactory.build(), transport=httpx.MockTransport(lambda _: response)
        )

        with pytest.raises(pydantic.ValidationError):
            await consume_llm_partial_responses(client.stream_llm_partial_messages(**LLMFuncRequestFactory.build()))


class TestOpenAILLMErrors:
    @pytest.mark.parametrize("stream", [True, False])
    @pytest.mark.parametrize("status_code", [400, 500])
    async def test_fails_with_unknown_error(self, stream: bool, status_code: int) -> None:
        client: typing.Final = any_llm_client.get_client(
            OpenAIConfigFactory.build(), transport=httpx.MockTransport(lambda _: httpx.Response(status_code))
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
            b'{"object":"error","message":"This model\'s maximum context length is 4096 tokens. However, you requested 5253 tokens in the messages, Please reduce the length of the messages.","type":"BadRequestError","param":null,"code":400}',  # noqa: E501
            b'{"object":"error","message":"This model\'s maximum context length is 16384 tokens. However, you requested 100000 tokens in the messages, Please reduce the length of the messages.","type":"BadRequestError","param":null,"code":400}',  # noqa: E501
        ],
    )
    async def test_fails_with_out_of_tokens_error(self, stream: bool, content: bytes | None) -> None:
        response: typing.Final = httpx.Response(400, content=content)
        client: typing.Final = any_llm_client.get_client(
            OpenAIConfigFactory.build(), transport=httpx.MockTransport(lambda _: response)
        )

        coroutine: typing.Final = (
            consume_llm_partial_responses(client.stream_llm_partial_messages(**LLMFuncRequestFactory.build()))
            if stream
            else client.request_llm_message(**LLMFuncRequestFactory.build())
        )

        with pytest.raises(any_llm_client.OutOfTokensOrSymbolsError):
            await coroutine


class TestOpenAIMessageAlternation:
    @pytest.mark.parametrize(
        ("messages", "expected_result"),
        [
            ([], []),
            ([any_llm_client.SystemMessage("")], []),
            ([any_llm_client.SystemMessage(" ")], []),
            ([any_llm_client.UserMessage("")], []),
            ([any_llm_client.AssistantMessage("")], []),
            ([any_llm_client.SystemMessage(""), any_llm_client.UserMessage("")], []),
            ([any_llm_client.SystemMessage(""), any_llm_client.AssistantMessage("")], []),
            (
                [
                    any_llm_client.SystemMessage(""),
                    any_llm_client.UserMessage(""),
                    any_llm_client.AssistantMessage(""),
                    any_llm_client.AssistantMessage(""),
                    any_llm_client.UserMessage(""),
                    any_llm_client.AssistantMessage(""),
                ],
                [],
            ),
            (
                [any_llm_client.SystemMessage("Be nice")],
                [ChatCompletionsMessage(role=any_llm_client.MessageRole.user, content="Be nice")],
            ),
            (
                [any_llm_client.UserMessage("Hi there"), any_llm_client.AssistantMessage("Hi! How can I help you?")],
                [
                    ChatCompletionsMessage(role=any_llm_client.MessageRole.user, content="Hi there"),
                    ChatCompletionsMessage(
                        role=any_llm_client.MessageRole.assistant, content="Hi! How can I help you?"
                    ),
                ],
            ),
            (
                [
                    any_llm_client.SystemMessage(""),
                    any_llm_client.UserMessage("Hi there"),
                    any_llm_client.AssistantMessage("Hi! How can I help you?"),
                ],
                [
                    ChatCompletionsMessage(role=any_llm_client.MessageRole.user, content="Hi there"),
                    ChatCompletionsMessage(
                        role=any_llm_client.MessageRole.assistant, content="Hi! How can I help you?"
                    ),
                ],
            ),
            (
                [any_llm_client.SystemMessage("Be nice"), any_llm_client.UserMessage("Hi there")],
                [ChatCompletionsMessage(role=any_llm_client.MessageRole.user, content="Be nice\n\nHi there")],
            ),
            (
                [
                    any_llm_client.SystemMessage("Be nice"),
                    any_llm_client.AssistantMessage("Hi!"),
                    any_llm_client.AssistantMessage("I'm your answer to everything."),
                    any_llm_client.AssistantMessage("How can I help you?"),
                    any_llm_client.UserMessage("Hi there"),
                    any_llm_client.UserMessage(""),
                    any_llm_client.UserMessage("Why is the sky blue?"),
                    any_llm_client.AssistantMessage(" "),
                    any_llm_client.AssistantMessage("Well..."),
                    any_llm_client.AssistantMessage(""),
                    any_llm_client.AssistantMessage(" \n "),
                    any_llm_client.UserMessage("Hmmm..."),
                ],
                [
                    ChatCompletionsMessage(role=any_llm_client.MessageRole.user, content="Be nice"),
                    ChatCompletionsMessage(
                        role=any_llm_client.MessageRole.assistant,
                        content="Hi!\n\nI'm your answer to everything.\n\nHow can I help you?",
                    ),
                    ChatCompletionsMessage(
                        role=any_llm_client.MessageRole.user, content="Hi there\n\nWhy is the sky blue?"
                    ),
                    ChatCompletionsMessage(role=any_llm_client.MessageRole.assistant, content="Well..."),
                    ChatCompletionsMessage(role=any_llm_client.MessageRole.user, content="Hmmm..."),
                ],
            ),
        ],
    )
    def test_with_alternation(
        self, messages: list[any_llm_client.Message], expected_result: list[ChatCompletionsMessage]
    ) -> None:
        client: typing.Final = any_llm_client.OpenAIClient(
            OpenAIConfigFactory.build(force_user_assistant_message_alternation=True)
        )
        assert client._prepare_messages(messages) == expected_result  # noqa: SLF001

    def test_without_alternation(self) -> None:
        client: typing.Final = any_llm_client.OpenAIClient(
            OpenAIConfigFactory.build(force_user_assistant_message_alternation=False)
        )
        assert client._prepare_messages(  # noqa: SLF001
            [any_llm_client.SystemMessage("Be nice"), any_llm_client.UserMessage("Hi there")]
        ) == [
            ChatCompletionsMessage(role=any_llm_client.MessageRole.system, content="Be nice"),
            ChatCompletionsMessage(role=any_llm_client.MessageRole.user, content="Hi there"),
        ]
