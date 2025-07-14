import typing

import faker
import httpx
import pytest
from polyfactory.factories.pydantic_factory import ModelFactory

import any_llm_client
from any_llm_client.clients.openai import (
    ChatCompletionsInputMessage,
    ChatCompletionsNotStreamingResponse,
    ChatCompletionsStreamingEvent,
    ChatCompletionsTextContentItem,
    OneNotStreamingChoice,
    OneNotStreamingChoiceMessage,
    OneStreamingChoice,
    OneStreamingChoiceDelta,
)
from any_llm_client.core import LLMResponseValidationError
from tests.conftest import LLMFuncRequest, LLMFuncRequestFactory, consume_llm_message_chunks


class OpenAIConfigFactory(ModelFactory[any_llm_client.OpenAIConfig]): ...


class TestOpenAIRequestLLMResponse:
    @pytest.mark.parametrize("func_request", LLMFuncRequestFactory.coverage())
    async def test_ok(self, func_request: LLMFuncRequest, random_llm_response: any_llm_client.LLMResponse) -> None:
        response: typing.Final = httpx.Response(
            200,
            json=ChatCompletionsNotStreamingResponse(
                choices=[
                    OneNotStreamingChoice(
                        message=OneNotStreamingChoiceMessage(
                            role=any_llm_client.MessageRole.assistant,
                            content=random_llm_response.content,
                            reasoning_content=random_llm_response.reasoning_content,
                        ),
                    ),
                ],
            ).model_dump(mode="json"),
        )

        result: typing.Final = await any_llm_client.get_client(
            OpenAIConfigFactory.build(),
            transport=httpx.MockTransport(lambda _: response),
        ).request_llm_message(**func_request)

        assert result == random_llm_response

    async def test_fails_without_alternatives(self) -> None:
        response: typing.Final = httpx.Response(
            200,
            json=ChatCompletionsNotStreamingResponse.model_construct(choices=[]).model_dump(mode="json"),
        )
        client: typing.Final = any_llm_client.get_client(
            OpenAIConfigFactory.build(),
            transport=httpx.MockTransport(lambda _: response),
        )

        with pytest.raises(LLMResponseValidationError):
            await client.request_llm_message(**LLMFuncRequestFactory.build())


class TestOpenAIRequestLLMMessageChunks:
    @pytest.mark.parametrize("func_request", LLMFuncRequestFactory.coverage())
    async def test_ok(self, faker: faker.Faker, func_request: LLMFuncRequest) -> None:
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
            any_llm_client.LLMResponse("H"),
            any_llm_client.LLMResponse("i"),
            any_llm_client.LLMResponse(" t"),
            any_llm_client.LLMResponse("here"),
            any_llm_client.LLMResponse(". How is you"),
            any_llm_client.LLMResponse("r day?"),
        ]
        config: typing.Final = OpenAIConfigFactory.build()
        response_content: typing.Final = (
            "\n\n".join(
                "data: "
                + ChatCompletionsStreamingEvent(choices=[OneStreamingChoice(delta=one_message)]).model_dump_json()
                for one_message in generated_messages
            )
            + f"\n\ndata: {ChatCompletionsStreamingEvent(choices=[]).model_dump_json()}"
            + f"\n\ndata: [DONE]\n\ndata: {faker.pystr()}\n\n"
        )
        response: typing.Final = httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            content=response_content,
        )
        client: typing.Final = any_llm_client.get_client(config, transport=httpx.MockTransport(lambda _: response))

        result: typing.Final = await consume_llm_message_chunks(client.stream_llm_message_chunks(**func_request))

        assert result == expected_result


class TestOpenAILLMErrors:
    @pytest.mark.parametrize("stream", [True, False])
    @pytest.mark.parametrize("status_code", [400, 500])
    async def test_fails_with_unknown_error(self, stream: bool, status_code: int) -> None:
        client: typing.Final = any_llm_client.get_client(
            OpenAIConfigFactory.build(),
            transport=httpx.MockTransport(lambda _: httpx.Response(status_code)),
        )

        coroutine: typing.Final = (
            consume_llm_message_chunks(client.stream_llm_message_chunks(**LLMFuncRequestFactory.build()))
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
    async def test_fails_with_out_of_tokens_error_on_status(self, stream: bool, content: bytes) -> None:
        response: typing.Final = httpx.Response(400, content=content)
        client: typing.Final = any_llm_client.get_client(
            OpenAIConfigFactory.build(),
            transport=httpx.MockTransport(lambda _: response),
        )

        coroutine: typing.Final = (
            consume_llm_message_chunks(client.stream_llm_message_chunks(**LLMFuncRequestFactory.build()))
            if stream
            else client.request_llm_message(**LLMFuncRequestFactory.build())
        )

        with pytest.raises(any_llm_client.OutOfTokensOrSymbolsError):
            await coroutine

    @pytest.mark.parametrize("stream", [True, False])
    @pytest.mark.parametrize(
        "content",
        [
            b'{"error": {"object": "error", "message": "The prompt (total length 6287) is too long to fit into the model (context length 4096). Make sure that `max_model_len` is no smaller than the number of text tokens plus multimodal tokens. For image inputs, the number of image tokens depends on the number of images, and possibly their aspect ratios as well.", "type": "BadRequestError", "param": null, "code": 400}}\n',  # noqa: E501
            b'{"object": "error", "message": "The prompt (total length 43431) is too long to fit into the model (context length 8192). Make sure that `max_model_len` is no smaller than the number of text tokens plus multimodal tokens. For image inputs, the number of image tokens depends on the number of images, and possibly their aspect ratios as well.", "type": "BadRequestError", "param": null, "code": 400}\n',  # noqa: E501
        ],
    )
    async def test_fails_with_out_of_tokens_error_on_validation(self, stream: bool, content: bytes) -> None:
        response: typing.Final = httpx.Response(
            200,
            content=f"data: {content.decode()}\n\n" if stream else content,
            headers={"Content-Type": "text/event-stream"} if stream else None,
        )
        client: typing.Final = any_llm_client.get_client(
            OpenAIConfigFactory.build(),
            transport=httpx.MockTransport(lambda _: response),
        )

        coroutine: typing.Final = (
            consume_llm_message_chunks(client.stream_llm_message_chunks(**LLMFuncRequestFactory.build()))
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
                [ChatCompletionsInputMessage(role=any_llm_client.MessageRole.user, content="Be nice")],
            ),
            (
                [any_llm_client.UserMessage("Hi there"), any_llm_client.AssistantMessage("Hi! How can I help you?")],
                [
                    ChatCompletionsInputMessage(role=any_llm_client.MessageRole.user, content="Hi there"),
                    ChatCompletionsInputMessage(
                        role=any_llm_client.MessageRole.assistant,
                        content="Hi! How can I help you?",
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
                    ChatCompletionsInputMessage(role=any_llm_client.MessageRole.user, content="Hi there"),
                    ChatCompletionsInputMessage(
                        role=any_llm_client.MessageRole.assistant,
                        content="Hi! How can I help you?",
                    ),
                ],
            ),
            (
                [any_llm_client.SystemMessage("Be nice"), any_llm_client.UserMessage("Hi there")],
                [ChatCompletionsInputMessage(role=any_llm_client.MessageRole.user, content="Be nice\n\nHi there")],
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
                    ChatCompletionsInputMessage(role=any_llm_client.MessageRole.user, content="Be nice"),
                    ChatCompletionsInputMessage(
                        role=any_llm_client.MessageRole.assistant,
                        content="Hi!\n\nI'm your answer to everything.\n\nHow can I help you?",
                    ),
                    ChatCompletionsInputMessage(
                        role=any_llm_client.MessageRole.user,
                        content="Hi there\n\nWhy is the sky blue?",
                    ),
                    ChatCompletionsInputMessage(role=any_llm_client.MessageRole.assistant, content="Well..."),
                    ChatCompletionsInputMessage(role=any_llm_client.MessageRole.user, content="Hmmm..."),
                ],
            ),
            (
                [
                    any_llm_client.SystemMessage("Be nice"),
                    any_llm_client.UserMessage(
                        [
                            any_llm_client.TextContentItem("Hi there"),
                            any_llm_client.TextContentItem("Why is the sky blue?"),
                        ],
                    ),
                ],
                [
                    ChatCompletionsInputMessage(
                        role=any_llm_client.MessageRole.user,
                        content=[
                            ChatCompletionsTextContentItem(text="Be nice"),
                            ChatCompletionsTextContentItem(text="Hi there"),
                            ChatCompletionsTextContentItem(text="Why is the sky blue?"),
                        ],
                    ),
                ],
            ),
            (
                [
                    any_llm_client.UserMessage([any_llm_client.TextContentItem("Hi")]),
                    any_llm_client.UserMessage([any_llm_client.TextContentItem("Hi there")]),
                    any_llm_client.AssistantMessage([any_llm_client.TextContentItem("Hi")]),
                ],
                [
                    ChatCompletionsInputMessage(
                        role=any_llm_client.MessageRole.user,
                        content=[
                            ChatCompletionsTextContentItem(text="Hi"),
                            ChatCompletionsTextContentItem(text="Hi there"),
                        ],
                    ),
                    ChatCompletionsInputMessage(
                        role=any_llm_client.MessageRole.assistant,
                        content=[ChatCompletionsTextContentItem(text="Hi")],
                    ),
                ],
            ),
        ],
    )
    def test_with_alternation(
        self,
        messages: list[any_llm_client.Message],
        expected_result: list[ChatCompletionsInputMessage],
    ) -> None:
        client: typing.Final = any_llm_client.OpenAIClient(
            OpenAIConfigFactory.build(force_user_assistant_message_alternation=True),
        )
        assert client._prepare_messages(messages) == expected_result  # noqa: SLF001

    def test_without_alternation(self) -> None:
        client: typing.Final = any_llm_client.OpenAIClient(
            OpenAIConfigFactory.build(force_user_assistant_message_alternation=False),
        )
        assert client._prepare_messages(  # noqa: SLF001
            [any_llm_client.SystemMessage("Be nice"), any_llm_client.UserMessage("Hi there")],
        ) == [
            ChatCompletionsInputMessage(role=any_llm_client.MessageRole.system, content="Be nice"),
            ChatCompletionsInputMessage(role=any_llm_client.MessageRole.user, content="Hi there"),
        ]
