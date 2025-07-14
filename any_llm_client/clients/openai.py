import contextlib
import dataclasses
import os
import types
import typing
from http import HTTPStatus

import annotated_types
import httpx
import httpx_sse
import pydantic
import typing_extensions

from any_llm_client.core import (
    LLMClient,
    LLMConfig,
    LLMConfigValue,
    LLMError,
    LLMResponse,
    LLMResponseValidationError,
    Message,
    MessageRole,
    OutOfTokensOrSymbolsError,
    TextContentItem,
    UserMessage,
)
from any_llm_client.http import get_http_client_from_kwargs, make_http_request, make_streaming_http_request
from any_llm_client.retry import RequestRetryConfig


OPENAI_AUTH_TOKEN_ENV_NAME: typing.Final = "ANY_LLM_CLIENT_OPENAI_AUTH_TOKEN"  # noqa: S105


class OpenAIConfig(LLMConfig):
    if typing.TYPE_CHECKING:
        url: str
    else:
        url: pydantic.HttpUrl
    auth_token: str | None = pydantic.Field(default_factory=lambda: os.environ.get(OPENAI_AUTH_TOKEN_ENV_NAME))
    model_name: str
    request_extra: dict[str, typing.Any] = pydantic.Field(default_factory=dict)
    force_user_assistant_message_alternation: bool = False
    "Gemma 2 doesn't support {role: system, text: ...} message, and requires alternated messages"
    api_type: typing.Literal["openai"] = "openai"


class ChatCompletionsTextContentItem(pydantic.BaseModel):
    type: typing.Literal["text"] = "text"
    text: str


class ChatCompletionsContentUrl(pydantic.BaseModel):
    url: str


class ChatCompletionsImageContentItem(pydantic.BaseModel):
    type: typing.Literal["image_url"] = "image_url"
    image_url: ChatCompletionsContentUrl


ChatCompletionsAnyContentItem = ChatCompletionsImageContentItem | ChatCompletionsTextContentItem
ChatCompletionsContentItemList = typing.Annotated[list[ChatCompletionsAnyContentItem], annotated_types.MinLen(1)]


class ChatCompletionsInputMessage(pydantic.BaseModel):
    role: MessageRole
    content: str | ChatCompletionsContentItemList


class ChatCompletionsRequest(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    stream: bool
    model: str
    messages: list[ChatCompletionsInputMessage]
    temperature: float


class OneStreamingChoiceDelta(pydantic.BaseModel):
    role: typing.Literal[MessageRole.assistant] | None = None
    content: str | None = None
    reasoning_content: str | None = None


class OneStreamingChoice(pydantic.BaseModel):
    delta: OneStreamingChoiceDelta


class ChatCompletionsStreamingEvent(pydantic.BaseModel):
    choices: list[OneStreamingChoice]


class OneNotStreamingChoiceMessage(pydantic.BaseModel):
    role: MessageRole
    content: str
    reasoning_content: str | None = None


class OneNotStreamingChoice(pydantic.BaseModel):
    message: OneNotStreamingChoiceMessage


class ChatCompletionsNotStreamingResponse(pydantic.BaseModel):
    choices: typing.Annotated[list[OneNotStreamingChoice], annotated_types.MinLen(1)]


def _prepare_one_message(one_message: Message) -> ChatCompletionsInputMessage:
    if isinstance(one_message.content, str):
        return ChatCompletionsInputMessage(role=one_message.role, content=one_message.content)
    content_items: typing.Final = [
        ChatCompletionsTextContentItem(text=one_content_item.text)
        if isinstance(one_content_item, TextContentItem)
        else ChatCompletionsImageContentItem(image_url=ChatCompletionsContentUrl(url=one_content_item.image_url))
        for one_content_item in one_message.content
    ]
    return ChatCompletionsInputMessage(role=one_message.role, content=content_items)


def _merge_content_chunks(
    content_chunks: list[str | ChatCompletionsContentItemList],
) -> str | ChatCompletionsContentItemList:
    if all(isinstance(one_content_chunk, str) for one_content_chunk in content_chunks):
        return "\n\n".join(typing.cast("list[str]", content_chunks))

    new_content_items: ChatCompletionsContentItemList = []
    for one_content_chunk in content_chunks:
        if isinstance(one_content_chunk, str):
            new_content_items.append(ChatCompletionsTextContentItem(text=one_content_chunk))
        else:
            new_content_items += one_content_chunk
    return new_content_items


def _make_user_assistant_alternate_messages(
    messages: typing.Iterable[ChatCompletionsInputMessage],
) -> typing.Iterable[ChatCompletionsInputMessage]:
    current_message_role = MessageRole.user
    current_message_content_chunks = []

    for one_message in messages:
        if isinstance(one_message.content, str) and not one_message.content.strip():
            continue

        if (
            one_message.role in {MessageRole.system, MessageRole.user} and current_message_role == MessageRole.user
        ) or one_message.role == current_message_role == MessageRole.assistant:
            current_message_content_chunks.append(one_message.content)
        else:
            if current_message_content_chunks:
                yield ChatCompletionsInputMessage(
                    role=current_message_role,
                    content=_merge_content_chunks(current_message_content_chunks),
                )
            current_message_content_chunks = [one_message.content]
            current_message_role = one_message.role

    if current_message_content_chunks:
        yield ChatCompletionsInputMessage(
            role=current_message_role,
            content=_merge_content_chunks(current_message_content_chunks),
        )


def _handle_status_error(*, status_code: int, content: bytes) -> typing.NoReturn:
    if status_code == HTTPStatus.BAD_REQUEST and b"Please reduce the length of the messages" in content:  # vLLM
        raise OutOfTokensOrSymbolsError(response_content=content)
    raise LLMError(response_content=content)


def _handle_validation_error(*, content: bytes, original_error: pydantic.ValidationError) -> typing.NoReturn:
    if b"is too long to fit into the model" in content:  # vLLM
        raise OutOfTokensOrSymbolsError(response_content=content)
    raise LLMResponseValidationError(response_content=content, original_error=original_error)


@dataclasses.dataclass(slots=True, init=False)
class OpenAIClient(LLMClient):
    config: OpenAIConfig
    httpx_client: httpx.AsyncClient
    request_retry: RequestRetryConfig

    def __init__(
        self,
        config: OpenAIConfig,
        *,
        request_retry: RequestRetryConfig | None = None,
        **httpx_kwargs: typing.Any,  # noqa: ANN401
    ) -> None:
        self.config = config
        self.request_retry = request_retry or RequestRetryConfig()
        self.httpx_client = get_http_client_from_kwargs(httpx_kwargs)

    def _build_request(self, payload: dict[str, typing.Any]) -> httpx.Request:
        return self.httpx_client.build_request(
            method="POST",
            url=str(self.config.url),
            json=payload,
            headers={"Authorization": f"Bearer {self.config.auth_token}"} if self.config.auth_token else None,
        )

    def _prepare_messages(self, messages: str | list[Message]) -> list[ChatCompletionsInputMessage]:
        messages = [UserMessage(messages)] if isinstance(messages, str) else messages
        initial_messages: typing.Final = (_prepare_one_message(one_message) for one_message in messages)
        return (
            list(_make_user_assistant_alternate_messages(initial_messages))
            if self.config.force_user_assistant_message_alternation
            else list(initial_messages)
        )

    def _prepare_payload(
        self,
        *,
        messages: str | list[Message],
        temperature: float,
        stream: bool,
        extra: dict[str, typing.Any] | None,
    ) -> dict[str, typing.Any]:
        return ChatCompletionsRequest(
            stream=stream,
            model=self.config.model_name,
            messages=self._prepare_messages(messages),
            temperature=self.config._resolve_request_temperature(temperature),  # noqa: SLF001
            **self.config.request_extra | (extra or {}),
        ).model_dump(mode="json")

    async def request_llm_message(
        self,
        messages: str | list[Message],
        *,
        temperature: float = LLMConfigValue(attr="temperature"),
        extra: dict[str, typing.Any] | None = None,
    ) -> LLMResponse:
        payload: typing.Final = self._prepare_payload(
            messages=messages,
            temperature=temperature,
            stream=False,
            extra=extra,
        )
        try:
            response: typing.Final = await make_http_request(
                httpx_client=self.httpx_client,
                request_retry=self.request_retry,
                build_request=lambda: self._build_request(payload),
            )
        except httpx.HTTPStatusError as exception:
            _handle_status_error(status_code=exception.response.status_code, content=exception.response.content)

        try:
            validated_message_model: typing.Final = (
                ChatCompletionsNotStreamingResponse.model_validate_json(response.content).choices[0].message
            )
        except pydantic.ValidationError as validation_error:
            _handle_validation_error(content=response.content, original_error=validation_error)
        finally:
            await response.aclose()

        return LLMResponse(
            content=validated_message_model.content,
            reasoning_content=validated_message_model.reasoning_content,
        )

    async def _iter_response_chunks(self, response: httpx.Response) -> typing.AsyncIterable[LLMResponse]:
        async for event in httpx_sse.EventSource(response).aiter_sse():
            if event.data == "[DONE]":
                break

            try:
                validated_response = ChatCompletionsStreamingEvent.model_validate_json(event.data)
            except pydantic.ValidationError as validation_error:
                _handle_validation_error(content=event.data.encode(), original_error=validation_error)

            if not (
                (validated_choices := validated_response.choices)
                and (validated_delta := validated_choices[0].delta)
                and (validated_delta.content or validated_delta.reasoning_content)
            ):
                continue

            yield LLMResponse(content=validated_delta.content, reasoning_content=validated_delta.reasoning_content)

    @contextlib.asynccontextmanager
    async def stream_llm_message_chunks(
        self,
        messages: str | list[Message],
        *,
        temperature: float = LLMConfigValue(attr="temperature"),
        extra: dict[str, typing.Any] | None = None,
    ) -> typing.AsyncIterator[typing.AsyncIterable[LLMResponse]]:
        payload: typing.Final = self._prepare_payload(
            messages=messages,
            temperature=temperature,
            stream=True,
            extra=extra,
        )
        try:
            async with make_streaming_http_request(
                httpx_client=self.httpx_client,
                request_retry=self.request_retry,
                build_request=lambda: self._build_request(payload),
            ) as response:
                yield self._iter_response_chunks(response)
        except httpx.HTTPStatusError as exception:
            content: typing.Final = await exception.response.aread()
            await exception.response.aclose()
            _handle_status_error(status_code=exception.response.status_code, content=content)

    async def __aenter__(self) -> typing_extensions.Self:
        await self.httpx_client.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None:
        await self.httpx_client.__aexit__(exc_type=exc_type, exc_value=exc_value, traceback=traceback)
