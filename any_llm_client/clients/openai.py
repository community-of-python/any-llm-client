import contextlib
import dataclasses
import typing
from http import HTTPStatus

import annotated_types
import httpx
import httpx_sse
import pydantic

from any_llm_client.core import LLMClient, LLMConfig, LLMError, Message, MessageRole, OutOfTokensOrSymbolsError
from any_llm_client.http import make_http_request, make_streaming_http_request


class OpenAIConfig(LLMConfig):
    url: pydantic.HttpUrl
    auth_token: str | None = None
    model_name: str
    force_user_assistant_message_alternation: bool = False
    "Gemma 2 doesn't support {role: system, text: ...} message, and requires alternated messages"
    api_type: typing.Literal["openai"] = "openai"


class ChatCompletionsMessage(pydantic.BaseModel):
    role: MessageRole
    content: str


class ChatCompletionsRequest(pydantic.BaseModel):
    stream: bool
    model: str
    messages: list[ChatCompletionsMessage]
    temperature: float


class OneStreamingChoiceDelta(pydantic.BaseModel):
    role: typing.Literal["assistant"] | None = None
    content: str | None = None


class OneStreamingChoice(pydantic.BaseModel):
    delta: OneStreamingChoiceDelta


class ChatCompletionsStreamingEvent(pydantic.BaseModel):
    choices: typing.Annotated[list[OneStreamingChoice], annotated_types.MinLen(1)]


class OneNotStreamingChoice(pydantic.BaseModel):
    message: ChatCompletionsMessage


class ChatCompletionsNotStreamingResponse(pydantic.BaseModel):
    choices: typing.Annotated[list[OneNotStreamingChoice], annotated_types.MinLen(1)]


def _make_user_assistant_alternate_messages(
    messages: typing.Iterable[ChatCompletionsMessage],
) -> typing.Iterable[ChatCompletionsMessage]:
    current_message_role: MessageRole = "user"
    current_message_content_chunks = []

    for one_message in messages:
        if not one_message.content.strip():
            continue

        if (
            one_message.role in {"system", "user"} and current_message_role == "user"
        ) or one_message.role == current_message_role == "assistant":
            current_message_content_chunks.append(one_message.content)
        else:
            if current_message_content_chunks:
                yield ChatCompletionsMessage(
                    role=current_message_role, content="\n\n".join(current_message_content_chunks)
                )
            current_message_content_chunks = [one_message.content]
            current_message_role = one_message.role

    if current_message_content_chunks:
        yield ChatCompletionsMessage(role=current_message_role, content="\n\n".join(current_message_content_chunks))


def _handle_status_error(*, status_code: int, content: bytes) -> typing.NoReturn:
    if status_code == HTTPStatus.BAD_REQUEST and b"Please reduce the length of the messages" in content:  # vLLM
        raise OutOfTokensOrSymbolsError(response_content=content)
    raise LLMError(response_content=content)


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class OpenAIClient(LLMClient):
    config: OpenAIConfig
    httpx_client: httpx.AsyncClient

    def _build_request(self, payload: dict[str, typing.Any]) -> httpx.Request:
        return self.httpx_client.build_request(
            method="POST",
            url=str(self.config.url),
            json=payload,
            headers={"Authorization": f"Bearer {self.config.auth_token}"} if self.config.auth_token else None,
        )

    def _prepare_messages(self, messages: list[Message]) -> list[ChatCompletionsMessage]:
        initial_messages: typing.Final = (
            ChatCompletionsMessage(role=one_message.role, content=one_message.text) for one_message in messages
        )
        return (
            list(_make_user_assistant_alternate_messages(initial_messages))
            if self.config.force_user_assistant_message_alternation
            else list(initial_messages)
        )

    async def request_llm_response(self, *, messages: list[Message], temperature: float) -> str:
        payload: typing.Final = ChatCompletionsRequest(
            stream=False,
            model=self.config.model_name,
            messages=self._prepare_messages(messages),
            temperature=temperature,
        ).model_dump(mode="json")
        try:
            response: typing.Final = await make_http_request(
                httpx_client=self.httpx_client,
                request_retry=self.request_retry,
                build_request=lambda: self._build_request(payload),
            )
        except httpx.HTTPStatusError as exception:
            _handle_status_error(status_code=exception.response.status_code, content=exception.response.content)
        try:
            return ChatCompletionsNotStreamingResponse.model_validate_json(response.content).choices[0].message.content
        finally:
            await response.aclose()

    async def _iter_partial_responses(self, response: httpx.Response) -> typing.AsyncIterable[str]:
        text_chunks: typing.Final = []
        async for event in httpx_sse.EventSource(response).aiter_sse():
            if event.data == "[DONE]":
                break
            validated_response = ChatCompletionsStreamingEvent.model_validate_json(event.data)
            if not (one_chunk := validated_response.choices[0].delta.content):
                continue
            text_chunks.append(one_chunk)
            yield "".join(text_chunks)

    @contextlib.asynccontextmanager
    async def stream_llm_partial_responses(
        self, *, messages: list[Message], temperature: float
    ) -> typing.AsyncIterator[typing.AsyncIterable[str]]:
        payload: typing.Final = ChatCompletionsRequest(
            stream=True,
            model=self.config.model_name,
            messages=self._prepare_messages(messages),
            temperature=temperature,
        ).model_dump(mode="json")
        try:
            async with make_streaming_http_request(
                httpx_client=self.httpx_client,
                request_retry=self.request_retry,
                build_request=lambda: self._build_request(payload),
            ) as response:
                yield self._iter_partial_responses(response)
        except httpx.HTTPStatusError as exception:
            content: typing.Final = await exception.response.aread()
            await exception.response.aclose()
            _handle_status_error(status_code=exception.response.status_code, content=content)
