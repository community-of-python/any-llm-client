import contextlib
import dataclasses
import typing
from http import HTTPStatus

import annotated_types
import httpx
import httpx_sse
import pydantic
import stamina

from any_llm_client.abc import LLMClient, LLMConfig, LLMError, Message, MessageRole, OutOfTokensOrSymbolsError


class OpenAIConfig(LLMConfig):
    url: pydantic.HttpUrl
    auth_token: str | None = None
    model_name: str
    # Gemma doesn't support {role: system, text: ...} message, and requires alternated messages
    force_user_assistant_message_alternation: bool = False
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
    if status_code == HTTPStatus.BAD_REQUEST and b"Please reduce the length of the messages" in content:
        raise OutOfTokensOrSymbolsError(response_content=content)
    raise LLMError(response_content=content)


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class OpenAIClient(LLMClient):
    config: OpenAIConfig
    httpx_client: httpx.AsyncClient

    @stamina.retry(on=httpx.HTTPError, attempts=3)
    async def _make_request(self, *, payload: dict[str, typing.Any], stream: bool) -> httpx.Response:
        request: typing.Final = self.httpx_client.build_request(
            method="POST",
            url=str(self.config.url),
            json=payload,
            headers={"Authorization": f"Bearer {self.config.auth_token}"} if self.config.auth_token else None,
        )
        response: typing.Final = await self.httpx_client.send(request, stream=stream)
        response.raise_for_status()
        return response

    def _prepare_messages(self, messages: list[Message]) -> list[ChatCompletionsMessage]:
        initial_messages = (
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
            response: typing.Final = await self._make_request(payload=payload, stream=False)
        except httpx.HTTPStatusError as exception:
            content: typing.Final = await exception.response.aread()
            await exception.response.aclose()
            _handle_status_error(status_code=exception.response.status_code, content=content)
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
    async def request_llm_partial_responses(
        self, *, messages: list[Message], temperature: float
    ) -> typing.AsyncGenerator[typing.AsyncIterable[str], None]:
        payload: typing.Final = ChatCompletionsRequest(
            stream=True,
            model=self.config.model_name,
            messages=self._prepare_messages(messages),
            temperature=temperature,
        ).model_dump(mode="json")
        try:
            response: typing.Final = await self._make_request(payload=payload, stream=True)
        except httpx.HTTPStatusError as exception:
            content: typing.Final = await exception.response.aread()
            await exception.response.aclose()
            _handle_status_error(status_code=exception.response.status_code, content=content)
        try:
            yield self._iter_partial_responses(response)
        finally:
            await response.aclose()
