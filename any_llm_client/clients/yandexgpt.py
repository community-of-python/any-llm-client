import contextlib
import dataclasses
import os
import types
import typing
from http import HTTPStatus

import annotated_types
import httpx
import pydantic
import typing_extensions

from any_llm_client.core import (
    ImageContentItem,
    LLMClient,
    LLMConfig,
    LLMConfigValue,
    LLMError,
    LLMRequestValidationError,
    LLMResponse,
    LLMResponseValidationError,
    Message,
    MessageRole,
    OutOfTokensOrSymbolsError,
)
from any_llm_client.http import get_http_client_from_kwargs, make_http_request, make_streaming_http_request
from any_llm_client.retry import RequestRetryConfig


YANDEXGPT_AUTH_HEADER_ENV_NAME: typing.Final = "ANY_LLM_CLIENT_YANDEXGPT_AUTH_HEADER"
YANDEXGPT_FOLDER_ID_ENV_NAME: typing.Final = "ANY_LLM_CLIENT_YANDEXGPT_FOLDER_ID"


class YandexGPTConfig(LLMConfig):
    if typing.TYPE_CHECKING:
        url: str = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
    else:
        url: pydantic.HttpUrl = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
    auth_header: str = pydantic.Field(  # type: ignore[assignment]
        default_factory=lambda: os.environ.get(YANDEXGPT_AUTH_HEADER_ENV_NAME),
        validate_default=True,
    )
    folder_id: str = pydantic.Field(  # type: ignore[assignment]
        default_factory=lambda: os.environ.get(YANDEXGPT_FOLDER_ID_ENV_NAME),
        validate_default=True,
    )
    model_name: str
    model_version: str = "latest"
    max_tokens: int = 7400
    api_type: typing.Literal["yandexgpt"] = "yandexgpt"


class YandexGPTCompletionOptions(pydantic.BaseModel):
    stream: bool
    temperature: float
    max_tokens: int = pydantic.Field(gt=0, alias="maxTokens")


class YandexGPTMessage(pydantic.BaseModel):
    role: MessageRole
    text: str


class YandexGPTRequest(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(protected_namespaces=(), extra="allow")
    model_uri: str = pydantic.Field(alias="modelUri")
    completion_options: YandexGPTCompletionOptions = pydantic.Field(alias="completionOptions")
    messages: list[YandexGPTMessage]


class YandexGPTAlternative(pydantic.BaseModel):
    message: YandexGPTMessage


class YandexGPTResult(pydantic.BaseModel):
    alternatives: typing.Annotated[list[YandexGPTAlternative], annotated_types.MinLen(1)]


class YandexGPTResponse(pydantic.BaseModel):
    result: YandexGPTResult


def _handle_status_error(*, status_code: int, content: bytes) -> typing.NoReturn:
    if status_code == HTTPStatus.BAD_REQUEST and (
        b"number of input tokens must be no more than" in content
        or (b"text length is" in content and b"which is outside the range" in content)
    ):
        raise OutOfTokensOrSymbolsError(response_content=content)
    raise LLMError(response_content=content)


@dataclasses.dataclass(slots=True, init=False)
class YandexGPTClient(LLMClient):
    config: YandexGPTConfig
    httpx_client: httpx.AsyncClient
    request_retry: RequestRetryConfig

    def __init__(
        self,
        config: YandexGPTConfig,
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
            headers={"Authorization": self.config.auth_header, "x-data-logging-enabled": "false"},
        )

    def _prepare_payload(
        self,
        *,
        messages: str | list[Message],
        temperature: float,
        stream: bool,
        extra: dict[str, typing.Any] | None,
    ) -> dict[str, typing.Any]:
        if isinstance(messages, str):
            prepared_messages = [YandexGPTMessage(role=MessageRole.user, text=messages)]
        else:
            prepared_messages = []
            for one_message in messages:
                if isinstance(one_message.content, list):
                    if len(one_message.content) != 1:
                        raise LLMRequestValidationError(
                            "YandexGPTClient does not support multiple content items per message",
                        )
                    message_content = one_message.content[0]
                    if isinstance(message_content, ImageContentItem):
                        raise LLMRequestValidationError("YandexGPTClient does not support image content items")
                    message_text = message_content.text
                else:
                    message_text = one_message.content
                prepared_messages.append(YandexGPTMessage(role=one_message.role, text=message_text))

        return YandexGPTRequest(
            modelUri=f"gpt://{self.config.folder_id}/{self.config.model_name}/{self.config.model_version}",
            completionOptions=YandexGPTCompletionOptions(
                stream=stream,
                temperature=self.config._resolve_request_temperature(temperature),  # noqa: SLF001
                maxTokens=self.config.max_tokens,
            ),
            messages=prepared_messages,
            **self.config.request_extra | (extra or {}),
        ).model_dump(mode="json", by_alias=True)

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
            validated_response: typing.Final = YandexGPTResponse.model_validate_json(response.content)
        except pydantic.ValidationError as validation_error:
            raise LLMResponseValidationError(
                response_content=response.content, original_error=validation_error
            ) from validation_error

        return LLMResponse(content=validated_response.result.alternatives[0].message.text)

    async def _iter_response_chunks(self, response: httpx.Response) -> typing.AsyncIterable[LLMResponse]:
        previous_cursor = 0
        async for one_line in response.aiter_lines():
            try:
                validated_response = YandexGPTResponse.model_validate_json(one_line)
            except pydantic.ValidationError as validation_error:
                raise LLMResponseValidationError(
                    response_content=one_line.encode(), original_error=validation_error
                ) from validation_error

            response_text = validated_response.result.alternatives[0].message.text
            yield LLMResponse(content=response_text[previous_cursor:])
            previous_cursor = len(response_text)

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
