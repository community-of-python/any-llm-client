import contextlib
import dataclasses
import os
import types
import typing
from http import HTTPStatus

import annotated_types
import niquests
import pydantic
import typing_extensions

from any_llm_client.core import LLMClient, LLMConfig, LLMError, Message, OutOfTokensOrSymbolsError, UserMessage
from any_llm_client.http import HttpClient, HttpStatusError
from any_llm_client.retry import RequestRetryConfig


YANDEXGPT_AUTH_HEADER_ENV_NAME: typing.Final = "ANY_LLM_CLIENT_YANDEXGPT_AUTH_HEADER"
YANDEXGPT_FOLDER_ID_ENV_NAME: typing.Final = "ANY_LLM_CLIENT_YANDEXGPT_FOLDER_ID"


class YandexGPTConfig(LLMConfig):
    if typing.TYPE_CHECKING:
        url: str = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
    else:
        url: pydantic.HttpUrl = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
    auth_header: str = pydantic.Field(  # type: ignore[assignment]
        default_factory=lambda: os.environ.get(YANDEXGPT_AUTH_HEADER_ENV_NAME), validate_default=True
    )
    folder_id: str = pydantic.Field(  # type: ignore[assignment]
        default_factory=lambda: os.environ.get(YANDEXGPT_FOLDER_ID_ENV_NAME), validate_default=True
    )
    model_name: str
    model_version: str = "latest"
    max_tokens: int = 7400
    api_type: typing.Literal["yandexgpt"] = "yandexgpt"


class YandexGPTCompletionOptions(pydantic.BaseModel):
    stream: bool
    temperature: float = 0.2
    max_tokens: int = pydantic.Field(gt=0, alias="maxTokens")


class YandexGPTRequest(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(protected_namespaces=(), extra="allow")
    model_uri: str = pydantic.Field(alias="modelUri")
    completion_options: YandexGPTCompletionOptions = pydantic.Field(alias="completionOptions")
    messages: list[Message]


class YandexGPTAlternative(pydantic.BaseModel):
    message: Message


class YandexGPTResult(pydantic.BaseModel):
    alternatives: typing.Annotated[list[YandexGPTAlternative], annotated_types.MinLen(1)]


class YandexGPTResponse(pydantic.BaseModel):
    result: YandexGPTResult


def _handle_status_error(error: HttpStatusError) -> typing.NoReturn:
    if error.status_code == HTTPStatus.BAD_REQUEST and (
        b"number of input tokens must be no more than" in error.content
        or (b"text length is" in error.content and b"which is outside the range" in error.content)
    ):
        raise OutOfTokensOrSymbolsError(response_content=error.content)
    raise LLMError(response_content=error.content)


@dataclasses.dataclass(slots=True, init=False)
class YandexGPTClient(LLMClient):
    config: YandexGPTConfig
    http_client: HttpClient

    def __init__(
        self,
        config: YandexGPTConfig,
        *,
        request_retry: RequestRetryConfig | None = None,
        **niquests_kwargs: typing.Any,  # noqa: ANN401
    ) -> None:
        self.config = config
        self.http_client = HttpClient(
            request_retry=request_retry or RequestRetryConfig(), niquests_kwargs=niquests_kwargs
        )

    def _build_request(self, payload: dict[str, typing.Any]) -> niquests.Request:
        return niquests.Request(
            method="POST",
            url=str(self.config.url),
            json=payload,
            headers={"Authorization": self.config.auth_header, "x-data-logging-enabled": "false"},
        )

    def _prepare_payload(
        self,
        *,
        messages: str | list[Message],
        temperature: float = 0.2,
        stream: bool,
        extra: dict[str, typing.Any] | None,
    ) -> dict[str, typing.Any]:
        messages = [UserMessage(messages)] if isinstance(messages, str) else messages
        return YandexGPTRequest(
            modelUri=f"gpt://{self.config.folder_id}/{self.config.model_name}/{self.config.model_version}",
            completionOptions=YandexGPTCompletionOptions(
                stream=stream, temperature=temperature, maxTokens=self.config.max_tokens
            ),
            messages=messages,
            **extra or {},
        ).model_dump(mode="json", by_alias=True)

    async def request_llm_message(
        self, messages: str | list[Message], *, temperature: float = 0.2, extra: dict[str, typing.Any] | None = None
    ) -> str:
        payload: typing.Final = self._prepare_payload(
            messages=messages, temperature=temperature, stream=False, extra=extra
        )

        try:
            response: typing.Final = await self.http_client.request(self._build_request(payload))
        except HttpStatusError as exception:
            _handle_status_error(exception)

        return YandexGPTResponse.model_validate_json(response).result.alternatives[0].message.text

    async def _iter_completion_messages(self, response: typing.AsyncIterable[bytes]) -> typing.AsyncIterable[str]:
        async for one_line in response:
            validated_response = YandexGPTResponse.model_validate_json(one_line)
            yield validated_response.result.alternatives[0].message.text

    @contextlib.asynccontextmanager
    async def stream_llm_partial_messages(
        self, messages: str | list[Message], *, temperature: float = 0.2, extra: dict[str, typing.Any] | None = None
    ) -> typing.AsyncIterator[typing.AsyncIterable[str]]:
        payload: typing.Final = self._prepare_payload(
            messages=messages, temperature=temperature, stream=True, extra=extra
        )

        try:
            async with self.http_client.stream(request=self._build_request(payload)) as response:
                yield self._iter_completion_messages(response)
        except HttpStatusError as exception:
            _handle_status_error(exception)

    async def __aenter__(self) -> typing_extensions.Self:
        await self.http_client.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None:
        await self.http_client.__aexit__(exc_type=exc_type, exc_value=exc_value, traceback=traceback)
