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
    model_config = pydantic.ConfigDict(protected_namespaces=())
    model_uri: str = pydantic.Field(alias="modelUri")
    completion_options: YandexGPTCompletionOptions = pydantic.Field(alias="completionOptions")
    messages: list[Message]


class YandexGPTAlternative(pydantic.BaseModel):
    message: Message


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
    httpx_client: HttpClient

    def __init__(
        self,
        config: YandexGPTConfig,
        *,
        request_retry: RequestRetryConfig | None = None,
        **httpx_kwargs: typing.Any,  # noqa: ANN401
    ) -> None:
        self.config = config
        self.httpx_client = HttpClient.build(request_retry=request_retry or RequestRetryConfig(), kwargs=httpx_kwargs)

    def _build_request(self, payload: dict[str, typing.Any]) -> niquests.Request:
        return niquests.Request(
            method="POST",
            url=str(self.config.url),
            json=payload,
            headers={"Authorization": self.config.auth_header, "x-data-logging-enabled": "false"},
        )

    def _prepare_payload(
        self, *, messages: str | list[Message], temperature: float = 0.2, stream: bool
    ) -> dict[str, typing.Any]:
        messages = [UserMessage(messages)] if isinstance(messages, str) else messages
        return YandexGPTRequest(
            modelUri=f"gpt://{self.config.folder_id}/{self.config.model_name}/{self.config.model_version}",
            completionOptions=YandexGPTCompletionOptions(
                stream=stream, temperature=temperature, maxTokens=self.config.max_tokens
            ),
            messages=messages,
        ).model_dump(mode="json", by_alias=True)

    async def request_llm_message(self, messages: str | list[Message], temperature: float = 0.2) -> str:
        payload: typing.Final = self._prepare_payload(messages=messages, temperature=temperature, stream=False)

        try:
            response: typing.Final = await self.httpx_client.request(self._build_request(payload))
        except HttpStatusError as exception:
            _handle_status_error(status_code=exception.status_code, content=exception.content)

        return YandexGPTResponse.model_validate_json(response.content).result.alternatives[0].message.text  # type: ignore[arg-type]

    async def _iter_completion_messages(self, response: niquests.AsyncResponse) -> typing.AsyncIterable[str]:
        async for one_line in response.iter_lines():
            validated_response = YandexGPTResponse.model_validate_json(one_line)
            yield validated_response.result.alternatives[0].message.text

    @contextlib.asynccontextmanager
    async def stream_llm_partial_messages(
        self, messages: str | list[Message], temperature: float = 0.2
    ) -> typing.AsyncIterator[typing.AsyncIterable[str]]:
        payload: typing.Final = self._prepare_payload(messages=messages, temperature=temperature, stream=True)

        try:
            async with self.httpx_client.stream(request=self._build_request(payload)) as response:
                yield self._iter_completion_messages(response)
        except HttpStatusError as exception:
            _handle_status_error(status_code=exception.status_code, content=exception.content)

    async def __aenter__(self) -> typing_extensions.Self:
        await self.httpx_client.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None:
        await self.httpx_client.__aexit__(exc_type, exc_value, traceback)
