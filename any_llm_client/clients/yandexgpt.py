import contextlib
import dataclasses
import types
import typing
from http import HTTPStatus

import annotated_types
import httpx
import pydantic
import typing_extensions

from any_llm_client.core import LLMClient, LLMConfig, LLMError, Message, OutOfTokensOrSymbolsError
from any_llm_client.http import make_http_request, make_streaming_http_request
from any_llm_client.retry import RequestRetryConfig


class YandexGPTConfig(LLMConfig):
    if typing.TYPE_CHECKING:
        url: str = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"  # pragma: no cover
    else:
        url: pydantic.HttpUrl = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
    auth_header: str | None = None
    folder_id: str | None = None
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
    httpx_client: httpx.AsyncClient
    request_retry: RequestRetryConfig

    def __init__(
        self,
        config: YandexGPTConfig,
        httpx_client: httpx.AsyncClient | None = None,
        request_retry: RequestRetryConfig | None = None,
    ) -> None:
        self.config = config
        self.httpx_client = httpx_client or httpx.AsyncClient()
        self.request_retry = request_retry or RequestRetryConfig()

    def _build_request(self, payload: dict[str, typing.Any]) -> httpx.Request:
        headers: typing.Final = {"x-data-logging-enabled": "false"}
        if self.config.auth_header:
            headers["Authorization"] = self.config.auth_header
        return self.httpx_client.build_request(method="POST", url=str(self.config.url), json=payload, headers=headers)

    def _prepare_payload(
        self, *, messages: str | list[Message], temperature: float = 0.2, stream: bool
    ) -> dict[str, typing.Any]:
        messages = [Message(role="user", text=messages)] if isinstance(messages, str) else messages
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
            response: typing.Final = await make_http_request(
                httpx_client=self.httpx_client,
                request_retry=self.request_retry,
                build_request=lambda: self._build_request(payload),
            )
        except httpx.HTTPStatusError as exception:
            _handle_status_error(status_code=exception.response.status_code, content=exception.response.content)

        return YandexGPTResponse.model_validate_json(response.content).result.alternatives[0].message.text

    async def _iter_completion_messages(self, response: httpx.Response) -> typing.AsyncIterable[str]:
        async for one_line in response.aiter_lines():
            validated_response = YandexGPTResponse.model_validate_json(one_line)
            yield validated_response.result.alternatives[0].message.text

    @contextlib.asynccontextmanager
    async def stream_llm_partial_messages(
        self, messages: str | list[Message], temperature: float = 0.2
    ) -> typing.AsyncIterator[typing.AsyncIterable[str]]:
        payload: typing.Final = self._prepare_payload(messages=messages, temperature=temperature, stream=True)

        try:
            async with make_streaming_http_request(
                httpx_client=self.httpx_client,
                request_retry=self.request_retry,
                build_request=lambda: self._build_request(payload),
            ) as response:
                yield self._iter_completion_messages(response)
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
