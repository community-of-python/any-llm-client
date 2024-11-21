import contextlib
import dataclasses
import typing
from http import HTTPStatus

import annotated_types
import httpx
import pydantic

from any_llm_client.core import LLMClient, LLMConfig, LLMError, Message, OutOfTokensOrSymbolsError
from any_llm_client.http import make_http_request, make_streaming_http_request


class YandexGPTConfig(LLMConfig):
    url: pydantic.HttpUrl = pydantic.HttpUrl("https://llm.api.cloud.yandex.net/foundationModels/v1/completion")
    auth_header: str
    folder_id: str
    model_name: str
    model_version: str = "latest"
    max_tokens: int = 7400
    api_type: typing.Literal["yandexgpt"] = "yandexgpt"


class YandexGPTCompletionOptions(pydantic.BaseModel):
    stream: bool
    temperature: float
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


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class YandexGPTClient(LLMClient):
    config: YandexGPTConfig
    httpx_client: httpx.AsyncClient

    def _build_request(self, payload: dict[str, typing.Any]) -> httpx.Request:
        return self.httpx_client.build_request(
            method="POST",
            url=str(self.config.url),
            json=payload,
            headers={"Authorization": self.config.auth_header, "x-data-logging-enabled": "false"},
        )

    def _prepare_payload(self, *, messages: list[Message], temperature: float, stream: bool) -> dict[str, typing.Any]:
        return YandexGPTRequest(
            modelUri=f"gpt://{self.config.folder_id}/{self.config.model_name}/{self.config.model_version}",
            completionOptions=YandexGPTCompletionOptions(
                stream=stream, temperature=temperature, maxTokens=self.config.max_tokens
            ),
            messages=messages,
        ).model_dump(mode="json", by_alias=True)

    async def request_llm_response(self, *, messages: list[Message], temperature: float) -> str:
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
    async def stream_llm_partial_responses(
        self, *, messages: list[Message], temperature: float
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
