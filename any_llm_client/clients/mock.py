import contextlib
import dataclasses
import types
import typing

import pydantic
import typing_extensions

from any_llm_client.core import LLMClient, LLMConfig, LLMConfigValue, LLMResponse, Message


class MockLLMConfig(LLMConfig):
    response_message: LLMResponse = LLMResponse(content="")
    stream_messages: list[LLMResponse] = pydantic.Field([])
    api_type: typing.Literal["mock"] = "mock"


@dataclasses.dataclass(slots=True)
class MockLLMClient(LLMClient):
    config: MockLLMConfig

    async def request_llm_message(
        self,
        messages: str | list[Message],  # noqa: ARG002
        *,
        temperature: float = LLMConfigValue(attr="temperature"),  # noqa: ARG002
        extra: dict[str, typing.Any] | None = None,  # noqa: ARG002
    ) -> LLMResponse:
        return self.config.response_message

    async def _iter_config_stream_messages(self) -> typing.AsyncIterable[LLMResponse]:
        for one_message in self.config.stream_messages:
            yield one_message

    @contextlib.asynccontextmanager
    async def stream_llm_message_chunks(
        self,
        messages: str | list[Message],  # noqa: ARG002
        *,
        temperature: float = LLMConfigValue(attr="temperature"),  # noqa: ARG002
        extra: dict[str, typing.Any] | None = None,  # noqa: ARG002
    ) -> typing.AsyncIterator[typing.AsyncIterable[LLMResponse]]:
        yield self._iter_config_stream_messages()

    async def __aenter__(self) -> typing_extensions.Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None: ...
