import contextlib
import dataclasses
import types
import typing

import pydantic
import typing_extensions

from any_llm_client.core import LLMClient, LLMConfig, Message


class MockLLMConfig(LLMConfig):
    response_message: str = ""
    stream_messages: list[str] = pydantic.Field([])
    api_type: typing.Literal["mock"] = "mock"


@dataclasses.dataclass(slots=True)
class MockLLMClient(LLMClient):
    config: MockLLMConfig

    async def request_llm_message(
        self,
        messages: str | list[Message],  # noqa: ARG002
        *,
        temperature: float = 0.2,  # noqa: ARG002
        extra: dict[str, typing.Any] | None = None,  # noqa: ARG002
    ) -> str:
        return self.config.response_message

    async def _iter_config_stream_messages(self) -> typing.AsyncIterable[str]:
        for one_message in self.config.stream_messages:
            yield one_message

    @contextlib.asynccontextmanager
    async def stream_llm_partial_messages(
        self,
        messages: str | list[Message],  # noqa: ARG002
        *,
        temperature: float = 0.2,  # noqa: ARG002
        extra: dict[str, typing.Any] | None = None,  # noqa: ARG002
    ) -> typing.AsyncIterator[typing.AsyncIterable[str]]:
        yield self._iter_config_stream_messages()

    async def __aenter__(self) -> typing_extensions.Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None: ...
