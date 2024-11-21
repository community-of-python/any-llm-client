import contextlib
import dataclasses
import typing

from any_llm_client.core import LLMClient, LLMConfig, Message


class MockLLMConfig(LLMConfig):
    response_message: str
    stream_messages: list[str]
    api_type: typing.Literal["mock"] = "mock"


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class MockLLMClient(LLMClient):
    config: MockLLMConfig

    async def request_llm_response(self, *, messages: list[Message], temperature: float) -> str:  # noqa: ARG002
        return self.config.response_message

    async def _iter_config_stream_messages(self) -> typing.AsyncIterable[str]:
        for one_message in self.config.stream_messages:
            yield one_message

    @contextlib.asynccontextmanager
    async def stream_llm_partial_responses(
        self,
        *,
        messages: list[Message],  # noqa: ARG002
        temperature: float,  # noqa: ARG002
    ) -> typing.AsyncIterator[typing.AsyncIterable[str]]:
        yield self._iter_config_stream_messages()
