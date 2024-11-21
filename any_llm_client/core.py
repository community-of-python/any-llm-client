import contextlib
import dataclasses
import typing

import pydantic

from any_llm_client.retry import RequestRetryConfig


MessageRole = typing.Literal["system", "user", "assistant"]


class Message(pydantic.BaseModel):
    role: MessageRole
    text: str


@dataclasses.dataclass
class LLMError(Exception):
    response_content: bytes

    def __str__(self) -> str:
        return self.__repr__().removeprefix(self.__class__.__name__)


@dataclasses.dataclass
class OutOfTokensOrSymbolsError(LLMError): ...


class LLMConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(protected_namespaces=())
    api_type: str


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class LLMClient(typing.Protocol):
    request_retry: RequestRetryConfig = dataclasses.field(default_factory=RequestRetryConfig)

    async def request_llm_response(self, *, messages: list[Message], temperature: float) -> str: ...  # raises LLMError

    @contextlib.asynccontextmanager
    def stream_llm_partial_responses(
        self, *, messages: list[Message], temperature: float
    ) -> typing.AsyncIterator[typing.AsyncIterable[str]]: ...  # raises LLMError
