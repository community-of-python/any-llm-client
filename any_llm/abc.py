import contextlib
import dataclasses
import typing

import pydantic


MessageRole = typing.Literal["system", "user", "assistant"]


class Message(pydantic.BaseModel):
    role: MessageRole
    text: str


@dataclasses.dataclass
class LLMError(Exception):
    response_content: bytes


@dataclasses.dataclass
class OutOfTokensOrSymbolsError(LLMError): ...


class LLMConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(protected_namespaces=())
    api_type: str


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class LLMClient(typing.Protocol):
    async def request_llm_response(self, *, messages: list[Message], temperature: float) -> str: ...  # raises LLMError

    @contextlib.asynccontextmanager
    def request_llm_partial_responses(
        self, *, messages: list[Message], temperature: float
    ) -> typing.AsyncGenerator[typing.AsyncIterable[str], None]: ...  # raises LLMError
