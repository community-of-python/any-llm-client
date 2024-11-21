import contextlib
import dataclasses
import types
import typing

import pydantic
import typing_extensions


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


@dataclasses.dataclass(slots=True, init=False)
class LLMClient(typing.Protocol):
    async def request_llm_message(
        self, messages: str | list[Message], *, temperature: float = 0.2
    ) -> str: ...  # raises LLMError

    @contextlib.asynccontextmanager
    def stream_llm_partial_messages(
        self, messages: str | list[Message], temperature: float = 0.2
    ) -> typing.AsyncIterator[typing.AsyncIterable[str]]: ...  # raises LLMError

    async def __aenter__(self) -> typing_extensions.Self: ...
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None: ...
