import contextlib
import dataclasses
import types
import typing

import pydantic
import typing_extensions


MessageRole = typing.Literal["system", "user", "assistant"]


@pydantic.dataclasses.dataclass(kw_only=True)
class Message:
    role: MessageRole
    text: str


if typing.TYPE_CHECKING:

    @pydantic.dataclasses.dataclass
    class SystemMessage(Message):
        role: typing.Literal["system"] = pydantic.Field("system", init=False)
        text: str

    @pydantic.dataclasses.dataclass
    class UserMessage(Message):
        role: typing.Literal["user"] = pydantic.Field("user", init=False)
        text: str

    @pydantic.dataclasses.dataclass
    class AssistantMessage(Message):
        role: typing.Literal["assistant"] = pydantic.Field("assistant", init=False)
        text: str
else:

    def SystemMessage(text: str) -> Message:  # noqa: N802
        return Message(role="system", text=text)

    def UserMessage(text: str) -> Message:  # noqa: N802
        return Message(role="user", text=text)

    def AssistantMessage(text: str) -> Message:  # noqa: N802
        return Message(role="assistant", text=text)


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
