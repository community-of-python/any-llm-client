import contextlib
import typing

import pytest
import stamina
import typing_extensions
from polyfactory.factories.typed_dict_factory import TypedDictFactory

import any_llm_client


@pytest.fixture(scope="session", autouse=True)
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(scope="session", autouse=True)
def _deactivate_retries() -> None:
    stamina.set_active(False)


class LLMFuncRequest(typing.TypedDict):
    messages: str | list[any_llm_client.Message]
    temperature: typing_extensions.NotRequired[float]
    extra: typing_extensions.NotRequired[dict[str, typing.Any] | None]


class LLMFuncRequestFactory(TypedDictFactory[LLMFuncRequest]):
    # Polyfactory ignores `NotRequired`:
    # https://github.com/litestar-org/polyfactory/issues/656
    @classmethod
    def coverage(cls, **kwargs: typing.Any) -> typing.Iterator[LLMFuncRequest]:  # noqa: ANN401
        yield from super().coverage(**kwargs)

        first_additional_example: typing.Final = cls.build(**kwargs)
        first_additional_example.pop("temperature")
        yield first_additional_example

        second_additional_example: typing.Final = cls.build(**kwargs)
        second_additional_example.pop("extra")
        yield second_additional_example

        third_additional_example: typing.Final = cls.build(**kwargs)
        third_additional_example.pop("extra")
        third_additional_example.pop("temperature")
        yield third_additional_example


async def consume_llm_message_chunks(
    stream_llm_message_chunks_context_manager: contextlib._AsyncGeneratorContextManager[typing.AsyncIterable[str]],
    /,
) -> list[str]:
    async with stream_llm_message_chunks_context_manager as response_iterable:
        return [one_item async for one_item in response_iterable]
