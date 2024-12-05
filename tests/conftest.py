import contextlib
import typing

import pytest
import stamina
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
    temperature: float
    extra: dict[str, typing.Any] | None


class LLMFuncRequestFactory(TypedDictFactory[LLMFuncRequest]): ...


async def consume_llm_message_chunks(
    stream_llm_message_chunks_context_manager: contextlib._AsyncGeneratorContextManager[typing.AsyncIterable[str]],
    /,
) -> list[str]:
    async with stream_llm_message_chunks_context_manager as response_iterable:
        return [one_item async for one_item in response_iterable]
