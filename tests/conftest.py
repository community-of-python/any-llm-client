import contextlib
import typing

import pytest
import stamina


@pytest.fixture(scope="session", autouse=True)
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(scope="session", autouse=True)
def _deactivate_retries() -> None:
    stamina.set_active(False)


async def consume_llm_partial_responses(
    request_llm_partial_responses_context_manager: contextlib._AsyncGeneratorContextManager[typing.AsyncIterable[str]],
    /,
) -> list[str]:
    async with request_llm_partial_responses_context_manager as response_iterable:
        return [one_item async for one_item in response_iterable]
