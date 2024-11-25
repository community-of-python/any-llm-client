import contextlib
import typing
from unittest import mock

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


async def consume_llm_partial_responses(
    request_llm_partial_responses_context_manager: contextlib._AsyncGeneratorContextManager[typing.AsyncIterable[str]],
    /,
) -> list[str]:
    async with request_llm_partial_responses_context_manager as response_iterable:
        return [one_item async for one_item in response_iterable]


def _make_async_stream_iterable(lines: str) -> typing.Any:  # noqa: ANN401
    async def iter_lines() -> typing.AsyncIterable[bytes]:
        for line in lines.splitlines():
            yield line.encode()

    return iter_lines()


def mock_http_client(llm_client: any_llm_client.LLMClient, request_mock: mock.AsyncMock) -> any_llm_client.LLMClient:
    assert hasattr(llm_client, "http_client")
    llm_client.http_client = mock.Mock(
        request=request_mock,
        stream=mock.Mock(
            return_value=mock.Mock(
                __aenter__=(
                    mock.AsyncMock(return_value=_make_async_stream_iterable(request_mock.return_value))
                    if isinstance(request_mock.return_value, str)
                    else request_mock
                ),
                __aexit__=mock.AsyncMock(return_value=None),
            )
        ),
    )
    return llm_client
