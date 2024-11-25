import typing
from http import HTTPStatus

import niquests
import pytest

from any_llm_client.http import HttpClient, HttpStatusError
from any_llm_client.retry import RequestRetryConfig


BASE_URL: typing.Final = "http://127.0.0.1:8000"


async def test_http_client_request_ok() -> None:
    client: typing.Final = HttpClient(request_retry=RequestRetryConfig(), niquests_kwargs={})
    result: typing.Final = await client.request(niquests.Request(method="GET", url=f"{BASE_URL}/request-ok"))
    assert result == b'{"ok":true}'


async def test_http_client_request_rail() -> None:
    client: typing.Final = HttpClient(request_retry=RequestRetryConfig(), niquests_kwargs={})
    with pytest.raises(HttpStatusError) as exc_info:
        await client.request(niquests.Request(method="GET", url=f"{BASE_URL}/request-fail"))
    assert exc_info.value.status_code == HTTPStatus.IM_A_TEAPOT
    assert exc_info.value.content == b'{"ok":false}'


async def test_http_client_stream_ok() -> None:
    client: typing.Final = HttpClient(request_retry=RequestRetryConfig(), niquests_kwargs={})
    async with client.stream(niquests.Request(method="GET", url=f"{BASE_URL}/stream-ok")) as response:
        result: typing.Final = [one_chunk async for one_chunk in response]
    assert result == [b"ok", b"true"]


async def test_http_client_stream_rail() -> None:
    client: typing.Final = HttpClient(request_retry=RequestRetryConfig(), niquests_kwargs={})
    with pytest.raises(HttpStatusError) as exc_info:
        await client.stream(niquests.Request(method="GET", url=f"{BASE_URL}/stream-fail")).__aenter__()
    assert exc_info.value.status_code == HTTPStatus.IM_A_TEAPOT
    assert exc_info.value.content == b"ok\nfalse"
