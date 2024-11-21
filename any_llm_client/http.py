import contextlib
import dataclasses
import typing

import httpx
import stamina

from any_llm_client.retry import RequestRetryConfig


async def make_http_request(
    *,
    httpx_client: httpx.AsyncClient,
    request_retry: RequestRetryConfig,
    build_request: typing.Callable[[], httpx.Request],
) -> httpx.Response:
    @stamina.retry(on=httpx.HTTPError, **dataclasses.asdict(request_retry))
    async def make_request_with_retries() -> httpx.Response:
        response: typing.Final = await httpx_client.send(build_request())
        response.raise_for_status()
        return response

    return await make_request_with_retries()


@contextlib.asynccontextmanager
async def make_streaming_http_request(
    *,
    httpx_client: httpx.AsyncClient,
    request_retry: RequestRetryConfig,
    build_request: typing.Callable[[], httpx.Request],
) -> typing.AsyncIterator[httpx.Response]:
    @stamina.retry(on=httpx.HTTPError, **dataclasses.asdict(request_retry))
    async def make_request_with_retries() -> httpx.Response:
        response: typing.Final = await httpx_client.send(build_request(), stream=True)
        response.raise_for_status()
        return response

    response: typing.Final = await make_request_with_retries()
    try:
        yield response
    finally:
        await response.aclose()
