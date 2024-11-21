import contextlib
import dataclasses
import typing

import httpx
import stamina

from any_llm_client.retry import RequestRetryConfig


@contextlib.asynccontextmanager
async def make_request(
    *,
    httpx_client: httpx.AsyncClient,
    request_retry: RequestRetryConfig,
    build_request: typing.Callable[[], httpx.Request],
    stream: bool,
) -> typing.AsyncIterator[httpx.Response]:
    @stamina.retry(on=httpx.HTTPError, **dataclasses.asdict(request_retry))
    async def _make_request_with_retries() -> httpx.Response:
        response: typing.Final = await httpx_client.send(build_request(), stream=stream)
        response.raise_for_status()
        return response

    response = await _make_request_with_retries()
    try:
        yield response
    finally:
        await response.aclose()
