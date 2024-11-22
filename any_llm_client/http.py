import contextlib
import dataclasses
import typing

import httpx
import niquests
import stamina

from any_llm_client.retry import RequestRetryConfig


DEFAULT_HTTP_TIMEOUT: typing.Final = httpx.Timeout(None, connect=5.0)


def get_http_client_from_kwargs(kwargs: dict[str, typing.Any]) -> niquests.AsyncSession:
    kwargs_with_defaults: typing.Final = kwargs.copy()
    kwargs_with_defaults.setdefault("timeout", DEFAULT_HTTP_TIMEOUT)
    return niquests.AsyncSession(**kwargs_with_defaults)


async def make_http_request(
    *,
    httpx_client: niquests.AsyncSession,
    request_retry: RequestRetryConfig,
    build_request: typing.Callable[[], niquests.PreparedRequest],
) -> niquests.Response:
    @stamina.retry(on=niquests.HTTPError, **dataclasses.asdict(request_retry))
    async def make_request_with_retries() -> niquests.Response:
        response: typing.Final = await httpx_client.send(build_request())
        response.raise_for_status()
        return response

    return await make_request_with_retries()


@contextlib.asynccontextmanager
async def make_streaming_http_request(
    *,
    httpx_client: niquests.AsyncSession,
    request_retry: RequestRetryConfig,
    build_request: typing.Callable[[], niquests.PreparedRequest],
) -> typing.AsyncIterator[niquests.AsyncResponse]:
    @stamina.retry(on=httpx.HTTPError, **dataclasses.asdict(request_retry))
    async def make_request_with_retries() -> niquests.AsyncResponse:
        response: typing.Final = await httpx_client.send(build_request(), stream=True)
        response.raise_for_status()
        return response  # type: ignore[return-value]

    response: typing.Final = await make_request_with_retries()
    try:
        response.__aenter__()
        yield response
    finally:
        await response.close()
