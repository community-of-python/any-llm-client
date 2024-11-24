import contextlib
import dataclasses
import types
import typing

import httpx
import niquests
import stamina
import typing_extensions
import urllib3

from any_llm_client.retry import RequestRetryConfig


DEFAULT_HTTP_TIMEOUT: typing.Final = urllib3.Timeout(total=None, connect=5.0)


@dataclasses.dataclass
class HttpStatusError(Exception):
    response: niquests.AsyncResponse


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class HttpClient:
    httpx_client: niquests.AsyncSession
    timeout: urllib3.Timeout
    request_retry: RequestRetryConfig
    _request_retry_dict: dict[str, typing.Any]

    @classmethod
    def build(cls, request_retry: RequestRetryConfig, kwargs: dict[str, typing.Any]) -> typing.Self:
        modified_kwargs: typing.Final = kwargs.copy()
        timeout: typing.Final = modified_kwargs.pop("timeout", DEFAULT_HTTP_TIMEOUT)
        proxies: typing.Final = modified_kwargs.pop("proxies", None)

        session: typing.Final = niquests.AsyncSession(**modified_kwargs)
        if proxies:
            session.proxies = proxies
        return cls(
            httpx_client=session,
            timeout=timeout,
            request_retry=request_retry,
            _request_retry_dict=dataclasses.asdict(request_retry),
        )

    async def request(self, request: niquests.Request) -> niquests.Response:
        @stamina.retry(on=(niquests.HTTPError, HttpStatusError), **self._request_retry_dict)
        async def make_request_with_retries() -> niquests.Response:
            response: typing.Final = await self.httpx_client.send(
                self.httpx_client.prepare_request(request), timeout=self.timeout
            )
            try:
                response.raise_for_status()
            except niquests.HTTPError as exception:
                raise HttpStatusError(response) from exception
            finally:
                response.close()
            return response

        return await make_request_with_retries()

    @contextlib.asynccontextmanager
    async def stream(
        self, request: typing.Callable[[], niquests.PreparedRequest]
    ) -> typing.AsyncIterator[niquests.AsyncResponse]:
        @stamina.retry(on=(niquests.HTTPError, HttpStatusError), **self._request_retry_dict)
        async def make_request_with_retries() -> niquests.AsyncResponse:
            response: typing.Final = await self.httpx_client.send(
                self.httpx_client.prepare_request(request), stream=True, timeout=self.timeout
            )
            try:
                response.raise_for_status()
            except niquests.HTTPError as exception:
                response.close()
                raise HttpStatusError(response) from exception
            return response  # type: ignore[return-value]

        response: typing.Final = await make_request_with_retries()
        try:
            response.__aenter__()
            yield response
        finally:
            await response.raw.close()  # type: ignore[union-attr]

    async def __aenter__(self) -> typing_extensions.Self:
        await self.httpx_client.__aenter__()  # type: ignore[no-untyped-call]
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None:
        await self.httpx_client.__aexit__(exc_type, exc_value, traceback)  # type: ignore[no-untyped-call]


def get_http_client_from_kwargs(kwargs: dict[str, typing.Any]) -> niquests.AsyncSession:
    modified_kwargs: typing.Final = kwargs.copy()
    timeout: typing.Final = modified_kwargs.pop("timeout", DEFAULT_HTTP_TIMEOUT)
    proxies: typing.Final = modified_kwargs.pop("proxies", None)

    session: typing.Final = niquests.AsyncSession(**modified_kwargs)
    session.custom_timeout = timeout  # type: ignore[attr-defined]
    if proxies:
        session.proxies = proxies
    return session


async def make_http_request(
    *,
    httpx_client: niquests.AsyncSession,
    request_retry: RequestRetryConfig,
    build_request: typing.Callable[[], niquests.PreparedRequest],
) -> niquests.Response:
    @stamina.retry(on=niquests.HTTPError, **dataclasses.asdict(request_retry))
    async def make_request_with_retries() -> niquests.Response:
        response: typing.Final = await httpx_client.send(build_request(), timeout=httpx_client.custom_timeout)  # type: ignore[attr-defined]
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
        response: typing.Final = await httpx_client.send(
            build_request(),
            stream=True,
            timeout=httpx_client.custom_timeout,  # type: ignore[attr-defined]
        )
        response.raise_for_status()
        return response  # type: ignore[return-value]

    response: typing.Final = await make_request_with_retries()
    try:
        response.__aenter__()
        yield response
    finally:
        await response.raw.close()  # type: ignore[union-attr]
