import contextlib
import dataclasses
import types
import typing

import niquests
import stamina
import typing_extensions
import urllib3

from any_llm_client.retry import RequestRetryConfig


DEFAULT_HTTP_TIMEOUT: typing.Final = urllib3.Timeout(total=None, connect=5.0)


@dataclasses.dataclass
class HttpStatusError(Exception):
    status_code: int
    content: bytes


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

    async def request(self, request: niquests.Request) -> bytes:
        @stamina.retry(on=(niquests.HTTPError, HttpStatusError), **self._request_retry_dict)
        async def make_request_with_retries() -> niquests.Response:
            response: typing.Final = await self.httpx_client.send(
                self.httpx_client.prepare_request(request), timeout=self.timeout
            )
            try:
                response.raise_for_status()
            except niquests.HTTPError as exception:
                raise HttpStatusError(status_code=response.status_code, content=response.content) from exception
            finally:
                response.close()
            return response

        response = await make_request_with_retries()
        return response.content

    @contextlib.asynccontextmanager
    async def stream(
        self, request: typing.Callable[[], niquests.PreparedRequest]
    ) -> typing.AsyncIterator[typing.AsyncIterable[bytes]]:
        @stamina.retry(on=(niquests.HTTPError, HttpStatusError), **self._request_retry_dict)
        async def make_request_with_retries() -> niquests.AsyncResponse:
            response: typing.Final = await self.httpx_client.send(
                self.httpx_client.prepare_request(request), stream=True, timeout=self.timeout
            )
            try:
                response.raise_for_status()
            except niquests.HTTPError as exception:
                status_code: typing.Final = response.status_code
                content: typing.Final = await response.content
                await response.close()
                raise HttpStatusError(status_code=status_code, content=content) from exception
            return response  # type: ignore[return-value]

        response: typing.Final = await make_request_with_retries()
        try:
            response.__aenter__()
            yield response.iter_lines()
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
