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


@dataclasses.dataclass(slots=True, init=False)
class HttpClient:
    client: niquests.AsyncSession
    timeout: urllib3.Timeout
    _make_not_streaming_request_with_retries: typing.Callable[[niquests.Request], typing.Awaitable[niquests.Response]]
    _make_streaming_request_with_retries: typing.Callable[[niquests.Request], typing.Awaitable[niquests.AsyncResponse]]
    _retried_exceptions: typing.ClassVar = (niquests.HTTPError, HttpStatusError)

    def __init__(self, request_retry: RequestRetryConfig, niquests_kwargs: dict[str, typing.Any]) -> None:
        modified_kwargs: typing.Final = niquests_kwargs.copy()
        self.timeout = modified_kwargs.pop("timeout", DEFAULT_HTTP_TIMEOUT)
        proxies: typing.Final = modified_kwargs.pop("proxies", None)

        self.client = niquests.AsyncSession(**modified_kwargs)
        if proxies:
            self.client.proxies = proxies

        request_retry_dict: typing.Final = dataclasses.asdict(request_retry)

        self._make_not_streaming_request_with_retries = stamina.retry(
            on=self._retried_exceptions, **request_retry_dict
        )(self._make_not_streaming_request)
        self._make_streaming_request_with_retries = stamina.retry(on=self._retried_exceptions, **request_retry_dict)(
            self._make_streaming_request
        )

    async def _make_not_streaming_request(self, request: niquests.Request) -> niquests.Response:
        response: typing.Final = await self.client.send(self.client.prepare_request(request), timeout=self.timeout)
        try:
            response.raise_for_status()
        except niquests.HTTPError as exception:
            raise HttpStatusError(status_code=response.status_code, content=response.content) from exception  # type: ignore[arg-type]
        finally:
            response.close()
        return response

    async def request(self, request: niquests.Request) -> bytes:
        response: typing.Final = await self._make_not_streaming_request_with_retries(request)
        return response.content  # type: ignore[return-value]

    async def _make_streaming_request(self, request: niquests.Request) -> niquests.AsyncResponse:
        response: typing.Final = await self.client.send(
            self.client.prepare_request(request), stream=True, timeout=self.timeout
        )
        try:
            response.raise_for_status()
        except niquests.HTTPError as exception:
            status_code: typing.Final = response.status_code
            content: typing.Final = await response.content  # type: ignore[misc]
            await response.close()  # type: ignore[misc]
            raise HttpStatusError(status_code=status_code, content=content) from exception  # type: ignore[arg-type]
        return response  # type: ignore[return-value]

    @contextlib.asynccontextmanager
    async def stream(self, request: niquests.Request) -> typing.AsyncIterator[typing.AsyncIterable[bytes]]:
        response: typing.Final = await self._make_streaming_request_with_retries(request)
        try:
            response.__aenter__()
            yield response.iter_lines()  # type: ignore[misc]
        finally:
            await response.raw.close()  # type: ignore[union-attr]

    async def __aenter__(self) -> typing_extensions.Self:
        await self.client.__aenter__()  # type: ignore[no-untyped-call]
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None:
        await self.client.__aexit__(exc_type, exc_value, traceback)  # type: ignore[no-untyped-call]
