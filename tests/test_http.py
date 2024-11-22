import copy
import typing

import httpx

from any_llm_client.http import DEFAULT_HTTP_TIMEOUT, get_http_client_from_kwargs


class TestGetHttpClientFromKwargs:
    def test_http_timeout_is_added(self) -> None:
        original_kwargs: typing.Final = {"mounts": {"http://": None}}
        passed_kwargs: typing.Final = copy.deepcopy(original_kwargs)

        client: typing.Final = get_http_client_from_kwargs(passed_kwargs)

        assert client.timeout == DEFAULT_HTTP_TIMEOUT
        assert original_kwargs == passed_kwargs

    def test_http_timeout_is_not_modified_if_set(self) -> None:
        timeout: typing.Final = httpx.Timeout(7, connect=5, read=3)
        original_kwargs: typing.Final = {"mounts": {"http://": None}, "timeout": timeout}
        passed_kwargs: typing.Final = copy.deepcopy(original_kwargs)

        client: typing.Final = get_http_client_from_kwargs(passed_kwargs)

        assert client.timeout == timeout
        assert original_kwargs == passed_kwargs
