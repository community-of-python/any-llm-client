import copy
import typing

import httpx

from any_llm_client.http import DEFAULT_HTTP_TIMEOUT, get_http_client_from_kwargs


class TestGetHttpClientFromKwargs:
    def test_http_timeout_is_added(self) -> None:
        original_kwargs: typing.Final = {"resolver": None}
        passed_kwargs: typing.Final = copy.deepcopy(original_kwargs)

        client: typing.Final = get_http_client_from_kwargs(passed_kwargs)

        assert client.custom_timeout == DEFAULT_HTTP_TIMEOUT  # type: ignore[attr-defined]
        assert original_kwargs == passed_kwargs

    def test_http_timeout_is_not_modified_if_set(self) -> None:
        timeout: typing.Final = httpx.Timeout(7, connect=5, read=3)
        original_kwargs: typing.Final = {"resolver": None, "timeout": timeout}
        passed_kwargs: typing.Final = copy.deepcopy(original_kwargs)

        client: typing.Final = get_http_client_from_kwargs(passed_kwargs)

        assert client.custom_timeout == timeout  # type: ignore[attr-defined]
        assert original_kwargs == passed_kwargs

    def test_http_proxies_are_added(self) -> None:
        proxies: typing.Final = {"http": "http://example"}
        original_kwargs: typing.Final = {"resolver": None, "proxies": proxies}
        passed_kwargs: typing.Final = copy.deepcopy(original_kwargs)

        client: typing.Final = get_http_client_from_kwargs(passed_kwargs)

        assert client.proxies == proxies
        assert original_kwargs == passed_kwargs
