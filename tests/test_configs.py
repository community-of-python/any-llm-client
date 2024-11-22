import copy
import typing

import faker
import httpx
import pydantic
import pytest

from any_llm_client.clients.openai import OPENAI_AUTH_TOKEN_ENV_NAME, OpenAIConfig
from any_llm_client.clients.yandexgpt import (
    YANDEXGPT_AUTH_HEADER_ENV_NAME,
    YANDEXGPT_FOLDER_ID_ENV_NAME,
    YandexGPTConfig,
)
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


class TestOpenAIAuthToken:
    @pytest.fixture(params=[True, False])
    def maybe_set_env(
        self, request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, faker: faker.Faker
    ) -> None:
        if request.param:
            monkeypatch.setenv(OPENAI_AUTH_TOKEN_ENV_NAME, faker.pystr())

    def test_not_pass(self, faker: faker.Faker) -> None:
        config: typing.Final = OpenAIConfig(url=faker.url(), model_name=faker.pystr())
        assert config.auth_token is None

    @pytest.mark.usefixtures("maybe_set_env")
    def test_pass_none(self, faker: faker.Faker) -> None:
        config: typing.Final = OpenAIConfig(url=faker.url(), model_name=faker.pystr(), auth_token=None)
        assert config.auth_token is None

    @pytest.mark.usefixtures("maybe_set_env")
    def test_pass_str(self, faker: faker.Faker) -> None:
        auth_token: typing.Final = faker.pystr()
        config: typing.Final = OpenAIConfig(url=faker.url(), model_name=faker.pystr(), auth_token=auth_token)
        assert config.auth_token is auth_token

    def test_from_env(self, monkeypatch: pytest.MonkeyPatch, faker: faker.Faker) -> None:
        auth_token: typing.Final = faker.pystr()
        monkeypatch.setenv(OPENAI_AUTH_TOKEN_ENV_NAME, auth_token)

        config: typing.Final = OpenAIConfig(url=faker.url(), model_name=faker.pystr())
        assert config.auth_token == auth_token


class TestYandexGPTAuthHeader:
    @pytest.fixture(params=[True, False])
    def maybe_set_env(
        self, request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, faker: faker.Faker
    ) -> None:
        if request.param:
            monkeypatch.setenv(YANDEXGPT_AUTH_HEADER_ENV_NAME, faker.pystr())

    def test_from_env(self, faker: faker.Faker, monkeypatch: pytest.MonkeyPatch) -> None:
        auth_header: typing.Final = faker.pystr()
        monkeypatch.setenv(YANDEXGPT_AUTH_HEADER_ENV_NAME, auth_header)

        config: typing.Final = YandexGPTConfig(model_name=faker.pystr(), folder_id=faker.pystr())
        assert config.auth_header == auth_header

    @pytest.mark.usefixtures("maybe_set_env")
    def test_pass_str(self, faker: faker.Faker) -> None:
        auth_header: typing.Final = faker.pystr()
        config: typing.Final = YandexGPTConfig(
            model_name=faker.pystr(), folder_id=faker.pystr(), auth_header=auth_header
        )
        assert config.auth_header == auth_header

    def test_fails(self, faker: faker.Faker) -> None:
        with pytest.raises(pydantic.ValidationError):
            YandexGPTConfig(model_name=faker.pystr(), folder_id=faker.pystr())


class TestYandexGPTFolderId:
    @pytest.fixture(params=[True, False])
    def maybe_set_env(
        self, request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, faker: faker.Faker
    ) -> None:
        if request.param:
            monkeypatch.setenv(YANDEXGPT_FOLDER_ID_ENV_NAME, faker.pystr())

    def test_from_env(self, faker: faker.Faker, monkeypatch: pytest.MonkeyPatch) -> None:
        folder_id: typing.Final = faker.pystr()
        monkeypatch.setenv(YANDEXGPT_FOLDER_ID_ENV_NAME, folder_id)

        config: typing.Final = YandexGPTConfig(model_name=faker.pystr(), folder_id=folder_id, auth_header=faker.pystr())
        assert config.folder_id == folder_id

    @pytest.mark.usefixtures("maybe_set_env")
    def test_pass_str(self, faker: faker.Faker) -> None:
        folder_id: typing.Final = faker.pystr()
        config: typing.Final = YandexGPTConfig(model_name=faker.pystr(), folder_id=folder_id, auth_header=faker.pystr())
        assert config.folder_id == folder_id

    def test_fails(self, faker: faker.Faker) -> None:
        with pytest.raises(pydantic.ValidationError):
            YandexGPTConfig(model_name=faker.pystr(), auth_header=faker.pystr())


class TestYandexGPTBothAuthHeaderAndFolderId:
    @pytest.fixture(params=[True, False])
    def maybe_set_env(
        self, request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, faker: faker.Faker
    ) -> None:
        if request.param:
            monkeypatch.setenv(YANDEXGPT_AUTH_HEADER_ENV_NAME, faker.pystr())
            monkeypatch.setenv(YANDEXGPT_FOLDER_ID_ENV_NAME, faker.pystr())

    def test_from_env(self, faker: faker.Faker, monkeypatch: pytest.MonkeyPatch) -> None:
        auth_header, folder_id = faker.pystr(), faker.pystr()
        monkeypatch.setenv(YANDEXGPT_AUTH_HEADER_ENV_NAME, auth_header)
        monkeypatch.setenv(YANDEXGPT_FOLDER_ID_ENV_NAME, folder_id)

        config: typing.Final = YandexGPTConfig(model_name=faker.pystr(), folder_id=folder_id, auth_header=auth_header)
        assert config.auth_header == auth_header
        assert config.folder_id == folder_id

    @pytest.mark.usefixtures("maybe_set_env")
    def test_pass_str(self, faker: faker.Faker) -> None:
        auth_header, folder_id = faker.pystr(), faker.pystr()
        config: typing.Final = YandexGPTConfig(model_name=faker.pystr(), folder_id=folder_id, auth_header=auth_header)
        assert config.auth_header == auth_header
        assert config.folder_id == folder_id

    def test_fails(self, faker: faker.Faker) -> None:
        with pytest.raises(pydantic.ValidationError):
            YandexGPTConfig(model_name=faker.pystr())
