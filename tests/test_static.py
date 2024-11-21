import inspect

import faker
import stamina

import any_llm_client


def test_request_retry_config_default_kwargs_match() -> None:
    config_defaults = inspect.getfullargspec(any_llm_client.RequestRetryConfig).kwonlydefaults
    assert config_defaults
    stamina_defaults = inspect.getfullargspec(stamina.retry).kwonlydefaults
    assert stamina_defaults

    for one_ignored_setting in ("attempts",):
        config_defaults.pop(one_ignored_setting)
        stamina_defaults.pop(one_ignored_setting)

    assert config_defaults == stamina_defaults


def test_llm_error_str(faker: faker.Faker) -> None:
    response_content = faker.pystr().encode()
    assert str(any_llm_client.LLMError(response_content=response_content)) == f"(response_content={response_content!r})"
