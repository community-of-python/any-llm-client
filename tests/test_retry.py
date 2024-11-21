import inspect

import stamina

from any_llm_client.retry import RequestRetryConfig


def test_request_retry_config_default_kwargs_match() -> None:
    config_defaults = inspect.getfullargspec(RequestRetryConfig).kwonlydefaults
    assert config_defaults
    stamina_defaults = inspect.getfullargspec(stamina.retry).kwonlydefaults
    assert stamina_defaults

    for one_ignored_setting in ("attempts",):
        config_defaults.pop(one_ignored_setting)
        stamina_defaults.pop(one_ignored_setting)

    assert config_defaults == stamina_defaults
