import inspect

import stamina

from any_llm_client.retry import RequestRetryConfig


def test_default_kwargs_match() -> None:
    assert (
        inspect.getfullargspec(RequestRetryConfig).kwonlydefaults
        == inspect.getfullargspec(stamina.retry).kwonlydefaults
    )
