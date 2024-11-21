import inspect
import typing

import faker
import stamina

import any_llm_client
from tests.conftest import LLMFuncRequest


def test_request_retry_config_default_kwargs_match() -> None:
    config_defaults: typing.Final = inspect.getfullargspec(any_llm_client.RequestRetryConfig).kwonlydefaults
    assert config_defaults
    stamina_defaults: typing.Final = inspect.getfullargspec(stamina.retry).kwonlydefaults
    assert stamina_defaults

    for one_ignored_setting in ("attempts",):
        config_defaults.pop(one_ignored_setting)
        stamina_defaults.pop(one_ignored_setting)

    assert config_defaults == stamina_defaults


def test_llm_error_str(faker: faker.Faker) -> None:
    response_content: typing.Final = faker.pystr().encode()
    assert str(any_llm_client.LLMError(response_content=response_content)) == f"(response_content={response_content!r})"


def test_llm_func_request_has_same_annotations_as_llm_client_methods() -> None:
    all_objects: typing.Final = (
        any_llm_client.LLMClient.request_llm_message,
        any_llm_client.LLMClient.stream_llm_partial_messages,
        LLMFuncRequest,
    )
    all_annotations: typing.Final = [typing.get_type_hints(one_object) for one_object in all_objects]

    for one_ignored_prop in ("return",):
        for annotations in all_annotations:
            if one_ignored_prop in annotations:
                annotations.pop(one_ignored_prop)

    assert all(annotations == all_annotations[0] for annotations in all_annotations)
