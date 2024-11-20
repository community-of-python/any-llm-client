from unittest import mock

import faker
import pytest

import any_llm


def test_unknown_client_raises_assertion_error(faker: faker.Faker) -> None:
    with pytest.raises(AssertionError):
        any_llm.get_model(faker.pyobject(), httpx_client=mock.Mock())  # type: ignore[arg-type]
