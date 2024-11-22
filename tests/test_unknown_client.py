import faker
import pytest

import any_llm_client


def test_unknown_client_raises_assertion_error(faker: faker.Faker) -> None:
    with pytest.raises(AssertionError):
        any_llm_client.get_client(faker.pyobject())  # type: ignore[arg-type]
