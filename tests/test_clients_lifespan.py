import pydantic
import pytest
from polyfactory.factories.pydantic_factory import ModelFactory

import any_llm_client


class ConfigHolder(pydantic.BaseModel):
    config: any_llm_client.AnyLLMConfig


@pytest.mark.parametrize(
    "config", [one_holder.config for one_holder in ModelFactory[ConfigHolder].create_factory(ConfigHolder).coverage()]
)
async def test_lifespan(config: any_llm_client.AnyLLMConfig) -> None:
    async with any_llm_client.get_client(config):
        pass
