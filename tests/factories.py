import typing

from polyfactory.factories.typed_dict_factory import TypedDictFactory

import any_llm


class LLMRequest(typing.TypedDict):
    messages: list[any_llm.Message]
    temperature: float


class LLMRequestFactory(TypedDictFactory[LLMRequest]): ...
