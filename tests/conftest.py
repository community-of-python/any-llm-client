import contextlib
import typing
from functools import reduce
from itertools import combinations

import faker
import pytest
import stamina
import typing_extensions
from polyfactory.factories import DataclassFactory
from polyfactory.factories.typed_dict_factory import TypedDictFactory

import any_llm_client
from any_llm_client.core import LLMResponse


@pytest.fixture(scope="session", autouse=True)
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(scope="session", autouse=True)
def _deactivate_retries() -> None:
    stamina.set_active(False)


class LLMFuncRequest(typing.TypedDict):
    messages: str | list[any_llm_client.Message]
    temperature: typing_extensions.NotRequired[float]
    extra: typing_extensions.NotRequired[dict[str, typing.Any] | None]


class MessageFactory(DataclassFactory[any_llm_client.Message]): ...


class ImageContentItemFactory(DataclassFactory[any_llm_client.ImageContentItem]): ...


class TextContentItemFactory(DataclassFactory[any_llm_client.TextContentItem]): ...


class LLMResponseFactory(DataclassFactory[any_llm_client.LLMResponse]): ...


@pytest.fixture
def random_llm_response(faker: faker.Faker) -> any_llm_client.LLMResponse:
    return LLMResponseFactory.build(content=faker.pystr())


def set_no_temperature(llm_func_request: LLMFuncRequest) -> LLMFuncRequest:
    llm_func_request.pop("temperature")
    return llm_func_request


def set_no_extra(llm_func_request: LLMFuncRequest) -> LLMFuncRequest:
    llm_func_request.pop("extra")
    return llm_func_request


def set_message_content_as_image_with_description(llm_func_request: LLMFuncRequest) -> LLMFuncRequest:
    llm_func_request["messages"] = [
        MessageFactory.build(content=[TextContentItemFactory.build(), ImageContentItemFactory.build()]),
    ]
    return llm_func_request


def set_message_content_one_text_item(llm_func_request: LLMFuncRequest) -> LLMFuncRequest:
    llm_func_request["messages"] = [MessageFactory.build(content=[TextContentItemFactory.build()])]
    return llm_func_request


def set_message_content_one_image_item(llm_func_request: LLMFuncRequest) -> LLMFuncRequest:
    llm_func_request["messages"] = [MessageFactory.build(content=[ImageContentItemFactory.build()])]
    return llm_func_request


class LLMFuncRequestFactory(TypedDictFactory[LLMFuncRequest]):
    MUTATIONS: typing.Final = (set_no_temperature, set_no_extra)
    ADDITIONAL_VARIANTS: typing.Final = (
        set_message_content_as_image_with_description,
        set_message_content_one_text_item,
        set_message_content_one_image_item,
    )

    # Polyfactory ignores `NotRequired`:
    # https://github.com/litestar-org/polyfactory/issues/656
    @classmethod
    def coverage(cls, **kwargs: typing.Any) -> typing.Iterator[LLMFuncRequest]:  # noqa: ANN401
        yield from super().coverage(**kwargs)

        for one_combination in combinations(cls.MUTATIONS, len(cls.MUTATIONS)):
            yield reduce(lambda accumulation, func: func(accumulation), one_combination, cls.build(**kwargs))
            for one_additional_option in cls.ADDITIONAL_VARIANTS:
                yield reduce(
                    lambda accumulation, func: func(accumulation),
                    (*one_combination, one_additional_option),
                    cls.build(**kwargs),
                )


async def consume_llm_message_chunks(
    stream_llm_message_chunks_context_manager: contextlib._AsyncGeneratorContextManager[
        typing.AsyncIterable[LLMResponse]
    ],
    /,
) -> list[LLMResponse]:
    async with stream_llm_message_chunks_context_manager as response_iterable:
        return [one_item async for one_item in response_iterable]
