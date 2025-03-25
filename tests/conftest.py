import contextlib
import typing
from functools import reduce
from itertools import combinations

import pytest
import stamina
import typing_extensions
from polyfactory.factories import DataclassFactory
from polyfactory.factories.typed_dict_factory import TypedDictFactory

import any_llm_client


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


def no_temperature(llm_func_request: LLMFuncRequest) -> LLMFuncRequest:
    llm_func_request.pop("temperature")
    return llm_func_request


def no_extra(llm_func_request: LLMFuncRequest) -> LLMFuncRequest:
    llm_func_request.pop("extra")
    return llm_func_request


class ImageContentItemFactory(DataclassFactory[any_llm_client.ImageContentItem]): ...


class TextContentItemFactory(DataclassFactory[any_llm_client.TextContentItem]): ...


def message_content_as_image_with_description(llm_func_request: LLMFuncRequest) -> LLMFuncRequest:
    llm_func_request["messages"] = [
        any_llm_client.Message(
            role=any_llm_client.MessageRole.user,
            content=[TextContentItemFactory.build(), ImageContentItemFactory.build()],
        )
    ]
    return llm_func_request


def message_content_one_text_item(llm_func_request: LLMFuncRequest) -> LLMFuncRequest:
    llm_func_request["messages"] = [
        any_llm_client.Message(role=any_llm_client.MessageRole.user, content=[TextContentItemFactory.build()])
    ]
    return llm_func_request


def message_content_one_image_item(llm_func_request: LLMFuncRequest) -> LLMFuncRequest:
    llm_func_request["messages"] = [
        any_llm_client.Message(role=any_llm_client.MessageRole.user, content=[ImageContentItemFactory.build()])
    ]
    return llm_func_request


MUTATIONS = (no_temperature, no_extra)
ADDITIONAL_OPTIONS = (
    message_content_as_image_with_description,
    message_content_one_text_item,
    message_content_one_image_item,
)


class LLMFuncRequestFactory(TypedDictFactory[LLMFuncRequest]):
    # Polyfactory ignores `NotRequired`:
    # https://github.com/litestar-org/polyfactory/issues/656
    @classmethod
    def coverage(cls, **kwargs: typing.Any) -> typing.Iterator[LLMFuncRequest]:  # noqa: ANN401
        yield from super().coverage(**kwargs)

        for one_combination in combinations(MUTATIONS, len(MUTATIONS)):
            yield reduce(lambda accumulation, func: func(accumulation), one_combination, cls.build(**kwargs))
            for one_additional_option in ADDITIONAL_OPTIONS:
                yield reduce(
                    lambda accumulation, func: func(accumulation),
                    (*one_combination, one_additional_option),
                    cls.build(**kwargs),
                )


async def consume_llm_message_chunks(
    stream_llm_message_chunks_context_manager: contextlib._AsyncGeneratorContextManager[typing.AsyncIterable[str]],
    /,
) -> list[str]:
    async with stream_llm_message_chunks_context_manager as response_iterable:
        return [one_item async for one_item in response_iterable]
