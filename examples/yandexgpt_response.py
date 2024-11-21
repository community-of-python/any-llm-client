import asyncio  # noqa: INP001
import os
import typing

import any_llm_client


config = any_llm_client.YandexGPTConfig(
    auth_header=os.environ["YANDEX_AUTH_HEADER"],
    folder_id=os.environ["YANDEX_FOLDER_ID"],
    model_name="yandexgpt",
)


async def main() -> None:
    async with any_llm_client.get_client(config) as client:
        response: typing.Final = await client.request_llm_message(
            messages=[
                any_llm_client.Message(role="system", text="Ты — опытный ассистент"),
                any_llm_client.Message(role="user", text="Привет!"),
            ],
            temperature=0.1,
        )
        print(response)  # noqa: T201


asyncio.run(main())
