import asyncio
import os

import any_llm_client


config = any_llm_client.YandexGPTConfig(
    auth_header=os.environ["YANDEX_AUTH_HEADER"], folder_id=os.environ["YANDEX_FOLDER_ID"], model_name="yandexgpt"
)


async def main() -> None:
    async with (
        any_llm_client.get_client(config) as client,
        client.stream_llm_partial_messages("Кек, чо как вообще на нарах?") as partial_messages,
    ):
        async for message in partial_messages:
            print(message, end="", flush=True)


asyncio.run(main())
