import asyncio
import os

import any_llm_client


config = any_llm_client.YandexGPTConfig(
    auth_header=os.environ["YANDEX_AUTH_HEADER"], folder_id=os.environ["YANDEX_FOLDER_ID"], model_name="yandexgpt"
)


async def main() -> None:
    async with (
        any_llm_client.get_client(config) as client,
        client.stream_llm_message_chunks("Кек, чо как вообще на нарах?") as message_chunks,
    ):
        async for chunk in message_chunks:
            print(chunk, end="", flush=True)


asyncio.run(main())
