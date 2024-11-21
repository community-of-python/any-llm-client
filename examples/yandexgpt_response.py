import asyncio  # noqa: INP001
import os

import any_llm_client


config = any_llm_client.YandexGPTConfig(
    auth_header=os.environ["YANDEX_AUTH_HEADER"], folder_id=os.environ["YANDEX_FOLDER_ID"], model_name="yandexgpt"
)


async def main() -> None:
    async with any_llm_client.get_client(config) as client:
        print(await client.request_llm_message("Кек, чо как вообще на нарах?"))  # noqa: T201


asyncio.run(main())
