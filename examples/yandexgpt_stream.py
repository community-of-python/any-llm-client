import asyncio  # noqa: INP001
import os
import sys

import httpx

import any_llm_client


config = any_llm_client.YandexGPTConfig(
    auth_header=os.environ["YANDEX_AUTH_HEADER"],
    folder_id=os.environ["YANDEX_FOLDER_ID"],
    model_name="yandexgpt",
)


async def main() -> None:
    async with (
        httpx.AsyncClient() as httpx_client,
        any_llm_client.get_client(config, httpx_client=httpx_client).request_llm_partial_responses(
            messages=[
                any_llm_client.Message(role="system", text="Ты — опытный ассистент"),
                any_llm_client.Message(role="user", text="Привет!"),
            ],
            temperature=0.1,
        ) as partial_messages,
    ):
        async for one_message in partial_messages:
            sys.stdout.write(f"\r{one_message}")
            sys.stdout.flush()


asyncio.run(main())
