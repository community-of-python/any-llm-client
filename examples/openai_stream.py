import asyncio  # noqa: INP001
import sys

import httpx
import pydantic

import any_llm_client


config = any_llm_client.OpenAIConfig(
    url=pydantic.HttpUrl("http://127.0.0.1:11434/v1/chat/completions"),  # ollama
    model_name="qwen2.5-coder:1.5b",
)


async def main() -> None:
    async with (
        httpx.AsyncClient() as httpx_client,
        any_llm_client.get_client(config, httpx_client=httpx_client).stream_llm_partial_responses(
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
