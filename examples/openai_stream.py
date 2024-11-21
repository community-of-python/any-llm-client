import asyncio  # noqa: INP001
import sys

import any_llm_client


config = any_llm_client.OpenAIConfig(
    url="http://127.0.0.1:11434/v1/chat/completions",
    model_name="qwen2.5-coder:1.5b",
)


async def main() -> None:
    async with (
        any_llm_client.get_client(config) as client,
        client.stream_llm_partial_messages(
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
