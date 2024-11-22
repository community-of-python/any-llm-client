import asyncio  # noqa: INP001

import any_llm_client


config = any_llm_client.OpenAIConfig(url="http://127.0.0.1:11434/v1/chat/completions", model_name="qwen2.5-coder:1.5b")


async def main() -> None:
    async with (
        any_llm_client.get_client(config) as client,
        client.stream_llm_partial_messages(
            messages=[
                any_llm_client.SystemMessage("Ты — опытный ассистент"),
                any_llm_client.UserMessage("Кек, чо как вообще на нарах?"),
            ],
            temperature=1.0,
        ) as partial_messages,
    ):
        async for message in partial_messages:
            print("\033[2J")  # clear screen  # noqa: T201
            print(message)  # noqa: T201


asyncio.run(main())
