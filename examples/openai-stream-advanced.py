"""Install ollama and pull the model to run this script: `ollama pull qwen2.5-coder:1.5b`."""

import asyncio

import any_llm_client


config = any_llm_client.OpenAIConfig(url="http://127.0.0.1:11434/v1/chat/completions", model_name="qwen2.5-coder:1.5b")


async def main() -> None:
    async with (
        any_llm_client.get_client(config) as client,
        client.stream_llm_message_chunks(
            messages=[
                any_llm_client.SystemMessage("Ты — опытный ассистент"),
                any_llm_client.UserMessage("Кек, чо как вообще на нарах?"),
            ],
            temperature=1.0,
        ) as message_chunks,
    ):
        async for chunk in message_chunks:
            print(chunk.content, end="", flush=True)


asyncio.run(main())
