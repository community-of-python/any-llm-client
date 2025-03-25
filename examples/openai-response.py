"""Install ollama and pull the model to run this script: `ollama pull qwen2.5-coder:1.5b`."""

import asyncio

import any_llm_client


config = any_llm_client.OpenAIConfig(url="http://127.0.0.1:11434/v1/chat/completions", model_name="qwen2.5-coder:1.5b")


async def main() -> None:
    async with any_llm_client.get_client(config) as client:
        print(await client.request_llm_message("Кек, чо как вообще на нарах?"))


asyncio.run(main())
