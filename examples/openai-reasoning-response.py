"""Install ollama and pull the model to run this script: `ollama pull qwen2.5-coder:1.5b`."""

import asyncio
import typing

import any_llm_client


config = any_llm_client.OpenAIConfig(url="http://127.0.0.1:11434/v1/chat/completions", model_name="qwen2.5-coder:1.5b")


async def main() -> None:
    async with any_llm_client.get_client(config) as client:
        llm_response: typing.Final = await client.request_llm_message(
            "Кек, чо как вообще на нарах? Порассуждай как философ.",
        )
        print(llm_response.reasoning_content)
        print(llm_response.content)


asyncio.run(main())
