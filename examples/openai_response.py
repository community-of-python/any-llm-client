import asyncio  # noqa: INP001

import httpx
import pydantic

import any_llm_client


config = any_llm_client.OpenAIConfig(
    url=pydantic.HttpUrl("http://127.0.0.1:11434/v1/chat/completions"),  # ollama
    model_name="qwen2.5-coder:1.5b",
)


async def main() -> None:
    async with httpx.AsyncClient() as httpx_client:
        response = await any_llm_client.get_client(config, httpx_client=httpx_client).request_llm_response(
            messages=[
                any_llm_client.Message(role="system", text="Ты — опытный ассистент"),
                any_llm_client.Message(role="user", text="Привет!"),
            ],
            temperature=0.1,
        )
        print(response)  # noqa: T201


asyncio.run(main())
