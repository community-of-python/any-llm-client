import asyncio  # noqa: INP001
import os
import typing

import httpx
import pydantic_settings

import any_llm_client


class Settings(pydantic_settings.BaseSettings):
    llm_model: any_llm_client.AnyLLMConfig


os.environ["LLM_MODEL"] = """{
    "api_type": "openai",
    "url": "http://127.0.0.1:11434/v1/chat/completions",
    "model_name": "qwen2.5-coder:1.5b"
}"""
settings = Settings()


async def main() -> None:
    async with httpx.AsyncClient() as httpx_client:
        response: typing.Final = await any_llm_client.get_client(
            settings.llm_model, httpx_client=httpx_client
        ).request_llm_response(
            messages=[
                any_llm_client.Message(role="system", text="Ты — опытный ассистент"),
                any_llm_client.Message(role="user", text="Привет!"),
            ],
            temperature=0.1,
        )
    print(response)  # noqa: T201


asyncio.run(main())
