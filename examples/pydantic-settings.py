"""Install ollama and pull the model to run this script: `ollama pull qwen2.5-coder:1.5b`."""

import asyncio
import os

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
    async with any_llm_client.get_client(settings.llm_model) as client:
        print(await client.request_llm_message("Кек, чо как вообще на нарах?"))


asyncio.run(main())
