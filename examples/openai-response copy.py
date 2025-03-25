"""Install ollama and pull the model to run this script: ollama pull qwen2.5-coder:1.5b."""

import asyncio
import base64
import pathlib

import any_llm_client


config = any_llm_client.OpenAIConfig(url="http://127.0.0.1:11434/v1/chat/completions", model_name="qwen2.5-coder:1.5b")

with pathlib.Path("~/Documents/IMG_4627.JPG").expanduser().open("rb") as f:
    image_content = f.read()


async def main() -> None:
    async with any_llm_client.get_client(config) as client:
        print(
            await client.request_llm_message(
                [
                    any_llm_client.UserMessage(
                        content=[
                            any_llm_client.TextContentItem("Кек, чо как вообще на нарах?"),
                            any_llm_client.ImageContentItem(
                                f"data:image/jpeg;base64,{base64.b64encode(image_content).decode('utf-8')}"
                            ),
                        ]
                    )
                ]
            )
        )


asyncio.run(main())
