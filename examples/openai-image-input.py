"""Install ollama and pull the model to run this script: `ollama pull llava`."""

import asyncio
import base64
import pathlib
import typing

import any_llm_client


config = any_llm_client.OpenAIConfig(url="http://127.0.0.1:11434/v1/chat/completions", model_name="llava")

with pathlib.Path(__file__).parent.joinpath("example-wikimedia-image.jpg").open("rb") as f:
    image_content: typing.Final = f.read()

message: typing.Final = any_llm_client.UserMessage(
    content=[
        any_llm_client.TextContentItem("What's on the image?"),
        any_llm_client.ImageContentItem(f"data:image/jpeg;base64,{base64.b64encode(image_content).decode('utf-8')}"),
    ],
)


async def main() -> None:
    async with (
        any_llm_client.get_client(config) as client,
        client.stream_llm_message_chunks(messages=[message]) as message_chunks,
    ):
        async for chunk in message_chunks:
            print(chunk.content, end="", flush=True)


asyncio.run(main())
