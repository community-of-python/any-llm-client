import typing

import httpx_sse
from httpx_sse._decoders import SSEDecoder


async def parse_sse_events(response: typing.AsyncIterable[bytes]) -> typing.AsyncIterator[httpx_sse.ServerSentEvent]:
    sse_decoder: typing.Final = SSEDecoder()
    async for one_line in response:
        if event := sse_decoder.decode(one_line.decode().rstrip("\n")):
            yield event
