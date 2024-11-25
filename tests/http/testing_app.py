import typing

import litestar
import litestar.background_tasks
from litestar.response import Stream


@litestar.get("/request-ok")
async def request_ok() -> dict[str, typing.Any]:
    return {"ok": True}


@litestar.get("/request-fail", status_code=418)
async def request_fail() -> dict[str, typing.Any]:
    return {"ok": False}


@litestar.get("/stream-ok")
async def stream_ok() -> Stream:
    return Stream("ok\ntrue")


@litestar.get("/stream-fail")
async def stream_fail() -> Stream:
    return Stream("ok\nfalse", status_code=418)


app = litestar.Litestar(route_handlers=[request_ok, request_fail, stream_ok, stream_fail])
