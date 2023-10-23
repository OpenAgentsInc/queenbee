import asyncio
import os

import aioboto3
import asyncio_atexit
from dotenv import load_dotenv

load_dotenv()

g_sess = None


def session():
    global g_sess
    if not g_sess:
        g_sess = aioboto3.Session(aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
                                  aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"))
    return g_sess


async def get_s3():
    loop = asyncio.get_running_loop()
    if not hasattr(loop, "s3"):
        # aioboto3 has a heavy startup, you don't want to actually do this over and over
        # but it also makes session creation/destruction hard with the context thing
        # maybe there's another way to do this?
        # associating the current session with the current loop works well
        obj = session().client("s3")
        loop.s3 = await obj.__aenter__()
        async def ex():
            await obj.__aexit__(None, None, None)

        # need to close on exit
        asyncio_atexit.register(ex, loop=loop)
    return loop.s3