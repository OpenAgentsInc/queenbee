import asyncio
import base64
import logging
import os
import re
from collections import defaultdict
from typing import Literal, Mapping, Any

import httpx
from dotenv import load_dotenv
from fastapi import HTTPException
from starlette.requests import Request

BYPASS_USER = "bypass"

log = logging.getLogger(__name__)

load_dotenv()


def get_bill_to(request):
    req_user = request.headers.get("Authorization")
    bill_to_token = ""
    
    if req_user and " " in req_user:
        bill_to_token = req_user.split(" ")[1]

    if not bill_to_token:
        bill_to_token = request.query_params.get("auth", "")

    return bill_to_token


BILLING_URL = os.environ["BILLING_URL"]
BILLING_TIMEOUT = 20


def get_model_size(model_mame):
    mod = model_mame
    m = re.search(r"(\d)+b(.*)", mod.lower())
    # todo: actually have a nice mapping of model sizes
    if m:
        msize = int(m[1])
        mod = m[2]
    else:
        msize = 13
    m = re.search(r"[Qq](\d)+[_f.-]", mod.lower())
    if m:
        quant = int(m[1])
        if quant == 2:
            msize *= 0.4
        elif quant == 3:
            msize *= 0.5
        elif quant == 4:
            msize *= 0.6
        elif quant == 5:
            msize *= 0.7
        elif quant == 6:
            msize *= 0.8
        elif quant == 8:
            msize *= 1.0
        else:
            # f16
            msize *= 2
    return msize


WORKER_TYPES = Literal["web", "cli", "any"]


async def bill_usage(request, msize: int, usage: dict, worker_info: dict, secs: float):
    # todo: this should bill based on model size * usage
    pay_to_lnurl = worker_info.get("ln_address", worker_info.get("ln_url"))  # todo: ln_url is old.
    pay_to_auth = worker_info.get("auth_key")

    bill_to_token = get_bill_to(request)

    command = dict(
        command="complete",
        bill_to_token=bill_to_token,
        pay_to_lnurl=pay_to_lnurl,
        pay_to_auth=pay_to_auth,
    )

    try:
        client = get_async_client(asyncio.get_running_loop())
        res = await client.post(BILLING_URL, json=command, timeout=BILLING_TIMEOUT)

        log.info("bill %s/%s/%s to: (%s), pay to: (%s)", usage, msize, secs, bill_to_token, worker_info)

        if res.status_code != 200:
            log.error("bill endpoint: %s/%s", res.status_code, res.text)
    except Exception as ex:
        log.error(f"billing error ({ex}): {usage}/{msize}/{secs} to: ({bill_to_token}), pay to: ({worker_info})")

    return True


loop_client: Mapping[Any, httpx.AsyncClient] = defaultdict(lambda: httpx.AsyncClient())


def get_async_client(loop):
    return loop_client[loop]


async def optional_bearer_token(request: Request) -> str:
    return await check_bearer_token(request, optional=True)


async def check_bearer_token(request: Request, optional=False) -> str:
    bill_to_token = get_bill_to(request)

    if bill_to_token == os.environ.get("BYPASS_TOKEN"):
        return BYPASS_USER

    return await query_bearer_token(bill_to_token)


async def query_bearer_token(bill_to_token: str, optional=False, timeout=BILLING_TIMEOUT) -> str:
    command = dict(
        command="check",
        bill_to_token=bill_to_token,
    )

    try:
        res = httpx.post(BILLING_URL, json=command, timeout=timeout)
    except Exception as ex:
        if optional:
            return None
        raise HTTPException(status_code=499, detail="billing endpoint error: %s" % ex)

    js = res.json()

    if js.get("user_id"):
        return js.get("user_id")

    if optional:
        return None
    raise HTTPException(status_code=400, detail="Invalid token")


USER_BUCKET_NAME = os.environ.get("AWS_USER_BUCKET", 'gputopia-user-bucket')
task_set = set()


def b64enc(byt):
    return base64.urlsafe_b64encode(byt).decode()


def b64dec(str_):
    return base64.urlsafe_b64decode(str_)


def schedule_task(coro):
    task = asyncio.create_task(coro)
    task_set.add(task)
    task.add_done_callback(lambda t: task_set.remove(t))
