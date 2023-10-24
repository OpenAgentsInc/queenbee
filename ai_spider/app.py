import asyncio
import json
import logging
import os
import re
import time
import weakref
from asyncio import Queue
from json import JSONDecodeError
from typing import Iterator, Optional, cast

import fastapi
import starlette.websockets
import websockets
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, Request, HTTPException, Depends
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.requests import HTTPConnection

from .db import init_db_store
from .openai_types import CompletionChunk, ChatCompletion, CreateChatCompletionRequest

from fastapi.middleware.cors import CORSMiddleware

from starlette.middleware.sessions import SessionMiddleware
from sse_starlette.sse import EventSourceResponse

from .qlogger import init_log
from .stats import init_stats, get_stats, punish_failure
from .files import app as file_router
from .fine_tune import app as finetune_router
from .util import get_bill_to, BILLING_URL, BILLING_TIMEOUT, get_model_size, WORKER_TYPES, bill_usage, get_async_client, \
    optional_bearer_token, schedule_task
from .workers import get_reg_mgr, QueueSocket, is_web_worker

log = logging.getLogger(__name__)

load_dotenv()

SECRET_KEY = os.environ["SECRET_KEY"]
APP_NAME = os.environ.get("APP_NAME", "GPUTopia QueenBee")
SLOW_SECS = 5
SLOW_TOTAL_SECS = 120
PUNISH_BUSY_SECS = 30

app = FastAPI(
    title=f"{APP_NAME} API",
    description=f"{APP_NAME} inference and worker information API",
    version="1.0",
)

app.include_router(file_router, prefix='/v1', tags=['files'])
app.include_router(finetune_router, prefix='/v1', tags=['files'])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)

init_log()


# change to "error" from "detail" to be compat with openai

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.status_code, "type": type(exc).__name__, "message": exc.detail}}
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"error": {"code": 400, "type": type(exc).__name__, "message": str(exc)}}
    )


@app.exception_handler(AssertionError)
async def assertion_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"error": {"code": 400, "type": type(exc).__name__, "message": str(exc)}}
    )


MODEL_MAP = [
    {
        "re": r"vicuna.*\b7b\b.*\bq4",
        "hf": "TheBloke/vicuna-7B-v1.5-GGUF:Q4_K_M",
        "web": "webgpu/vicuna-v1-7b-q4f32_0",
    },
    {
        "re": r"llama-2.*\b13b\b.*\bq4",
        "hf": "TheBloke/Llama-2-13B-chat-GGUF:Q4_K_M",
        "web": "webgpu/Llama-2-13b-chat-hf-q4f32_1"
    },
    {
        "re": r"llama-2.*\b17b\b.*\bq4",
        "hf": "TheBloke/Llama-2-7B-Chat-GGUF:Q4_K_M",
        "web": "webgpu/Llama-2-7b-chat-hf-q4f32_1"
    }
]


def alt_models(model) -> dict | None:
    for ent in MODEL_MAP:
        if re.match(ent["re"], model, re.I):
            return ent
    return None


async def check_creds_and_funds(request):
    bill_to_token = get_bill_to(request)

    if bill_to_token == os.environ.get("BYPASS_TOKEN"):
        return True

    command = dict(
        command="check",
        bill_to_token=bill_to_token,
    )

    try:
        client = get_async_client(asyncio.get_running_loop())
        res = await client.post(BILLING_URL, json=command, timeout=BILLING_TIMEOUT)
    except Exception as ex:
        raise HTTPException(status_code=500, detail="billing endpoint error: %s" % repr(ex))

    if res.status_code != 200:
        log.error("bill endpoint: %s/%s", res.status_code, res.text)
        raise HTTPException(status_code=500, detail="billing endpoint error: %s/%s" % (res.status_code, res.text))

    js = res.json()

    if js.get("ok"):
        return True

    log.debug("no balance for %s", bill_to_token)
    raise HTTPException(status_code=422, detail="insufficient funds in account, or incorrect auth token")


init_stats(init_db_store())


def record_stats(sock, msize, usage, secs):
    get_stats().bump(sock, msize, usage, secs)


async def check_bill_usage(request, msize: int, js: dict, worker_info: dict, secs: float):
    if js.get("usage"):
        await bill_usage(request, msize, js["usage"], worker_info, secs)


@app.get("/worker/stats", tags=["worker"])
async def worker_stats() -> dict:
    """Simple free/busy total count"""
    mgr = get_reg_mgr()
    return mgr.worker_stats()


@app.get("/worker/detail", tags=["worker"])
async def worker_detail(query: Optional[str] = None, user_id: str = Depends(optional_bearer_token)) -> list:
    """List of all workers, with anonymized info"""
    mgr = get_reg_mgr()
    if query == "user":
        query = "uid:user_id"
    if query and query.startswith("uid:"):
        return []
    return mgr.worker_anon_detail(query=query)


@app.post("/v1/chat/completions")
async def create_chat_completion(
        request: Request,
        body: CreateChatCompletionRequest,
) -> ChatCompletion:
    """Openai compatible chat completion endpoint."""
    await check_creds_and_funds(request)

    worker_type = worker_type_from_model_name(body.model)

    msize = get_model_size(body.model)
    mgr = get_reg_mgr()
    gpu_filter = body.gpu_filter

    try:
        try:
            with mgr.get_socket_for_inference(msize, worker_type, gpu_filter) as ws:
                return await do_inference(request, body, ws)
        except (fastapi.WebSocketDisconnect, HTTPException) as ex:
            if type(ex) is HTTPException and "gguf" in ex.detail:
                raise
            log.error("try again: %s: ", repr(ex))
            await asyncio.sleep(0.5)
            with mgr.get_socket_for_inference(msize, worker_type, gpu_filter) as ws:
                return await do_inference(request, body, ws)
    except HTTPException as ex:
        log.error("inference failed : %s", repr(ex))
        raise
    except TimeoutError as ex:
        log.error("inference failed : %s", repr(ex))
        raise HTTPException(408, detail=repr(ex))
    except AssertionError as ex:
        log.error("inference failed : %s", repr(ex))
        raise HTTPException(400, detail=repr(ex))
    except Exception as ex:
        log.exception("unknown error : %s", repr(ex))
        raise HTTPException(500, detail=repr(ex))


def worker_type_from_model_name(model) -> WORKER_TYPES:
    worker_type: WORKER_TYPES = "cli"
    if model.startswith("webgpu/"):
        worker_type = "web"
    elif alt_models(model):
        worker_type = "any"
    return worker_type


def augment_reply(body: CreateChatCompletionRequest, js, prev_js={}):
    # todo: this should be used to VALIDATE the reply, not "fix" it!
    # todo: this all happens because the web-worker is total hack, need to clean it up

    if not js.get("model"):
        js["model"] = prev_js.get("model", body.model)

    if not js.get("object"):
        js["object"] = "chat.completion"

    if not js.get("created"):
        js["created"] = prev_js.get("created", int(time.time()))

    if not js.get("id"):
        js["id"] = prev_js.get("id", os.urandom(16).hex())

    inp = sum(len(msg.content) for msg in body.messages) // 3
    c0 = js["choices"][0]

    if c0.get("finish_reason"):
        if not js.get("usage"):
            msg = c0.get("message", {}).get("content", "")
            out = len(msg) // 3
            js["usage"] = dict(
                prompt_tokens=int(inp),
                completion_tokens=int(out),
                total_tokens=int(inp + out),
            )


def get_stream_final(body: CreateChatCompletionRequest, prev_js, content_len):
    js = {"choices": [{"finish_reason": "stop", "content": ""}]}
    augment_reply(body, js, prev_js)
    inp = js["usage"]["prompt_tokens"]
    out = content_len // 3
    js["usage"].update(dict(
        completion_tokens=int(out),
        total_tokens=int(inp + out),
    ))
    return js


def adjust_model_for_worker(model, info) -> str:
    model = model.strip()

    want_type = worker_type_from_model_name(model)

    is_web = is_web_worker(info)

    assert not (want_type == "web" and not is_web), f"invalid model for worker: {model}"

    if want_type != "any":
        return model

    alt = alt_models(model)

    assert alt, f"invalid model name {model}"

    if is_web:
        return alt["web"]

    return alt["hf"]


weak_task_set = weakref.WeakSet()


async def do_inference(request, body: CreateChatCompletionRequest, ws: "QueueSocket"):
    # be sure we don't alter the original request, so it can be retried
    body = body.model_copy()

    body.model = adjust_model_for_worker(body.model, ws.info)

    msize = get_model_size(body.model)

    if not msize:
        err = "unknown model size: %s" % body.model
        log.error(err)
        raise HTTPException(status_code=400, detail=err)

    while ws.results.qsize():
        ws.results.get_nowait()

    await ws.queue.put({
        "openai_url": "/v1/chat/completions",
        "openai_req": body.model_dump(mode="json")
    })

    if ws.info.get("current_model", "") == body.model:
        already_loaded = True
    else:
        ws.info["current_model"] = body.model
        # None == unsure.  We're not sure if it's already loaded, because we have no protocol for that yet
        already_loaded = None

    start_time = time.monotonic()

    if body.stream:
        async def stream() -> Iterator[CompletionChunk]:
            prev_js = {}
            total_content_len = 0
            prev_time = start_time
            while True:
                try:
                    js: dict = await asyncio.wait_for(ws.results.get(), timeout=body.timeout)

                    log.debug("got msg %s", js)

                    if not js and prev_js:
                        fin = get_stream_final(body, prev_js, total_content_len)
                        yield json.dumps(fin)
                        # bill when stream is done, for now, could actually charge per stream, but whatever
                        end_time = time.monotonic()
                        schedule_task(check_bill_usage(request, msize, fin, ws.info, end_time - start_time))
                        record_stats(ws, msize, fin.get("usage"), end_time - start_time)
                        break

                    c0 = js["choices"][0]
                    # fix bug:
                    if c0.get("message") and not c0.get("delta"):
                        c0["delta"] = c0.pop("message")

                    c_len = len(c0.get("delta", {}).get("content", ""))
                    cur_time = time.monotonic()
                    token_time = cur_time - prev_time
                    prev_time = cur_time

                    # first token can be long (load time), but between tokens should be fast!
                    if token_time > SLOW_SECS and total_content_len > 0:
                        punish_failure(ws, "slow worker, try again: %s" % token_time)
                        raise HTTPException(status_code=438, detail="slow worker, try again")

                    total_content_len += c_len

                    augment_reply(body, js, prev_js)

                    js["object"] = "chat.completion.chunk"

                    prev_js = js

                    yield json.dumps(js)

                    if c0.get("finish_reason"):
                        schedule_task(check_bill_usage(request, msize, js, ws.info, cur_time - start_time))
                        break

                    if js.get("error"):
                        log.info("got an error: %s", js["error"])
                        raise HTTPException(status_code=400, detail=json.dumps(js))
                except Exception as ex:
                    if isinstance(ex, (KeyError, IndexError)):
                        punish_failure(ws, repr(ex))
                    log.exception("error during stream")
                    yield json.dumps({"error": repr(ex), "error_type": type(ex).__name__})
                    break

        return EventSourceResponse(stream())
    else:
        js = await asyncio.wait_for(ws.results.get(), timeout=body.timeout)
        if err := js.get("error"):
            if err == "busy":
                # don't punish a busy worker for long
                punish_failure(ws, "error: %s" % err, PUNISH_BUSY_SECS)
            else:
                punish_failure(ws, "error: %s" % err)
            raise HTTPException(status_code=400, detail=json.dumps(js))
        end_time = time.monotonic()
        augment_reply(body, js)
        schedule_task(check_bill_usage(request, msize, js, ws.info, end_time - start_time))
        if usage := js.get("usage"):
            record_stats(ws, msize, usage, end_time - start_time)
        if already_loaded and (end_time - start_time) > SLOW_TOTAL_SECS:
            # mark bad... too slow
            punish_failure(ws, "slow inference: %s" % int(end_time - start_time))
        return js


def get_ip(request: HTTPConnection):
    ip = request.client.host
    if not (ip.startswith("192.168") or ip.startswith("10.10")):
        return ip
    if ip := request.headers.get("x-real-ip"):
        return ip
    if ip := request.headers.get("x-forwarded-for"):
        return ip
    return ip


def validate_worker_info(js):
    # pk = js.get("pubkey", None)
    # sig = js.pop("sig", None)
    # todo: raise an error if invalid sig
    pass


@app.websocket("/worker")
async def worker_connect(websocket: WebSocket):
    # request dependencies don't work with websocket, so just roll our own
    try:
        await websocket.accept()
        js = await websocket.receive_json()
    except (websockets.ConnectionClosedOK, RuntimeError, starlette.websockets.WebSocketDisconnect) as ex:
        log.debug("aborted connection before message: %s", repr(ex))
        return

    websocket = cast(QueueSocket, websocket)

    validate_worker_info(js)

    # turn it into a queuesocket by adding "info" and a queue
    websocket.info = js

    # get the source ip, for long-term punishment of bad actors
    req = HTTPConnection(websocket.scope)
    websocket.info["ip"] = get_ip(req)

    websocket.queue = Queue()
    websocket.results = Queue()

    # debug: log everything
    log.debug("connected: %s", websocket.info)

    mgr = get_reg_mgr()
    mgr.register_js(sock=websocket, info=js)

    get_stats().queue_load(websocket)

    while True:
        try:
            pending = [asyncio.create_task(ent) for ent in [websocket.queue.get(), websocket.receive_json()]]
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for fut in done:
                try:
                    action = await fut
                    # this is either a queued result, a new request or a busy statement
                    # could distinguish them above with some intermediate functions that return tuples, but no need yet
                    if "openai_req" in action:
                        log.info("action %s", action)
                        while not websocket.results.empty():
                            # toss previous results if any
                            websocket.results.get_nowait()
                        await websocket.send_json(action)
                    elif "busy" in action:
                        log.info("action %s", action)
                        mgr.set_busy(websocket, action.get("busy"))
                    else:
                        log.debug("action %s", action)
                        await websocket.results.put(action)
                except JSONDecodeError:
                    log.warning("punish worker failure")
                    punish_failure(websocket, "json decode error")
                    # continue so we don't get a new ws, and lose stats
                    # todo: do this by inbound ip instead of ws handle, so they persist across connections
                    # then we can disconnect here if we want, and even block reconn for a while
                except (websockets.ConnectionClosedOK, RuntimeError, starlette.websockets.WebSocketDisconnect):
                    # disconnect means drop out of the loop
                    raise
                except Exception:
                    # other exceptions could be my logic error, try again until disconnected
                    log.exception("exception in loop")
                finally:
                    # clean up futures
                    for ent in pending:
                        ent.cancel()
        except (websockets.ConnectionClosedOK, RuntimeError, starlette.websockets.WebSocketDisconnect):
            log.info("dropped worker")
            break
    mgr.drop_worker(websocket)
