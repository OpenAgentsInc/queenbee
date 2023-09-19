import contextlib
import json
import logging
import os
import random
import re
from asyncio import Queue
from threading import RLock
from typing import Iterator, Optional, Generator

import fastapi
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, Request, HTTPException

from .openai_types import CompletionChunk, ChatCompletion, CreateChatCompletionRequest

from fastapi.middleware.cors import CORSMiddleware

from starlette.middleware.sessions import SessionMiddleware
from sse_starlette.sse import EventSourceResponse

log = logging.getLogger(__name__)

load_dotenv()

SECRET_KEY = os.environ["SECRET_KEY"]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)


def check_creds_and_funds(request):
    # todo: this just needs to check if balance > 0
    return True


def bill_usage(request, msize: int, usage: dict, worker_info: dict):
    # todo: this should bill based on model size * usage
    return True


def check_bill_usage(request, msize: int, js: dict, worker_info: dict):
    if js.get("usage"):
        bill_usage(request, msize, js["usage"], worker_info)


def check_bill_usage_str(request, msize: int, msg: str, worker_info: dict):
    if "usage" in msg:
        try:
            js = json.loads(msg)
            check_bill_usage(request, msize, js, worker_info)
        except json.JSONDecodeError:
            log.error("usage tracking failed: %s", msg)


@app.post("/v1/chat/completions")
async def create_chat_completion(
        request: Request,
        body: CreateChatCompletionRequest,
) -> ChatCompletion:
    check_creds_and_funds(request)

    web_only = body.model.startswith("webgpu/")
    msize = get_model_size(body.model)
    mgr = get_reg_mgr()

    try:
        try:
            with mgr.get_socket_for_inference(msize, web_only) as ws:
                return await do_inference(request, body, ws)
        except fastapi.WebSocketDisconnect:
            with mgr.get_socket_for_inference(msize, web_only) as ws:
                return await do_inference(request, body, ws)
    except Exception as ex:
        raise HTTPException(500, detail=repr(ex))


async def do_inference(request: Request, body: CreateChatCompletionRequest, ws: "QueueSocket"):
    msize = get_model_size(body.model)
    await ws.queue.put({
        "openai_url": "/v1/chat/completions",
        "openai_req": body.model_dump(mode="json")
    })
    if body.stream:
        async def stream() -> Iterator[CompletionChunk]:
            prev_msg = ""
            while True:
                try:
                    msg = await ws.receive_text()
                    if not msg and prev_msg:
                        # bill when stream is done, for now, could actually charge per stream, but whatever
                        check_bill_usage_str(request, msize, prev_msg, ws.info)
                        break
                    prev_msg = msg
                    yield msg

                    if "error" in msg:
                        try:
                            js = json.loads(msg)
                            # top level error in dict, means the backend failed
                            if js.get("error"):
                                log.info("got an error: %s", js["error"])
                                raise HTTPException(status_code=400, detail=json.dumps(js))
                        except json.JSONDecodeError:
                            pass
                except Exception as ex:
                    log.exception("error during stream")
                    yield json.dumps({"error": str(ex), "error_type": type(ex).__name__})

        return EventSourceResponse(stream())
    else:
        js = await ws.receive_json()
        if js.get("error"):
            log.info("got an error: %s", js["error"])
            raise HTTPException(status_code=400, detail=json.dumps(js))
        if ws.info.get("ln_url"):
            js["ln_url"] = ws.info["ln_url"]
        check_bill_usage(request, msize, js, ws.info)
        return js


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


class QueueSocket(WebSocket):
    queue: Queue
    info: dict


class Worker:
    def __init__(self, sock, info):
        self.sock = sock
        self.info = info


class WorkerManager:
    def __init__(self):
        self.lock = RLock()
        self.socks = dict[WebSocket, dict]()
        self.busy = dict()

    def register_js(self, sock: WebSocket, info: dict):
        self.socks[sock] = info

    def drop_worker(self, sock):
        self.socks.pop(sock, None)
        self.busy.pop(sock, None)

    @contextlib.contextmanager
    def get_socket_for_inference(self, msize: int, web_only = False) -> Generator[QueueSocket, None, None]:
        # msize is params adjusted by quant level with a heuristic

        # nv gpu memory is reported in MB
        gpu_needed = msize * 1000

        disk_needed = msize * 1000 * 1.5

        # cpu memory is reported in bytes, it's ok to have less... cuz llama.cpp is good about that
        cpu_needed = msize * 1000000000 * 0.75
        
        if web_only:
            cpu_needed = min(cpu_needed, 8000000000)

        with self.lock:
            good = []
            close = []
            for sock, info in self.socks.items():
                cpu_vram = info.get("vram", 0)
                disk_space = info.get("disk_space", 0)
                nv_gpu_ram = sum([el["memory"] for el in info.get("nv_gpus", [])])
                cl_gpu_ram = sum([el["memory"] for el in info.get("cl_gpus", [])])
                have_web_gpus = sum([1 for el in info.get("web_gpus", [])])

                if web_only:
                    if cpu_needed < cpu_vram and have_web_gpus:
                        # very little ability to check here
                        # todo: end the whole self reporting thing and just bench
                        good.append(sock)
                else:
                    if gpu_needed < nv_gpu_ram and cpu_needed < cpu_vram and disk_needed < disk_space:
                        good.append(sock)
                    elif gpu_needed < cl_gpu_ram and cpu_needed < cpu_vram and disk_needed < disk_space:
                        good.append(sock)
                    elif gpu_needed < nv_gpu_ram * 1.2 and cpu_needed < cpu_vram and disk_needed < disk_space:
                        close.append(sock)
                    elif gpu_needed < cl_gpu_ram * 1.2 and cpu_needed < cpu_vram and disk_needed < disk_space:
                        close.append(sock)

            if not good and not close:
                assert False, "No workers available"

            if len(good):
                num = random.randint(0, len(good) - 1)
                choice = good[num]
            elif len(close):
                num = random.randint(0, len(close) - 1)
                choice = close[num]

            info = self.socks.pop(choice)
            self.busy[choice] = info

        choice.info = info

        yield choice

        with self.lock:
            self.socks[choice] = info
            self.busy.pop(choice)


g_reg_mgr: Optional[WorkerManager] = None


def get_reg_mgr() -> WorkerManager:
    global g_reg_mgr
    if not g_reg_mgr:
        g_reg_mgr = WorkerManager()
    return g_reg_mgr


@app.websocket("/worker")
async def worker_connect(websocket: WebSocket):
    # request dependencies don't work with websocket, so just roll our own
    await websocket.accept()
    js = await websocket.receive_json()
    mgr = get_reg_mgr()
    mgr.register_js(sock=websocket, info=js)
    websocket.queue = Queue()
    while True:
        job = await websocket.queue.get()
        await websocket.send_json(job)
