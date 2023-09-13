import json
import logging
import os
from asyncio import Queue
from typing import Iterator, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, Request

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


def check_creds_and_funds(request, args: CreateChatCompletionRequest):
    return True


@app.post("/v1/chat/completions")
async def create_chat_completion(
        request: Request,
        body: CreateChatCompletionRequest,
) -> ChatCompletion:
    check_creds_and_funds(request, body)

    mgr = get_reg_mgr()
    ws = mgr.get_socket_for_inference(body)

    await ws.queue.put({
        "openai_url": "/v1/chat/completions",
        "openai_req": body.model_dump(mode="json")
    })

    if body.stream:
        async def stream() -> Iterator[CompletionChunk]:
            while True:
                try:
                    msg = await ws.receive_text()
                    if not msg:
                        break
                    yield msg
                    if "error" in msg:
                        try:
                            js = json.loads(msg)
                            if js.get("error"):
                                log.info("got an error: %s", js["error"])
                                break
                        except json.JSONDecodeError:
                            pass
                except Exception as ex:
                    log.exception("error during stream")
                    yield json.dumps({"error": str(ex), "error_type": type(ex).__name__})
        return EventSourceResponse(stream())
    else:
        js = await ws.receive_json()
        return js


class QueueSocket(WebSocket):
    queue: Queue


class Worker:
    def __init__(self, sock, info):
        self.sock = sock
        self.info = info


class WorkerManager:
    def __init__(self):
        self.socks = dict()

    def register_js(self, sock, info):
        w = Worker(sock=sock, info=info)
        self.socks[sock] = info

    def drop_worker(self, sock):
        del self.socks[sock]

    def get_socket_for_inference(self, inf: CreateChatCompletionRequest) -> QueueSocket:
        return next(iter(self.socks.keys()))


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
