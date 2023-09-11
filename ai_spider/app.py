import logging as log
import os
from typing import List, Annotated, Optional, cast

from fastapi import FastAPI, Depends, WebSocket, Request, WebSocketDisconnect
from pydantic import BaseModel

from .openai_types import CompletionChunk, ChatCompletion

from fastapi.middleware.cors import CORSMiddleware

from starlette.middleware.sessions import SessionMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)


class CreateChatCompletionRequest(BaseModel):
    messages: List[ChatCompletionRequestMessage] = Field(
        default=[], description="A list of messages to generate completions for."
    )
    max_tokens: int = max_tokens_field
    temperature: float = temperature_field
    top_p: float = top_p_field
    mirostat_mode: int = mirostat_mode_field
    mirostat_tau: float = mirostat_tau_field
    mirostat_eta: float = mirostat_eta_field
    stop: Optional[List[str]] = stop_field
    stream: bool = stream_field
    presence_penalty: Optional[float] = presence_penalty_field
    frequency_penalty: Optional[float] = frequency_penalty_field
    logit_bias: Optional[Dict[str, float]] = Field(None)

    # ignored or currently unsupported
    model: Optional[str] = model_field
    n: Optional[int] = 1
    user: Optional[str] = Field(None)

    # llama.cpp specific parameters
    top_k: int = top_k_field
    repeat_penalty: float = repeat_penalty_field
    logit_bias_type: Optional[Literal["input_ids", "tokens"]] = Field(None)



@app.post("/v1/chat/completions")
async def create_chat_completion(
    request: Request,
    body: CreateChatCompletionRequest,
) -> ChatCompletion:
    ws = self.pick_runner(body)
    
    ws.send({
        "openai_url": "/v1/chat/completions",
        "openai_req": body.to_json()
    })

    if body.stream:
        async def stream() -> Iterator[CompletionChunk]:
            while True:
                js = await ws.recv()
                yield CompletionChunk.parse_raw(js)
        return EventSourceResponse(stream())
    else:
        return ws.recv()


@app.websocket("/v1/worker/connect")
async def worker_connect(websocket: WebSocket):
    # request dependencies don't work with websocket, so just roll our own
    await websocket.accept()
    js = await websocket.receive_json()
    mgr = get_reg_mgr()
    mgr.register_js(js)