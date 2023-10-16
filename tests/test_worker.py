import asyncio
import contextlib
import json
import os

import time
from dataclasses import dataclass
from multiprocessing import Process
from typing import Any
from unittest.mock import patch, AsyncMock

from ai_worker.main import WorkerMain, Config
from dotenv import load_dotenv
from httpx_sse import connect_sse

from ai_spider.app import app, get_reg_mgr, g_stats
from util import set_bypass_token

from threading import Thread
import httpx
import pytest
from uvicorn import Config as UVConfig, Server
import logging as log

from ai_spider.util import BILLING_URL, BILLING_TIMEOUT

set_bypass_token()
load_dotenv()


@dataclass
class SPServer:
    url: str
    httpx: Any


@pytest.fixture(scope="module")
def sp_server():
    config = UVConfig(app=app, host="127.0.0.1", port=0, loop="asyncio")

    with patch("ai_spider.app.httpx") as cli:
        server = Server(config=config)
        thread = Thread(target=server.run)
        thread.daemon = True  # Daemon threads are abruptly stopped at shutdown
        thread.start()

        # uvicorn has no way to wait fo start
        while not server.started:
            time.sleep(.1)

        port = server.servers[0].sockets[0].getsockname()[1]

        yield SPServer(url=f"ws://127.0.0.1:{port}", httpx=cli)

    server.shutdown()


@pytest.mark.asyncio
async def test_websocket_fail(sp_server):
    ws_uri = f"{sp_server.url}/worker"
    with spawn_worker(ws_uri):

        with httpx.Client(timeout=30) as client:
            res = client.post(f"{sp_server.url}/v1/chat/completions", json={
                "model": "TheBloke/WizardLM-7B-uncensored-GGML:q4_ZZ",
                "messages": [
                    {"role": "system", "content": "you are a helpful assistant"},
                    {"role": "user", "content": "write a frog story"}
                ],
                "max_tokens": 20
            }, headers={
                "authorization": "bearer: " + os.environ["BYPASS_TOKEN"]
            }, timeout=1000)

            assert res.status_code >= 400
            js = res.json()
            assert js.get("error")


@pytest.mark.asyncio
@patch("ai_spider.app.SLOW_SECS", 0)
async def test_websocket_slow(sp_server):
    ws_uri = f"{sp_server.url}/worker"
    with spawn_worker(ws_uri):
        with httpx.Client(timeout=30) as client:
            with connect_sse(client, "POST", f"{sp_server.url}/v1/chat/completions", json={
                "model": "TheBloke/WizardLM-7B-uncensored-GGML:q4_K_M",
                "stream": True,
                "messages": [
                    {"role": "system", "content": "you are a helpful assistant"},
                    {"role": "user", "content": "write a story about a frog"}
                ],
                "max_tokens": 100
            }, headers={
                "authorization": "bearer: " + os.environ["BYPASS_TOKEN"]
            }, timeout=1000) as sse:
                events = [ev for ev in sse.iter_sse()]
                assert len(events) > 2
                assert json.loads(events[-1].data).get("error")


def wait_for(func, timeout=5):
    s = time.monotonic()
    last_ex = None

    def try_func():
        nonlocal last_ex
        try:
            if func():
                return func()
        except Exception as ex:
            last_ex = ex

    while not try_func() and time.monotonic() < s + timeout:
        time.sleep(0.1)

    if last_ex:
        raise last_ex


@pytest.mark.asyncio
async def test_websocket_conn(sp_server):
    token = os.environ["BYPASS_TOKEN"]
    ws_uri = f"{sp_server.url}/worker"
    sp_server.httpx.reset_mock()
    with spawn_worker(ws_uri, 2):
        with httpx.Client(timeout=30) as client:
            res = client.post(f"{sp_server.url}/v1/chat/completions", json={
                "model": "TheBloke/WizardLM-7B-uncensored-GGML:q4_K_M",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": "Write a two sentence frog story"}
                ],
                "max_tokens": 20
            }, headers={
                "authorization": "bearer: " + token
            }, timeout=1000)

            log.info("got completion")

            assert res.status_code == 200
            js = res.json()
            assert not js.get("error")
            assert js.get("usage")
            post = sp_server.httpx.AsyncClient().post
            wait_for(lambda: post.called)
            post.assert_called_with(BILLING_URL,
                                    json=dict(command="complete", bill_to_token=token,
                                              pay_to_lnurl='DONT_PAY_ME', pay_to_auth=''),
                                    timeout=BILLING_TIMEOUT)

            sock = list(g_stats.stats.keys())[0]
            perf1 = g_stats.perf(sock, 7)
            assert perf1 < 10

            log.info("try again, with slow patched")

            with patch("ai_spider.app.SLOW_TOTAL_SECS", 0):
                # slow total means punish worker, but only if the model is loaded
                res = client.post(f"{sp_server.url}/v1/chat/completions", json={
                    "model": "TheBloke/WizardLM-7B-uncensored-GGML:q4_K_M",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant"},
                        {"role": "user", "content": "Write a two sentence frog story"}
                    ],
                    "max_tokens": 20
                }, headers={
                    "authorization": "bearer: " + token
                }, timeout=1000)
                perf2 = g_stats.perf(sock, 7)
                assert perf2 > 999
                assert res.status_code == 200


def wm_run(ws_uri, loops=1):
    wm = WorkerMain(Config(queen_url=ws_uri, loops=loops))
    asyncio.run(wm.run())


@contextlib.contextmanager
def spawn_worker(ws_uri, loops=1):
    thread = Process(target=wm_run, daemon=True, args=(ws_uri, loops))
    thread.start()

    while not get_reg_mgr().socks:
        time.sleep(0.1)

    yield

    thread.join()

    while get_reg_mgr().socks:
        time.sleep(0.1)


@pytest.mark.asyncio
async def test_websocket_stream(sp_server):
    ws_uri = f"{sp_server.url}/worker"
    with spawn_worker(ws_uri):
        with httpx.Client(timeout=30) as client:
            with connect_sse(client, "POST", f"{sp_server.url}/v1/chat/completions", json={
                "model": "TheBloke/WizardLM-7B-uncensored-GGML:q4_K_M",
                "stream": True,
                "messages": [
                    {"role": "system", "content": "you are a helpful assistant"},
                    {"role": "user", "content": "write a story about a frog"}
                ],
                "max_tokens": 100
            }, headers={
                "authorization": "bearer: " + os.environ["BYPASS_TOKEN"]
            }, timeout=1000) as sse:
                events = [ev for ev in sse.iter_sse()]
                assert len(events) > 2
                assert json.loads(events[-1].data).get("usage").get("completion_tokens")
