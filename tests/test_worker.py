import asyncio

import time
from multiprocessing import Process

from ai_worker.main import WorkerMain, Config
from httpx_sse import connect_sse

from ai_spider.app import app, get_reg_mgr

from threading import Thread
import httpx
import pytest
from uvicorn import Config as UVConfig, Server


@pytest.fixture(scope="module")
def sp_server():
    config = UVConfig(app=app, host="127.0.0.1", port=0, loop="asyncio")
    server = Server(config=config)
    thread = Thread(target=server.run)
    thread.daemon = True  # Daemon threads are abruptly stopped at shutdown
    thread.start()

    # uvicorn has no way to wait fo start
    while not server.started:
        time.sleep(.1)

    port = server.servers[0].sockets[0].getsockname()[1]

    yield f"ws://127.0.0.1:{port}"

    server.shutdown()


@pytest.mark.asyncio  # Mark the test as asyncio (as WebSocket is asCopy code
async def test_websocket_conn(sp_server):
    ws_uri = f"{sp_server}/worker"
    spawn_worker(ws_uri)
    with httpx.Client() as client:
        res = client.post(f"{sp_server}/v1/chat/completions", json={
            "model": "TheBloke/WizardLM-7B-uncensored-GGML:q4_K_M",
            "messages": [
                {"role": "system", "content": "you are a helpful assistant"},
                {"role": "user", "content": "write a frog story"}
            ]
        }, timeout=1000)

        assert res.status_code == 200
        js = res.json()
        assert not js.get("error")


def wm_run(ws_uri):
    wm = WorkerMain(Config(spider_url=ws_uri, once=True))
    asyncio.run(wm.run())


def spawn_worker(ws_uri):
    thread = Process(target=wm_run, daemon=True, args=(ws_uri,))
    thread.start()

    while not get_reg_mgr().socks:
        time.sleep(0.1)


@pytest.mark.asyncio  # Mark the test as asyncio (as WebSocket is asCopy code
async def test_websocket_stream(sp_server):
    ws_uri = f"{sp_server}/worker"
    spawn_worker(ws_uri)

    with httpx.Client() as client:
        with connect_sse(client, "POST", f"{sp_server}/v1/chat/completions", json={
            "model": "TheBloke/WizardLM-7B-uncensored-GGML:q4_K_M",
            "stream": True,
            "messages": [
                {"role": "system", "content": "you are a helpful assistant"},
                {"role": "user", "content": "write a frog story"}
            ]
        }, timeout=1000) as sse:
            events = [ev for ev in sse.iter_sse()]
            assert len(events) > 2
