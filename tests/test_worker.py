import asyncio

import time

from ai_worker.main import WorkerMain, Config
from httpx_sse import connect_sse

from ai_spider.app import app, get_reg_mgr

from threading import Thread
import httpx
import pytest
from uvicorn import Config as UVConfig, Server


@pytest.fixture(scope="module")
def test_app():
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
async def test_websocket_conn(test_app):

    ws_uri = f"{test_app}/worker"
    wm = WorkerMain(Config(spider_url=ws_uri))

    def wm_run():
        loop = asyncio.new_event_loop()
        loop.run_until_complete(wm.run())

    thread = Thread(target=wm_run, daemon=True)
    thread.start()

    while not get_reg_mgr().socks:
        time.sleep(0.1)

    with httpx.Client() as client:
        res = client.post(f"{test_app}/v1/chat/completions", json={
            "model": "TheBloke/WizardLM-7B-uncensored-GGML:q4_K_M",
            "messages": [
                {"system": "you are a helpful assistant"},
                {"user": "write a frog story"}
            ]
        }, timeout=1000)

        assert res.status_code == 200
        assert res.json()

        with connect_sse(client, "POST", f"{test_app}/v1/chat/completions", json={
            "model": "TheBloke/WizardLM-7B-uncensored-GGML:q4_K_M",
            "stream": True,
            "messages": [
                {"system": "you are a helpful assistant"},
                {"user": "write a frog story"}
            ]
        }, timeout=1000) as sse:
            for ev in sse:
                print(ev.event, ev.data, ev.id, ev.retry)

    assert res.status_code == 200
