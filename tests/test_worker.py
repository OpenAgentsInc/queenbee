import asyncio
import contextlib
import json
import os
import sys

import time
from dataclasses import dataclass
from functools import partial
from multiprocessing import Process
from typing import Any
from unittest.mock import patch

from ai_worker.main import WorkerMain, Config
from dotenv import load_dotenv
from httpx_sse import connect_sse
from notanorm import open_db

from ai_spider.app import app
from ai_spider.db import init_db_store
from ai_spider.workers import get_reg_mgr
from ai_spider.stats import StatsContainer, init_stats
from tests.test_db import create_workers_table
from util import set_bypass_token

from threading import Thread
import httpx
import pytest
from uvicorn import Config as UVConfig, Server
import logging as log

from ai_spider.util import BILLING_URL, BILLING_TIMEOUT

set_bypass_token()
load_dotenv()

# when debugging, allow breakpoints to work nicely
TIMEOUT = 1000 if getattr(sys, 'gettrace', None) and sys.gettrace() else 10


@dataclass
class SPServer:
    url: str
    httpx: Any
    stats: Any


@pytest.fixture(scope="module")
def sp_server(tmp_path_factory):
    td = tmp_path_factory.mktemp("test_worker")
    fil = td / "db"

    db = open_db(f"sqlite:{fil}")
    db.execute(create_workers_table)
    db.close()

    os.environ["DB_URI"] = f"sqlite:{fil}"
    st = init_stats(store=init_db_store())

    config = UVConfig(app=app, host="127.0.0.1", port=0, loop="asyncio")

    with patch("ai_spider.util.httpx") as cli:
        server = Server(config=config)
        thread = Thread(target=server.run)
        thread.daemon = True  # Daemon threads are abruptly stopped at shutdown
        thread.start()

        # uvicorn has no way to wait fo start
        to = time.monotonic() + TIMEOUT
        while not server.started and time.monotonic() < to:
            time.sleep(.1)

        port = server.servers[0].sockets[0].getsockname()[1]

        yield SPServer(url=f"ws://127.0.0.1:{port}", httpx=cli, stats=st)

    for s in server.servers:
        s.close()


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
                "max_tokens": 20,
                "ft_timeout": 60
            }, headers={
                "authorization": "bearer: " + token
            }, timeout=200)

            log.info("got completion")

            assert res.status_code == 200
            js = res.json()
            assert not js.get("error")
            assert js.get("usage")
            post = sp_server.httpx.AsyncClient().post
            wait_for(lambda: post.called)
            post.assert_called_with(BILLING_URL,
                                    json=dict(command="complete", bill_to_token=token,
                                              pay_to_lnurl='DONT_PAY_ME', pay_to_auth='keyme'),
                                    timeout=BILLING_TIMEOUT)

            key = list(sp_server.stats.worker_stats.keys())[0]
            assert key == "keyme" or key == StatsContainer.ALL_KEY
            mgr = get_reg_mgr()
            sock = list(mgr.socks.keys())[0]
            perf1 = sp_server.stats.perf(sock, 7)

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
                perf2 = sp_server.stats.perf(sock, 7)
                assert perf2 > 999
                assert res.status_code == 200


@pytest.mark.manual
def test_embed_live_fe(sp_server):
    ws_uri = f"{sp_server.url}/worker"
    sp_server.httpx.reset_mock()
    with spawn_worker(ws_uri, 2):
        with httpx.Client(timeout=30) as client:
            response = client.post(
                f"{sp_server.url}/v1/embeddings",
                json={
                    "input": ["embedding doc 1", "embedding doc 2"],
                    "model": "fastembed:BAAI/bge-base-en-v1.5"
                },
                headers={
                    "authorization": "bearer: " + os.environ["BYPASS_TOKEN"]
                },
            )
            log.info(response.text)
            assert response.status_code == 200
            data = response.json()
            assert data["object"] == "list"


@pytest.mark.manual
def test_embed_live_llama(sp_server):
    ws_uri = f"{sp_server.url}/worker"
    sp_server.httpx.reset_mock()
    with spawn_worker(ws_uri, 2):
        with httpx.Client(timeout=TIMEOUT) as client:
            response = client.post(
                f"{sp_server.url}/v1/embeddings",
                json={
                    "input": ["embedding doc 1", "embedding doc 2"],
                    "model": "TheBloke/WizardLM-7B-uncensored-GGML:q4_K_M",
                },
                headers={
                    "authorization": "bearer: " + os.environ["BYPASS_TOKEN"]
                },
            )
            log.info(response.text)
            assert response.status_code == 200
            data = response.json()
            assert data["object"] == "list"


def wm_run(ws_uri, loops=1):
    wm = WorkerMain(Config(queen_url=ws_uri, loops=loops, auth_key="keyme"))
    asyncio.run(wm.run())


class MockWorkerMain(WorkerMain):
    def __init__(self, conf, resp):
        super().__init__(conf)
        self.resp = iter(resp)

    async def run_one(self):
        helo = next(self.resp)
        await self.ws_send(json.dumps(helo))
        req_str = await self.ws_recv()
        json.loads(req_str)
        for resp in self.resp:
            if resp.get("DELAY"):
                time.sleep(resp.get("DELAY"))
                continue
            await self.ws_send(json.dumps(resp))


def mock_wm_run(resp, auth_key, ws_uri, loops=1):
    wm = MockWorkerMain(Config(queen_url=ws_uri, loops=loops, auth_key=auth_key), resp)
    asyncio.run(wm.run())


def patched_worker_target(target, ws_uri, loops):
    import ai_worker.main
    with patch.object(ai_worker.main.nvidia_smi, "getInstance") as gi:
        gi().DeviceQuery.return_value = dict(
            count=1,
            driver_version="fake",
            gpu=[
                dict(
                    product_name="nvidia fake",
                    fb_memory_usage={"total": 4000},
                    clocks={"graphics_clock": 400, "unit": "ghz"},
                )
            ]
        )
        target(ws_uri, loops)


@contextlib.contextmanager
def spawn_worker(ws_uri, loops=1, target=wm_run):
    thread = Process(target=patched_worker_target, daemon=True, args=(target, ws_uri, loops))
    thread.start()

    to = time.monotonic() + TIMEOUT
    while not get_reg_mgr().socks and time.monotonic() < to:
        time.sleep(0.1)

    yield

    thread.join(timeout=TIMEOUT)

    to = time.monotonic() + TIMEOUT
    while get_reg_mgr().socks and time.monotonic() < to:
        time.sleep(0.1)


@contextlib.contextmanager
def spawn_fake_worker(ws_uri, responses, loops=1, auth_key="keyme"):
    with spawn_worker(ws_uri=ws_uri, loops=loops, target=partial(mock_wm_run, responses, auth_key)) as wk:
        yield wk


async def test_websocket_stream_one_bad_worker(sp_server):
    ws_uri = f"{sp_server.url}/worker"
    with spawn_fake_worker(ws_uri,
                           [{"worker_version": "9.9.9"}, {"DELAY": 2}, {"choices": [{"delta": {"content": "ok"}}]}, {}],
                           auth_key="w1"):
        with spawn_fake_worker(ws_uri, [{"worker_version": "9.9.9"}, {"choices": [{"delta": {"content": "ok"}}]}, {}],
                               auth_key="w2"):
            mgr = get_reg_mgr()

            to = time.monotonic() + TIMEOUT
            while len(list(mgr.socks.keys())) < 2 and time.monotonic() < to:
                time.sleep(0.1)

            # ensure w1 is chosen
            sp_server.stats.bump("w1", 7, {"total_tokens": 1000}, 0.1)
            sp_server.stats.bump("w2", 7, {"total_tokens": 1000}, 10000)

            with httpx.Client(timeout=30) as client:
                with connect_sse(client, "POST", f"{sp_server.url}/v1/chat/completions", json={
                    "model": "TheBloke/WizardLM-7B-uncensored-GGML:q4_K_M",
                    "stream": True,
                    "messages": [
                        {"role": "system", "content": "you are a helpful assistant"},
                        {"role": "user", "content": "write a story about a frog"}
                    ],
                    "max_tokens": 100,
                    "ft_timeout": 1
                }, headers={
                    "authorization": "bearer: " + os.environ["BYPASS_TOKEN"]
                }, timeout=1000) as sse:
                    events = [ev for ev in sse.iter_sse()]
                    assert len(events)


###
#
#  These tests need to be run separately
#  Possibly a bug in using httpx as an async stream request against a test app.
#  Getting rid of the use of the test app could fix this.
#
###

@pytest.mark.manual
async def test_websocket_xx_stream(sp_server):
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
                "max_tokens": 100,
                "ft_timeout": 60
            }, headers={
                "authorization": "bearer: " + os.environ["BYPASS_TOKEN"]
            }, timeout=1000) as sse:
                events = [ev for ev in sse.iter_sse()]
                assert len(events) >= 2
                assert json.loads(events[-1].data).get("usage").get("completion_tokens")


@pytest.mark.manual
async def test_websocket_xx_slow(sp_server):
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
                "max_tokens": 100,
                "ft_timeout": 0
            }, headers={
                "authorization": "bearer: " + os.environ["BYPASS_TOKEN"]
            }, timeout=1000) as sse:
                events = [ev for ev in sse.iter_sse()]
                assert len(events) == 1
                assert json.loads(events[-1].data).get("error")
