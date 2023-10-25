import asyncio
import contextlib
import json
import time
from asyncio import Queue
from threading import RLock
from typing import Generator, Optional, Tuple

from fastapi import WebSocket, HTTPException
from ai_spider.stats import get_stats, punish_failure
from ai_spider.util import WORKER_TYPES

DEFAULT_JOB_TIMEOUT = 60

PUNISH_BUSY_SECS = 30


class QueueSocket(WebSocket):
    queue: Queue
    results: Queue
    info: dict


def is_web_worker(info):
    return info.get("worker_version", "") == "web" or sum([1 for _ in info.get("web_gpus", [])])


def anon_info(ent, **fin):
    info = ent.info
    fin["worker_version"] = info.get("worker_version")
    nv_gpu_cnt = sum([1 for _ in info.get("nv_gpus", [])])
    cl_gpu_cnt = sum([1 for _ in info.get("cl_gpus", [])])
    web_gpu_cnt = sum([1 for _ in info.get("web_gpus", [])])
    fin["gpu_cnt"] = max(nv_gpu_cnt, cl_gpu_cnt, web_gpu_cnt)
    fin["perf"] = get_stats().perf(ent, 5)
    fin["cnt"] = get_stats().cnt(ent)
    return fin


class WorkerManager:
    def __init__(self):
        self.lock = RLock()
        self.socks = dict[QueueSocket, dict]()
        # todo: clean this up
        # im using it *or* someone said they were busy/not-busy
        self.busy = dict()
        # im actually using it
        self.very_busy = set()

    def register_js(self, sock: QueueSocket, info: dict):
        self.socks[sock] = info

    def drop_worker(self, sock):
        self.socks.pop(sock, None)
        self.busy.pop(sock, None)

    @contextlib.contextmanager
    def get_socket_for_inference(self, msize: int, worker_type: WORKER_TYPES, gpu_filter={}) \
            -> Generator[QueueSocket, None, None]:
        # msize is params adjusted by quant level with a heuristic

        # nv gpu memory is reported in MB
        gpu_needed = msize * 1000

        disk_needed = msize * 1000 * 1.5

        # cpu memory is reported in bytes, it's ok to have less... cuz llama.cpp is good about that
        # todo: this looks wrong, should be 1GB * 0.75 * model_size
        cpu_needed = msize * 100000000 * 0.75

        if worker_type in ("any", "web"):
            cpu_needed = min(cpu_needed, 8000000000)

        with self.lock:
            good = []
            close = []
            for sock, info in self.socks.items():
                cpu_vram = info.get("vram", 0)
                disk_space = info.get("disk_space", 0)
                nv_gpu_ram = sum([el.get("memory", 0) for el in info.get("nv_gpus", [])])
                cl_gpu_ram = sum([el.get("memory", 0) for el in info.get("cl_gpus", [])])
                have_web_gpus = is_web_worker(info)

                if ver := gpu_filter.get("min_version"):
                    try:
                        tup_filter = tuple(int(el) for el in ver.split("."))
                        tup_worker = tuple(int(el) for el in info.get("worker_version").split("."))
                    except (TypeError, ValueError):
                        continue
                    if tup_worker < tup_filter:
                        continue

                if caps := gpu_filter.get("capabilities"):
                    try:
                        worker_caps = set(info.get("capabilities", ["llama-infer"]))
                    except TypeError:
                        continue
                    if not all(c in worker_caps for c in caps):
                        continue

                if wid := gpu_filter.get("pubkey", gpu_filter.get("worker_id")):
                    if info.get("worker_id", info.get("pubkey")) != wid:
                        continue

                if slug := gpu_filter.get("slug"):
                    if info.get("slug") != slug:
                        continue

                if uid := gpu_filter.get("user_id"):
                    if (info.get("auth_key") != "uid:" + str(uid)) and info.get("user_id") != uid:
                        continue

                if worker_type in ("any", "web"):
                    if cpu_needed < cpu_vram and have_web_gpus:
                        # very little ability to check here
                        # todo: end the whole self reporting thing and just bench
                        good.append(sock)

                if worker_type in ("any", "cli"):
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
                num = get_stats().pick_best(good, msize)
                choice = good[num]
            elif len(close):
                num = get_stats().pick_best(close, msize)
                choice = close[num]

            info = self.socks.pop(choice, None)
            self.busy[choice] = info
            self.very_busy.add(choice)

        choice.info = info

        try:
            yield choice
        finally:
            with self.lock:
                self.socks[choice] = info
                self.busy.pop(choice, None)
                self.very_busy.discard(choice)

    def set_busy(self, sock, val):
        with self.lock:
            if sock in self.very_busy:
                return
            if val:
                info = self.socks.pop(sock, None)
                if info:
                    self.busy[sock] = info
            else:
                info = self.busy.pop(sock, None)
                if info:
                    self.socks[sock] = info

    def worker_stats(self):
        connected = len(self.socks)
        busy = len(self.busy)
        return dict(
            connected=connected,
            busy=busy
        )

    def worker_detail(self):
        ret = {}
        for ent in self.socks:
            ret[id(ent)] = ent.info.copy()
            ret[id(ent)]["busy"] = False
            ret[id(ent)]["perf"] = get_stats().perf(ent, 5)
        for ent in self.busy:
            ret[id(ent)] = ent.info.copy()
            ret[id(ent)]["busy"] = True
            ret[id(ent)]["perf"] = get_stats().perf(ent, 5)
        return ret

    def worker_anon_detail(self, *, query):
        anon = {}
        for ent in self.socks:
            if query and not self.filter_match(ent.info, query):
                continue
            anon[id(ent)] = anon_info(ent, busy=False)
        for ent in self.busy:
            if query and not self.filter_match(ent.info, query):
                continue
            anon[id(ent)] = anon_info(ent, busy=True)
        return list(anon.values())

    @staticmethod
    def filter_match(info, query):
        if query and info.get("auth_key") == query:
            return True
        if query and info.get("worker_id") == query:
            return True
        if query and info.get("pubkey") == query:
            return True
        if query and info.get("user_id") == "uid:" + query:
            return True
        return False


async def do_model_job(url: str, req: dict, ws: "QueueSocket", stream=False, stream_timeout=None) -> Generator[Tuple[dict, float], None, None]:
    await ws.queue.put({
        "openai_url": url,
        "openai_req": req
    })
    timeout = req.get("timeout", DEFAULT_JOB_TIMEOUT)
    start_time = time.monotonic()
    js = await asyncio.wait_for(ws.results.get(), timeout=timeout)
    if not js:
        raise HTTPException(status_code=400, detail="No results")
    if err := js.get("error"):
        if err == "busy":
            # don't punish a busy worker for long
            punish_failure(ws, "error: %s" % err, PUNISH_BUSY_SECS)
        else:
            punish_failure(ws, "error: %s" % err)
        raise HTTPException(status_code=400, detail=json.dumps(js))
    end_time = time.monotonic()
    ws.info["current_model"] = req["model"]
    to = stream_timeout or timeout
    if to == -1:
        to = None
    while stream and js:
        yield js, end_time - start_time
        js = await asyncio.wait_for(ws.results.get(), timeout=to)

g_reg_mgr: Optional[WorkerManager] = None


def get_reg_mgr() -> WorkerManager:
    global g_reg_mgr
    if not g_reg_mgr:
        g_reg_mgr = WorkerManager()
    return g_reg_mgr
