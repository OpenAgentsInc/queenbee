import math
import random
import time
from collections import defaultdict
from queue import Empty
from threading import Thread

from ai_spider.unique_queue import UniqueQueue
from sqlalchemy import and_

STATS_EMA_ALPHA = 0.9
PUNISH_SECS = 60 * 15
PUNISH_BAD_PERF = 9999

class StatsStore:
    def __init__(self):
        self.update_queue = UniqueQueue(key=lambda el: el[0])
        self.update_thread = Thread(daemon=True, target=self.update_loop)
        self.update_thread.start()

    def stop(self, join=True):
        self.update_queue.put([None, None])
        if join:
            self.update_thread.join()

    def update_loop(self):
        while True:
            try:
                key, val = self.update_queue.get(timeout=4)
                if key is None:
                    break
                self._update(key, val)
            except Empty:
                pass

    def update(self, key: str, val: dict):
        self.update_queue.put([key, val])

    def _update(self, key: str, vals: dict):
        raise NotImplementedError

    def get(self, key: str) -> dict:
        raise NotImplementedError
    
class MySqlWorkers(StatsStore):
    def __init__(self, connection=None):
        super.__init__()
        self.connection=connection
            
    def _text_to_dict(self, text: str) -> dict:
        # assume this is how we struture the vals field in the db
        # {
        #     bad: 0,
        #     count: 0,
        #     size1: 123,
        #     size2: 456,
        #     size3: 789
        # }
        lines = text.strip().strip("{}").split(",")
        result = {}
        for line in lines:
            key, value = [item.strip() for item in line.split(":")]
            if value.isdigit():
                value = int(value)
            result[key] = value
        return result
    
    def get(self, key: str) -> dict:
        get_query=("""
            SELECT * FROM workers where key = %s
        """)
        cursor = self.connection.cursor()
        cursor.execute(get_query, (key))
        row =  cursor.fetchone()
        cursor.close()
        dictionary_vals = self._text_to_dict(text=row.vals)
        return dictionary_vals
    
    def _update(self, key: str, vals: dict):
        insert_query=("""
        INSERT INTO workers (key, vals)
        VALUES (%s, %s)
        ON DUPLICATE KEY UPDATE
        vals = VALUES(vals)
        """)
        cursor = self.connection.cursor()
        cursor.execute(insert_query, (key, vals))
        self.connection.commit()
        cursor.close()        
    
class StatsBin:
    def __init__(self, alpha, val=None):
        self.val = val
        self.alpha = alpha

    def bump(self, secs):
        if self.val is None:
            self.val = secs
        else:
            self.val = self.alpha * self.val + (1 - self.alpha) * secs
       
            
class StatsWorker:
    def __init__(self, alpha):
        self.alpha = alpha
        self.msize_stats: dict[int, StatsBin] = defaultdict(lambda: StatsBin(alpha))
        self.bad = None
        self.cnt = 0

    def bump(self, msize, usage, secs):
        self.cnt += 1
        toks = usage.get("total_tokens")
        # similar-sized models are lumped together
        msize_bin = round(math.sqrt(msize))
        # similar-sized token counts are lumped together too
        self.msize_stats[msize_bin].bump(secs / toks)
        # all is forgiven
        self.bad = None

    def perf(self, msize):
        if self.bad and self.bad > time.monotonic():
            return PUNISH_BAD_PERF
        if not self.msize_stats:
            return None
        msize_bin = round(math.sqrt(msize))
        exact = self.msize_stats.get(msize_bin)
        if exact:
            return exact.val
        close_bin = sorted(self.msize_stats.keys())[-1]
        close = self.msize_stats.get(close_bin)
        # adjustment for model size is (msize/closest_size) ** 1.5
        if msize_bin < close_bin:
            approx = close.val * (msize / (close_bin ** 2.0)) ** 1.5
        else:
            # more conservative for upsizing
            approx = close.val * (msize / (close_bin ** 2.0)) ** 1.8
        return approx

    def punish(self, secs):
        self.bad = time.monotonic() + secs

    def dump(self):
        return dict(
            dat={k: v.val for k, v in self.msize_stats.items()},
            cnt=self.cnt
        )

    def load(self, dct):
        self.cnt = dct["cnt"]
        self.msize_stats = {k: StatsBin(self.alpha, v) for k, v in dct["dat"].items()}


# 2 == very skewed (prefer first, but the rest are skewed to the front)
# 1 == not skewed (prefer first, but the rest are even)

POWER = 2
    
class StatsContainer:
    ALL_KEY = "<all>"

    def __init__(self, alpha=STATS_EMA_ALPHA, key=None, store: StatsStore = None):
        # create the worker in the db
        store.update(key=key)

        self.worker_stats: dict[str, StatsWorker] = defaultdict(lambda: StatsWorker(alpha))
        self.all = StatsWorker(alpha)
        ident = lambda k: k
        self.key_func = key or ident
        self.store = store
        self.load(self.ALL_KEY)

    def load(self, key):
        if self.store:
            dat = self.store.get(key)
            if dat:
                self.worker_stats[key].load(dat)
                return self.worker_stats.get(key)

    def bump(self, key, msize, usage, secs):
        key = self.key_func(key)
        self.worker_stats[key].bump(msize, usage, secs)

        if self.store:
            if isinstance(key, str) and "<" not in key:
                self.store.update(key, self.worker_stats[key].dump())

        self.all.bump(msize, usage, secs)

        if self.store:
            self.store.update(self.ALL_KEY, self.worker_stats[key].dump())

    def get(self, key):
        key = self.key_func(key)
        wrk = self.worker_stats.get(key)
        if not wrk:
            wrk = self.load(key)
        return wrk

    def perf(self, key, msize):
        wrk = self.get(key)
        if wrk:
            ret = wrk.perf(msize)
            if ret:
                return ret
        # assume average perf
        return self.all.perf(msize) or 0

    def pick_best(self, choices, msize):
        ordered = sorted(enumerate(choices), key=lambda ent: self.perf(ent[1], msize))
        # simple skewed range between 0 and 0.5
        pick = int(max(random.random() ** POWER - 0.5, 0) * (len(choices) * 2))
        return ordered[pick][0]

    def punish(self, key, secs=PUNISH_SECS):
        wrk = self.get(key)
        if wrk:
            wrk.punish(secs, key)

    def cnt(self, key):
        wrk = self.get(key)
        if wrk:
            return wrk.cnt
        return 0
    def drop(self, key):
        self.worker_stats.pop(key, None)