import math
import random
from collections import defaultdict

STATS_EMA_ALPHA = 0.9


class StatsBin:
    def __init__(self, alpha):
        self.val = None
        self.alpha = alpha

    def bump(self, secs):
        if self.val is None:
            self.val = secs
        else:
            self.val = self.alpha * self.val + (1-self.alpha) * secs


class StatsWorker:
    def __init__(self, alpha):
        self.stats: dict[int, StatsBin] = defaultdict(lambda: StatsBin(alpha))

    def bump(self, msize, usage, secs):
        toks = usage.get("total_tokens")
        # similar-sized models are lumped together
        msize_bin = round(math.sqrt(msize))
        # similar-sized token counts are lumped together too
        self.stats[msize_bin].bump(secs/toks)

    def perf(self, msize):
        if not self.stats:
            return None
        msize_bin = round(math.sqrt(msize))
        exact = self.stats.get(msize_bin)
        if exact:
            return exact.val
        close_bin = sorted(self.stats.keys())[-1]
        close = self.stats.get(close_bin)
        # adjustment for model size is (msize/closest_size) ** 1.5
        if msize_bin < close_bin:
            approx = close.val * (msize / (close_bin ** 2.0)) ** 1.5
        else:
            # more conservative for upsizing
            approx = close.val * (msize / (close_bin ** 2.0)) ** 1.8
        return approx

POWER = 2

class StatsContainer:
    def __init__(self, alpha=STATS_EMA_ALPHA):
        self.stats: dict[str, StatsWorker] = defaultdict(lambda: StatsWorker(alpha))
        self.all = StatsWorker(alpha)

    def bump(self, key, msize, usage, secs):
        self.stats[key].bump(msize, usage, secs)
        self.all.bump(msize, usage, secs)

    def perf(self, key, msize):
        wrk = self.stats.get(key)
        if wrk:
            return wrk.perf(msize)
        # assume average perf
        return self.all.perf(msize) or 0

    def pick_best(self, choices, msize):
        ordered = sorted(enumerate(choices), key=lambda ent: self.perf(ent[1], msize))
        # simple skewed range between 0 and 0.5
        pick = int(max(random.random() ** POWER - 0.5, 0) * (len(choices)*2))
        return ordered[pick][0]
