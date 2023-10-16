from collections import defaultdict

from ai_spider.stats import StatsContainer


def test_stats():
    s = StatsContainer()
    s.bump("id", 7, dict(total_tokens=2), 1)
    s.bump("id", 7, dict(total_tokens=2), 1)
    assert s.perf("id", 7) == 1 / 2
    # worse than 1/2 but better than 1 for no-info on 13 gb model
    assert 1 / 2 < s.perf("id", 13) < 1


def test_stats_k():
    s = StatsContainer(key=lambda e: e if isinstance(e, str) else e["id"])
    s.bump({"id": 1}, 7, dict(total_tokens=2), 1)
    s.bump({"id": 1, "whatever": 5}, 7, dict(total_tokens=2), 1)
    s.bump({"id": 2}, 7, dict(total_tokens=4), 1)
    assert s.perf({"id": 1, "ignore": 2}, 7) == 1 / 2
    assert s.perf({"id": 2}, 7) == 1 / 4


def usually(f):
    ent = defaultdict(lambda: 0)
    for _ in range(1000):
        z = f()
        ent[z] += 1
    vals = sorted(ent.items(), key=lambda kv: kv[1], reverse=True)

    # more than double the 2nd best
    assert vals[0][1] > vals[1][1] * 2

    return vals[0][0]


def test_stats_pick():
    s = StatsContainer()
    assert s.pick_best(["id1"], 1) == 0

    s.bump("id1", 7, dict(total_tokens=100), 1)
    s.bump("id2", 7, dict(total_tokens=100), 10)
    s.bump("id3", 7, dict(total_tokens=100), 100)
    assert usually(lambda: s.pick_best(["id1", "id2", "id3"], 13)) == 0
    assert usually(lambda: s.pick_best(["id3", "id2", "id1"], 13)) == 2

    s.punish("id1")
    
    assert usually(lambda: s.pick_best(["id3", "id2", "id1"], 13)) == 1

