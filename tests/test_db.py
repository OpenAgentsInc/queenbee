import os
from unittest import mock

import pytest
from notanorm import open_db

from ai_spider.db import DbStats
from ai_spider.stats import StatsContainer

create_workers_table = """
CREATE TABLE IF NOT EXISTS worker_stats (
    wid VARCHAR(32) NOT NULL PRIMARY KEY,
    val TEXT
);
"""


@pytest.fixture
def db():
    conn = open_db("sqlite://:memory:")
    conn.execute(create_workers_table)
    yield conn


def test_stats_uri(db, tmp_path):
    fil = tmp_path / "fil"
    conn = open_db(f"sqlite:{fil}")
    conn.execute(create_workers_table)
    with mock.patch.dict(os.environ, {"DB_URI": f"sqlite:{fil}"}):
        s = DbStats()
        s.update("a", {"x": 1})
        s.wait()
        assert s.get("a")["x"] == 1


def test_stats_race(db):
    s = StatsContainer(store=DbStats(conn=db))
    s.bump("id1", 7, dict(total_tokens=100), 1)
    s.wait()
    del s

    s = StatsContainer(store=DbStats(conn=db))

    # kill processing
    s.load_queue.put(None)
    s.load_thread.join()

    s.load_queue.put("id1")

    # bump before loaded
    s.bump("id1", 7, dict(total_tokens=10), 1)

    # loaded after the bump
    s._load("id1")

    assert s.perf("id1", 7) == 0.055


def test_stats_pick(db):
    s = StatsContainer(store=DbStats(conn=db))
    assert s.pick_best(["idx1"], 1) == 0

    s.bump("idx1", 7, dict(total_tokens=100), 1)
    s.bump("idx2", 7, dict(total_tokens=100), 100)

    s = StatsContainer(store=DbStats(conn=db))

    s.store.wait()

    assert s.store.get("idx1")
    assert not s.store.get("idx3")

    assert not s.get("idx1")

    s.queue_load("idx1")

    s.wait()

    assert s.get("idx1")
