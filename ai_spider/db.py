import json
import os
from contextlib import contextmanager

from notanorm import open_db

from ai_spider.stats import StatsStore

DEFAULT_TABLE_NAME = "worker_stats"

g_store: "DbStats"


def init_db_store():
    global g_store
    g_store = DbStats() if os.environ.get("DB_URI") else None


def connect_to_db():
    return open_db(os.environ["DB_URI"])


class DbStats(StatsStore):
    def __init__(self, conn=None, table_name: str = DEFAULT_TABLE_NAME):
        self.table_name = table_name
        self.conn = conn or connect_to_db()
        super().__init__()

    def get(self, key: str) -> dict:
        got = self.conn.select_one(self.table_name, ["val"], wid=key)
        if not got:
            return {}
        return json.loads(got.val)

    def _update(self, key: str, vals: dict):
        self.conn.upsert(self.table_name, wid=key, val=json.dumps(vals))

    @contextmanager
    def _transaction(self):
        with self.conn.transaction():
            yield
