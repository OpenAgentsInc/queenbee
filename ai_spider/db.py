import json
import os

from notanorm import open_db

from ai_spider.stats import StatsStore

DEFAULT_TABLE_NAME = "worker_stats"


def connect_to_db():
    return open_db(os.environ["DB_URI"])


class DbStats(StatsStore):
    def __init__(self, conn=None, table_name: str = DEFAULT_TABLE_NAME):
        self.table_name = table_name
        self.conn = conn or connect_to_db()
        super().__init__()

    def get(self, key: str) -> dict:
        got = self.conn.select_one(self.table_name, ["val"], key=key)
        if not got:
            return {}
        return json.loads(got.val)

    def _update(self, key: str, vals: dict):
        self.conn.upsert(self.table_name, key=key, val=json.dumps(vals))
