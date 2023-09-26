from ai_spider.stats import StatsContainer


def test_stats():
    s = StatsContainer()
    s.bump("id", 7, dict(total_tokens=300), 1)
    s.bump("id", 7, dict(total_tokens=300), 1)
