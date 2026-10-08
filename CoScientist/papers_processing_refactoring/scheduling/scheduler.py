from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, Optional

from ..domain.entities import Article
from ..sources.base import ArticleSource


class Schedule:
    def __init__(self, interval: timedelta):
        self.interval = interval


class IngestionScheduler:

    def __init__(self, on_batch: Callable[[list[Article]], None]):
        self._on_batch = on_batch
        self._sources: Dict[ArticleSource, Schedule] = {}
        self._last_run: Dict[ArticleSource, Optional[datetime]] = {}

    def register(self, source: ArticleSource, schedule: Schedule) -> None:
        self._sources[source] = schedule
        self._last_run.setdefault(source, datetime.min.replace(tzinfo=timezone.utc))

    def poll(self, now: datetime | None = None) -> None:
        now = now or datetime.now(timezone.utc)

        for source, schedule in self._sources.items():
            last = self._last_run[source]

            if now - last < schedule.interval:
                continue

            articles = list(source.list_articles())
            if articles:
                self._on_batch(articles)

            self._last_run[source] = now
