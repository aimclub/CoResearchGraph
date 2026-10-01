"""Paper-level statistics over the ChromaDB collection of paper chunks.

The collection holds one row per chunk, not per paper. A single paper
contributes body chunks, image-description chunks and (from an older pipeline)
summary chunks. Every row carries ``article_id``, ``paper_title``, ``domain``
and ``field`` in its metadata, so paper-level statistics are an aggregation
over chunk metadata: unique ``article_id`` values, each counted once.

Summary chunks are outdated and are excluded. A paper that has nothing but
summary chunks is therefore not counted at all, which is the intended
behaviour - it is not part of the current index.

Reading every chunk's metadata takes time that grows with the collection, so
the MCP tool never scans on request. ``PaperStatisticsCache`` keeps the result
in memory and refreshes it from a background thread: a full scan at startup,
then a cheap ``count()`` every few minutes, with a rescan only when the count
changes or the result gets old. The tool answers from memory, instantly.

Nothing here opens a connection. The caller passes the server's own
``ChromaVectorStore`` - the one the retrieval tools already use - and this
module only reads from it, through ``iter_metadata`` and ``collection.count()``.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Iterable, Iterator, Sequence

if TYPE_CHECKING:
    from CoScientist.papers_processing_refactoring.storage.vector import ChromaVectorStore

logger = logging.getLogger(__name__)

# Chunk roles that carry no current paper content (see ChunkRole in
# CoScientist/papers_processing_refactoring/domain/entities.py).
EXCLUDED_ROLES: tuple[str, ...] = ("summary",)

# Rows per request during a scan. Metadata only, so a page stays a few MB.
DEFAULT_PAGE_SIZE = 5000
# Until a first result exists, a failed scan is retried sooner than the
# regular check, so a database that is briefly down at startup costs a minute.
RETRY_WITHOUT_RESULT_SECONDS = 60
UNKNOWN = "unknown"


@dataclass(frozen=True)
class PaperStatistics:
    """Aggregated state of the paper collection."""

    total_papers: int
    domains: Counter
    fields: Counter
    # Keyed by (domain, field), for the per-domain breakdown.
    domain_fields: Counter
    total_chunks: int
    counted_chunks: int
    skipped_chunks: int


def aggregate_paper_statistics(
    metadatas: Iterable[dict],
    excluded_roles: Sequence[str] = EXCLUDED_ROLES,
) -> PaperStatistics:
    """Aggregate chunk metadata into paper statistics.

    Roles are filtered here rather than through a Chroma ``where`` clause: a
    ``$nin`` filter also drops rows that carry no ``role`` key at all, which
    would silently undercount papers written by an older pipeline version.
    """
    excluded = {role.casefold() for role in excluded_roles}

    # A paper's chunks should all agree on domain and field; counting per paper
    # and taking the majority keeps one mislabelled chunk from moving a paper.
    domains_per_paper: dict[str, Counter] = {}
    fields_per_paper: dict[str, Counter] = {}

    total_chunks = 0
    counted_chunks = 0

    for metadata in metadatas:
        total_chunks += 1
        metadata = metadata or {}

        if str(metadata.get("role", "")).casefold() in excluded:
            continue

        article_id = metadata.get("article_id")
        if not article_id:
            continue

        article_id = str(article_id)
        counted_chunks += 1
        domains_per_paper.setdefault(article_id, Counter())[
            str(metadata.get("domain") or UNKNOWN)
        ] += 1
        fields_per_paper.setdefault(article_id, Counter())[
            str(metadata.get("field") or UNKNOWN)
        ] += 1

    # One resolved (domain, field) pair per paper - every distribution below is
    # a view over these pairs, so all of them share the same denominator.
    paper_pairs = [
        (
            domains_per_paper[article_id].most_common(1)[0][0],
            fields_per_paper[article_id].most_common(1)[0][0],
        )
        for article_id in domains_per_paper
    ]

    return PaperStatistics(
        total_papers=len(paper_pairs),
        domains=Counter(domain for domain, _ in paper_pairs),
        fields=Counter(field for _, field in paper_pairs),
        domain_fields=Counter(paper_pairs),
        total_chunks=total_chunks,
        counted_chunks=counted_chunks,
        skipped_chunks=total_chunks - counted_chunks,
    )


class _ScanStopped(Exception):
    """Raised inside a scan when the cache is shutting down."""


class PaperStatisticsCache:
    """Paper statistics kept current by one background thread.

    ``start()`` launches the thread (idempotent). The thread scans the whole
    collection once, then every ``check_interval_minutes`` compares
    ``collection.count()`` with the chunk count of the last scan and rescans
    when they differ - or when the last result is older than ``max_age_hours``,
    which catches changes that keep the count, such as a paper re-ingested
    with a different field. ``report()`` never touches the database; it
    formats whatever the thread produced last.
    """

    def __init__(
        self,
        store: "ChromaVectorStore",
        check_interval_minutes: float = 10,
        max_age_hours: float = 24,
        page_size: int = DEFAULT_PAGE_SIZE,
        excluded_roles: Sequence[str] = EXCLUDED_ROLES,
    ):
        if check_interval_minutes <= 0 or max_age_hours <= 0:
            raise ValueError("check_interval_minutes and max_age_hours must be positive")
        self._store = store
        self._check_interval = check_interval_minutes * 60
        self._max_age = max_age_hours * 3600
        self._page_size = page_size
        self._excluded_roles = tuple(excluded_roles)

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._stopped = False

        # Guarded by _lock, except _scan_rows: only the thread writes it, and
        # a slightly stale value in a progress message is harmless.
        self._stats: PaperStatistics | None = None
        self._computed_at: datetime | None = None
        self._computed_mono = 0.0
        self._scan_started_at: datetime | None = None  # None: no scan running
        self._scan_started_mono = 0.0
        self._scan_total: int | None = None
        self._scan_rows = 0
        self._error: str | None = None
        self._error_at: datetime | None = None

    # ── lifecycle ───────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background thread, once. A stopped cache stays stopped."""
        with self._lock:
            if self._stopped or self._thread is not None:
                return
            self._thread = threading.Thread(
                target=self._run, name="paper-statistics", daemon=True
            )
            self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        """Ask the thread to stop and wait briefly.

        A request already sent to Chroma cannot be interrupted, so this does
        not wait for it; the thread is a daemon and cannot keep the process
        alive.
        """
        with self._lock:
            self._stopped = True
            thread = self._thread
        self._stop_event.set()
        if thread is not None:
            thread.join(timeout)

    # ── background thread ───────────────────────────────────────────────────

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                if self._needs_scan():
                    self._scan()  # clears the error when it succeeds
                else:
                    # The check reached the database and found nothing new, so
                    # an earlier failure is over.
                    with self._lock:
                        self._error = None
                        self._error_at = None
            except _ScanStopped:
                return
            except Exception as e:
                logger.exception("Paper statistics refresh failed")
                with self._lock:
                    self._error = f"{type(e).__name__}: {e}"
                    self._error_at = _now()
                    self._scan_started_at = None
            with self._lock:
                has_result = self._stats is not None
            delay = self._check_interval
            if not has_result:
                delay = min(delay, RETRY_WITHOUT_RESULT_SECONDS)
            self._stop_event.wait(delay)

    def _needs_scan(self) -> bool:
        with self._lock:
            stats, computed_mono = self._stats, self._computed_mono
        if stats is None:
            return True
        if time.monotonic() - computed_mono >= self._max_age:
            return True
        return self._store.collection.count() != stats.total_chunks

    def _scan(self) -> None:
        total = self._store.collection.count()
        with self._lock:
            self._scan_started_at = _now()
            self._scan_started_mono = time.monotonic()
            self._scan_total = total
            self._scan_rows = 0
        logger.info("Paper statistics: scanning %d chunks", total)

        stats = aggregate_paper_statistics(
            self._counted(self._store.iter_metadata(self._page_size)),
            excluded_roles=self._excluded_roles,
        )

        with self._lock:
            took = time.monotonic() - self._scan_started_mono
            self._stats = stats
            self._computed_at = _now()
            self._computed_mono = time.monotonic()
            self._scan_started_at = None
            self._error = None
            self._error_at = None
        logger.info(
            "Paper statistics: %d papers from %d chunks, scanned in %.1fs",
            stats.total_papers,
            stats.total_chunks,
            took,
        )

    def _counted(self, metadatas: Iterable[dict]) -> Iterator[dict]:
        for metadata in metadatas:
            if self._stop_event.is_set():
                raise _ScanStopped
            self._scan_rows += 1
            yield metadata

    # ── reading ─────────────────────────────────────────────────────────────

    def report(self) -> str:
        """The current statistics as the tool's answer (Markdown, in Russian). Instant."""
        self.start()  # normally already running, started with the server
        with self._lock:
            stats = self._stats
            computed_at, computed_mono = self._computed_at, self._computed_mono
            scan_started_at = self._scan_started_at
            scan_mono, scan_total = self._scan_started_mono, self._scan_total
            error, error_at = self._error, self._error_at
        scan_rows = self._scan_rows
        now_mono = time.monotonic()

        progress = None
        if scan_started_at is not None:
            progress = _progress(
                scan_started_at, now_mono - scan_mono, scan_rows, scan_total
            )

        if stats is None:
            if progress:
                return (
                    "Статистика базы статей ещё рассчитывается "
                    f"({progress}). Повторите запрос через несколько минут."
                )
            if error:
                return (
                    "Не удалось рассчитать статистику базы статей: попытка "
                    f"{_clock(error_at)} завершилась ошибкой ({error}). "
                    "Повторная попытка будет выполнена автоматически."
                )
            return (
                "Статистика базы статей готовится. "
                "Повторите запрос через несколько минут."
            )

        notes = [
            f"Данные на {_clock(computed_at)} ({_ago(now_mono - computed_mono)}). "
            f"Коллекция проверяется на изменения каждые {_every(self._check_interval)}; "
            "при изменениях статистика пересчитывается автоматически."
        ]
        if progress:
            notes.append(
                f"Идёт обновление ({progress}); показаны данные предыдущего расчёта."
            )
        if error:
            notes.append(
                f"Последнее обновление ({_clock(error_at)}) завершилось ошибкой "
                f"({error}); показаны данные предыдущего расчёта."
            )
        return format_paper_statistics(stats, notes)


# ── formatting ───────────────────────────────────────────────────────────────
# The report is Markdown in Russian: the web page renders it as a dashboard,
# agents read the same text. Numbers use the Russian style - a no-break space
# between thousands, a decimal comma.

NBSP = " "

# OpenAlex domain and field names, as the metadata stores them, in Russian.
# A name that is not listed is shown as stored.
RUSSIAN_NAMES = {
    UNKNOWN: "Не указано",
    # domains
    "Physical Sciences": "Физические науки",
    "Life Sciences": "Науки о жизни",
    "Health Sciences": "Науки о здоровье",
    "Social Sciences": "Социальные науки",  # also a field
    # fields
    "Agricultural and Biological Sciences": "Сельскохозяйственные и биологические науки",
    "Arts and Humanities": "Искусство и гуманитарные науки",
    "Biochemistry, Genetics and Molecular Biology": "Биохимия, генетика и молекулярная биология",
    "Business, Management and Accounting": "Бизнес, менеджмент и бухгалтерский учёт",
    "Chemical Engineering": "Химическая технология",
    "Chemistry": "Химия",
    "Computer Science": "Компьютерные науки",
    "Decision Sciences": "Науки о принятии решений",
    "Dentistry": "Стоматология",
    "Earth and Planetary Sciences": "Науки о Земле и планетах",
    "Economics, Econometrics and Finance": "Экономика, эконометрика и финансы",
    "Energy": "Энергетика",
    "Engineering": "Инженерные науки",
    "Environmental Science": "Науки об окружающей среде",
    "Health Professions": "Медицинские профессии",
    "Immunology and Microbiology": "Иммунология и микробиология",
    "Materials Science": "Материаловедение",
    "Mathematics": "Математика",
    "Medicine": "Медицина",
    "Neuroscience": "Нейронауки",
    "Nursing": "Сестринское дело",
    "Pharmacology, Toxicology and Pharmaceutics": "Фармакология, токсикология и фармацевтика",
    "Physics and Astronomy": "Физика и астрономия",
    "Psychology": "Психология",
    "Veterinary": "Ветеринария",
}


def _now() -> datetime:
    return datetime.now().astimezone()


def _clock(moment: datetime | None) -> str:
    return moment.strftime("%d.%m.%Y %H:%M %Z") if moment else "в неизвестное время"


def _ago(seconds: float) -> str:
    minutes = int(seconds // 60)
    if minutes < 1:
        return "меньше минуты назад"
    if minutes < 60:
        return f"{minutes} мин назад"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} ч {minutes} мин назад"


def _every(seconds: float) -> str:
    """The check interval as the freshness note states it: "10 мин", "1,5 мин", "30 с"."""
    if seconds < 60:
        return f"{max(1, round(seconds))} с"
    return f"{seconds / 60:g}".replace(".", ",") + " мин"


def _num(count: int) -> str:
    return f"{count:,}".replace(",", NBSP)


def _pct(part: int, whole: int) -> str:
    return f"{part / whole * 100:.1f}".replace(".", ",") + f"{NBSP}%"


def _plural(count: int, one: str, few: str, many: str) -> str:
    """Russian noun form for a count: 1 статья, 2 статьи, 5 статей, 11 статей."""
    count = abs(count) % 100
    if 11 <= count <= 14:
        return many
    count %= 10
    if count == 1:
        return one
    if 2 <= count <= 4:
        return few
    return many


def _papers(count: int) -> str:
    return f"{_num(count)} {_plural(count, 'статья', 'статьи', 'статей')}"


def _progress(started_at: datetime, elapsed: float, rows: int, total: int | None) -> str:
    parts = [f"начато {_clock(started_at)}"]
    parts.append(
        f"прочитано фрагментов: {_num(rows)} из {_num(total)}"
        if total
        else f"прочитано фрагментов: {_num(rows)}"
    )
    if rows and total and rows < total:
        left = (total - rows) * elapsed / rows
        parts.append(
            "осталось меньше минуты" if left < 60 else f"осталось около {math.ceil(left / 60)} мин"
        )
    return ", ".join(parts)


def _name(stored: str) -> str:
    # A pipe or a line break would break the Markdown table row.
    return RUSSIAN_NAMES.get(stored, stored).replace("|", "/").replace("\n", " ")


def _table(headers: Sequence[str], rows: Iterable[Sequence[str]], align: str) -> str:
    """A Markdown table; ``align`` holds "l" or "r" for each column."""
    marks = {"l": "---", "r": "---:"}
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(marks[a] for a in align) + "|",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _by_count(items: Iterable[tuple[str, int]]) -> list[tuple[str, int]]:
    """Largest first, ties by name; papers without a label always last."""
    return sorted(items, key=lambda item: (item[0] == UNKNOWN, -item[1], _name(item[0])))


def _share_table(name_header: str, share_header: str, counter: Counter, total: int) -> str:
    rows = _by_count(counter.items())
    return _table(
        (name_header, "Статей", share_header),
        [(_name(name), _num(count), _pct(count, total)) for name, count in rows],
        "lrr",
    )


def format_paper_statistics(stats: PaperStatistics, notes: Sequence[str] = ()) -> str:
    """Render the statistics as the report the MCP tool returns: Markdown, in Russian.

    ``notes`` come first: freshness, a running refresh, a failed one. The
    report does not name the collection or where the database runs.
    """
    named_domains = [d for d in stats.domains if d != UNKNOWN]
    named_fields = [f for f in stats.fields if f != UNKNOWN]
    blocks = [
        *notes,
        "## Сводка",
        _table(
            ("Показатель", "Значение"),
            [
                ("Статей в базе", _num(stats.total_papers)),
                ("Областей науки", _num(len(named_domains))),
                ("Научных направлений", _num(len(named_fields))),
                ("Фрагментов в коллекции", _num(stats.total_chunks)),
            ],
            "lr",
        ),
    ]

    if not stats.total_papers:
        blocks.append(
            "Статей пока нет: в коллекции не найдено фрагментов с идентификатором "
            "статьи (устаревшие аннотации не учитываются)."
        )
        return "\n\n".join(blocks)

    fields_of: dict[str, Counter] = {}
    for (domain, field), count in stats.domain_fields.items():
        fields_of.setdefault(domain, Counter())[field] = count
    domains = _by_count(stats.domains.items())

    blocks += [
        "## Области науки",
        _share_table("Область науки", "Доля", stats.domains, stats.total_papers),
        "## Направления внутри областей",
    ]
    for domain, domain_total in domains:
        if domain == UNKNOWN and set(fields_of[domain]) == {UNKNOWN}:
            continue  # nothing to break down; the domain table already counts these papers
        blocks += [
            f"### {_name(domain)} · {_papers(domain_total)}",
            _share_table("Направление", "Доля в области", fields_of[domain], domain_total),
        ]
    blocks += [
        "## Все научные направления",
        _share_table("Научное направление", "Доля", stats.fields, stats.total_papers),
        f"Учтено фрагментов: {_num(stats.counted_chunks)} из {_num(stats.total_chunks)}. "
        "Устаревшие аннотации и фрагменты без идентификатора статьи не учитываются; "
        "область и направление статьи определяются по большинству её фрагментов.",
    ]
    return "\n\n".join(blocks)
