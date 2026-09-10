"""Анализ эффективности дедупликации по gts.log и БД."""
from __future__ import annotations

import re
import sqlite3
from collections import Counter
from pathlib import Path

import config

LOG_PATH = Path(config.LOG_FILE)
DB_PATH = Path(config.DB_PATH)


def parse_log_metrics(log_text: str) -> dict:
    """Извлекает последний METRICS REPORT и счётчики фильтров."""
    reports = re.findall(
        r"News: (\d+) sent / (\d+) received.*?"
        r"Filters: Src=(\d+), URL=(\d+), Hash=(\d+), Fzy=(\d+), Sem=(\d+).*?"
        r"DB_Dup=(\d+)",
        log_text,
        re.DOTALL,
    )
    if not reports:
        return {}
    last = reports[-1]
    sent, received, src, url, hash_, fzy, sem, db_dup = map(int, last)
    filtered = url + hash_ + fzy + sem + db_dup
    return {
        "sent": sent,
        "received": received,
        "filtered_pre_ai": filtered,
        "filter_rate_pct": round(filtered / received * 100, 1) if received else 0,
        "url": url,
        "hash": hash_,
        "fuzzy": fzy,
        "semantic": sem,
        "db_dup": db_dup,
        "source_filtered": src,
    }


def analyze_db_duplicates(limit: int = 2000) -> dict:
    if not DB_PATH.exists():
        return {"error": "DB not found"}
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT title, link, timestamp FROM events ORDER BY timestamp DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()

    from engine import clean_title, is_fuzzy_duplicate

    titles = [r["title"] for r in rows]
    dup_pairs = 0
    checked = 0
    for i, t1 in enumerate(titles):
        for t2 in titles[i + 1 : i + 51]:  # окно 50 для скорости
            checked += 1
            if is_fuzzy_duplicate(t1, [t2], config.DUPLICATE_TITLE_THRESHOLD):
                dup_pairs += 1
                break

    link_dupes = len(titles) - len({r["link"] for r in rows if r["link"]})
    clean_counts = Counter(clean_title(t) for t in titles)
    exact_dupes = sum(c - 1 for c in clean_counts.values() if c > 1)

    return {
        "events_sampled": len(rows),
        "exact_title_dupes": exact_dupes,
        "duplicate_links": link_dupes,
        "fuzzy_dupes_in_sample": dup_pairs,
        "unique_clean_titles": len(clean_counts),
    }


def main():
    print("=" * 60)
    print("GTS DEDUPLICATION AUDIT")
    print("=" * 60)

    if LOG_PATH.exists():
        text = LOG_PATH.read_text(encoding="utf-8", errors="ignore")
        m = parse_log_metrics(text)
        if m:
            print("\n--- Последний METRICS REPORT (из лога) ---")
            print(f"  Получено RSS:     {m['received']}")
            print(f"  Отправлено в TG:  {m['sent']}")
            print(f"  Отсеяно до AI:    {m['filtered_pre_ai']} ({m['filter_rate_pct']}%)")
            print(f"    URL dup:        {m['url']}")
            print(f"    Hash dup:       {m['hash']}")
            print(f"    Fuzzy dup:      {m['fuzzy']}")
            print(f"    Semantic dup:   {m['semantic']}")
            print(f"    DB dup:         {m['db_dup']}")
            if m["semantic"] == 0:
                print("  ⚠ Semantic=0 — эмбеддинги отключены (нет Gemini/OpenRouter quota)")
        else:
            print("\nMETRICS REPORT не найден в логе")
    else:
        print(f"\nЛог {LOG_PATH} не найден")

    db_stats = analyze_db_duplicates()
    if "error" not in db_stats:
        print("\n--- Аудит БД (последние события) ---")
        print(f"  Событий в выборке:     {db_stats['events_sampled']}")
        print(f"  Точных дублей title:   {db_stats['exact_title_dupes']}")
        print(f"  Повторных URL:         {db_stats['duplicate_links']}")
        print(f"  Fuzzy-дублей (sample): {db_stats['fuzzy_dupes_in_sample']}")
        leak_rate = db_stats["fuzzy_dupes_in_sample"] / max(db_stats["events_sampled"], 1) * 100
        print(f"  Оценка «протечек»:     ~{leak_rate:.1f}% fuzzy-дублей в ленте")

    print("\n--- Текущие настройки ---")
    print(f"  MAX_NEWS_AGE_HOURS:        {config.MAX_NEWS_AGE_HOURS}")
    print(f"  ONLY_NEWS_AFTER_STARTUP:   {config.ONLY_NEWS_AFTER_STARTUP}")
    print(f"  STARTUP_GRACE_MINUTES:     {config.STARTUP_GRACE_MINUTES}")
    print(f"  DUPLICATE_TITLE_THRESHOLD: {config.DUPLICATE_TITLE_THRESHOLD}")
    print(f"  FUZZY_MIN_WORD_JACCARD:    {config.FUZZY_MIN_WORD_JACCARD}")
    print(f"  USE_EMBEDDINGS:            {config.USE_EMBEDDINGS}")


if __name__ == "__main__":
    main()
