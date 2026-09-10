"""
Utilities for SOXS v5.0 signal matching, inference, and factor validation.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import config

# Ticker aliases for matching event_key / title / summary / entities
SOXS_TICKER_ALIASES: Dict[str, List[str]] = {
    "MSFT": ["MSFT", "MICROSOFT"],
    "META": ["META", "FACEBOOK"],
    "AMZN": ["AMZN", "AMAZON"],
    "GOOGL": ["GOOGL", "GOOGLE", "ALPHABET"],
    "NVDA": ["NVDA", "NVIDIA"],
    "AVGO": ["AVGO", "BROADCOM"],
    "AMD": ["AMD", "ADVANCED_MICRO_DEVICES"],
    "MU": ["MU", "MICRON"],
}

_CAPEX_TICKERS = set(config.SOXS_CAPEX_WEIGHTS.keys())
_GUIDANCE_TICKERS = set(config.SOXS_GUIDANCE_WEIGHTS.keys())

_GUIDANCE_KEYWORDS = re.compile(
    r"\b("
    r"guidance|outlook|forecast|revenue\s+(?:outlook|forecast|expectation)|"
    r"earnings\s+(?:outlook|forecast|warning|miss|beat)|"
    r"datacenter\s+demand|data\s+center\s+demand|"
    r"lower(?:ed|ing)?\s+(?:outlook|forecast|guidance|expectations)|"
    r"cut(?:s|ting)?\s+(?:outlook|forecast|guidance)|"
    r"raise(?:s|d|ing)?\s+(?:outlook|forecast|guidance)|"
    r"upgrade(?:d|s)?|downgrade(?:d|s)?"
    r")\b",
    re.IGNORECASE,
)

_CAPEX_KEYWORDS = re.compile(
    r"\b("
    r"capex|capital\s+expenditure|ai\s+spending|data\s+center\s+spending|"
    r"cloud\s+spending|infrastructure\s+spending|"
    r"cut(?:s|ting)?\s+capex|reduce(?:s|d|ing)?\s+(?:capex|spending)|"
    r"boost(?:s|ed|ing)?\s+(?:capex|spending)|increase(?:s|d|ing)?\s+capex"
    r")\b",
    re.IGNORECASE,
)

_BEARISH_GUIDANCE = re.compile(
    r"\b(downgrade|lower(?:ed|ing)?|cut(?:s|ting)?|miss(?:ed|es)?|warning|weak|disappoint)\b",
    re.IGNORECASE,
)
_BULLISH_GUIDANCE = re.compile(
    r"\b(upgrade|raise(?:s|d|ing)?|beat(?:s|ing)?|surge|strong|hike|exceed)\b",
    re.IGNORECASE,
)
_BEARISH_CAPEX = re.compile(
    r"\b(cut(?:s|ting)?|reduce(?:s|d|ing)?|lower(?:ed|ing)?|slow(?:s|ed|ing)?|pause[sd]?|freeze[sd]?)\b",
    re.IGNORECASE,
)
_BULLISH_CAPEX = re.compile(
    r"\b(boost(?:s|ed|ing)?|increase(?:s|d|ing)?|raise(?:s|d|ing)?|expand(?:s|ed|ing)?|surge)\b",
    re.IGNORECASE,
)


def _normalize_text(*parts: Optional[str]) -> str:
    blob = " ".join(p for p in parts if p).upper()
    return re.sub(r"[^A-Z0-9_\s]", " ", blob)


def match_soxs_ticker(
    ticker: str,
    event_key: str = "",
    title: str = "",
    summary: str = "",
    slug: str = "",
    entities: Optional[Sequence[str]] = None,
) -> bool:
    """Return True if ticker (or alias) appears in any text field."""
    aliases = SOXS_TICKER_ALIASES.get(ticker, [ticker])
    haystack = _normalize_text(event_key, title, summary, slug, " ".join(entities or []))
    haystack_compact = haystack.replace(" ", "_")
    for alias in aliases:
        alias_up = alias.upper()
        if alias_up in haystack_compact.split():
            return True
        if f"_{alias_up}_" in f"_{haystack_compact}_":
            return True
        if alias_up in haystack_compact:
            return True
    return False


def resolve_matched_tickers(
    tickers: Iterable[str],
    event_key: str = "",
    title: str = "",
    summary: str = "",
    slug: str = "",
    entities: Optional[Sequence[str]] = None,
) -> List[str]:
    return [
        t for t in tickers
        if match_soxs_ticker(t, event_key, title, summary, slug, entities)
    ]


def infer_signal_from_text(
    text: str,
    bearish_re: re.Pattern,
    bullish_re: re.Pattern,
) -> Optional[int]:
    if not text:
        return None
    bear = bool(bearish_re.search(text))
    bull = bool(bullish_re.search(text))
    if bear and not bull:
        return -1
    if bull and not bear:
        return 1
    return None


def enrich_soxs_signals(
    entities: Optional[List[str]],
    slug: Optional[str],
    title: str,
    summary: str,
    capex_signal: Optional[int],
    guidance_signal: Optional[int],
) -> Tuple[Optional[int], Optional[int]]:
    """
    Post-process AI signals: infer from text when AI returned null but content matches.
    Does not override explicit non-null AI values.
    """
    blob = f"{title} {summary}"
    event_key = slug or ""

    capex_matches = resolve_matched_tickers(_CAPEX_TICKERS, event_key, title, summary, slug or "", entities)
    guidance_matches = resolve_matched_tickers(_GUIDANCE_TICKERS, event_key, title, summary, slug or "", entities)

    if capex_signal is None and capex_matches and _CAPEX_KEYWORDS.search(blob):
        capex_signal = infer_signal_from_text(blob, _BEARISH_CAPEX, _BULLISH_CAPEX)

    if guidance_signal is None and guidance_matches and _GUIDANCE_KEYWORDS.search(blob):
        guidance_signal = infer_signal_from_text(blob, _BEARISH_GUIDANCE, _BULLISH_GUIDANCE)

    return capex_signal, guidance_signal


def match_prediction_to_ticker(
    ticker: str,
    event_key: str,
    title: str = "",
    summary: str = "",
    slug: str = "",
) -> bool:
    """Match a DB prediction row to a SOXS ticker bucket."""
    return match_soxs_ticker(ticker, event_key, title, summary, slug)


def divergence_weight(row: dict) -> int:
    """Score divergence event severity for 10-point scale."""
    w = 0
    if row.get("guidance_signal") == -1:
        w += config.SOXS_DIVERGENCE_GUIDANCE_WEIGHT
    if row.get("capex_signal") == -1:
        w += config.SOXS_DIVERGENCE_CAPEX_WEIGHT
    score = float(row.get("score") or 0)
    target = (row.get("target_asset") or "").lower()
    if (
        score >= config.SOXS_DIVERGENCE_MIN_SCORE
        and target in config.SOXS_DIVERGENCE_TARGET_ASSETS
        and row.get("is_correct") == 0
        and row.get("resolved", 0) >= 1
    ):
        w += config.SOXS_DIVERGENCE_NEWS_WEIGHT
    return w
