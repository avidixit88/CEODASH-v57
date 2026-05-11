"""Recency-aware sponsor evidence and catalyst intelligence layer.

ClinicalTrials.gov tells us who is active in the battlefield. This engine is an
*enrichment* layer: it checks available sponsor/news handles for external data,
conference, readout, and safety language, then suppresses stale catalyst noise.

Design principles:
- discovered sponsors remain the source of truth for who gets considered;
- ticker/news mappings are optional enrichment handles, not discovery gates;
- stale conference items such as "AACR 2024" should not be narrated as active
  2026 catalysts;
- every run returns an audit object so the dashboard can show coverage quality.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from email.utils import parsedate_to_datetime
import re
import time
from typing import Any, Iterable, Protocol
from urllib.parse import quote_plus, urljoin, urlparse
from urllib.request import Request, urlopen
import html
import xml.etree.ElementTree as ET


class DiscoveredSponsorLike(Protocol):
    sponsor_name: str
    normalized_name: str
    matched_lanes: tuple[str, ...]
    program_terms: tuple[str, ...]
    relevance_score: int
    evidence_queries: tuple[str, ...]

try:  # optional in tests/fallbacks
    import yfinance as yf
except Exception:  # pragma: no cover - environment-specific
    yf = None  # type: ignore[assignment]

from config.sponsor_evidence_sources import (
    MAX_NEWS_ITEMS_PER_TICKER,
    MAX_SPONSORS_PER_RUN,
    SPONSOR_EVIDENCE_LOOKUP,
    SponsorEvidenceSource,
)


RESULT_TERMS = (
    "orr", "objective response", "overall response", "response rate",
    "pfs", "progression-free", "duration of response", "dor",
    "overall survival", " os ", "complete response", "partial response",
    "phase 2 data", "phase 3 data", "clinical data", "updated data",
)
SAFETY_TERMS = (
    "safety", "tolerability", "adverse event", "toxicity", "grade 3",
    "discontinuation", "dose limiting", "recommended phase 2", "rp2d",
)
DATA_TIMING_TERMS = (
    "asco", "aacr", "esmo", "sitc", "sabcs", "present", "presentation", "abstract",
    "poster", "oral", "data", "readout", "topline", "interim", "updated results",
    "unveil", "late-breaking", "plenary", "investor day", "conference call",
)
CLINICAL_CONTEXT_TERMS = (
    "ovarian", "cdh6", "cadherin 6", "b7-h4", "b7h4", "vtcn1",
    "antibody-drug conjugate", "antibody drug conjugate", " adc ", "platinum-resistant",
    "proc", "gynecologic", "gynecological", "endometrial", "solid tumor",
)

CONFERENCE_TERMS = ("asco", "aacr", "esmo", "sitc", "sabcs")
CURRENT_YEAR = datetime.now(UTC).year

# Performance-bounded two-speed evidence screening. The fast pass reads only
# lightweight title/date/source metadata from a public news RSS route, then only
# promotes likely data/catalyst hits into the evidence model. This prevents the
# app from deep-parsing hundreds of sponsors while still avoiding the old
# 11/559 bottleneck.
#
# v0.9.41 adjustment: the previous 90-sponsor cap could silently skip relevant
# lower-ranked sponsors. We still keep the dashboard bounded, but screen a wider
# ranked universe with shorter per-source timeouts and record high-priority
# unscreened entities explicitly in the audit.
MAX_FAST_SCREEN_SPONSORS = 260
MAX_FAST_SCREEN_ITEMS_PER_SPONSOR = 5
MAX_FAST_SCREEN_SECONDS = 20.0
FAST_SCREEN_TIMEOUT_SECONDS = 0.85
MAX_PROMOTED_SCREEN_ITEMS = 90

# Generic IR/newsroom discovery. This is not a hardcoded sponsor-page list.
# It uses sponsor names to find likely IR/news-release pages, lightly parses
# recent release titles/dates, and lets the same evidence classifier decide
# freshness, stage, and catalyst relevance.
MAX_IR_SCREEN_SPONSORS = 42
MAX_IR_CANDIDATE_URLS = 3
MAX_IR_LINKS_PER_PAGE = 10
IR_SCREEN_TIMEOUT_SECONDS = 1.2

# Publication freshness and catalyst timing are separate concepts. A recent
# press release that says “will present data at ASCO 2026” should remain active
# even if it was published months before the event. A recent article recapping
# an AACR 2024 poster should not be treated as an active catalyst.
RECENT_PUBLICATION_DAYS = 180
AGING_PUBLICATION_DAYS = 365

PROMOTION_TERMS = (
    "data", "results", "readout", "topline", "interim", "present", "presentation",
    "poster", "oral", "abstract", "released", "announced", "reported", "clinical",
    "preclinical", "phase 1", "phase 2", "phase 3", "safety", "orr", "pfs", "dor",
    "asco", "aacr", "esmo", "sitc", "sabcs", "conference", "investor day",
)

PRESS_RELEASE_TERMS = (
    "press release", "business wire", "businesswire", "globenewswire",
    "pr newswire", "prnewswire", "investor relations", "newsroom",
)

DATA_STAGE_PATTERNS = (
    ("PHASE3", ("phase 3", "phase iii", "phase3")),
    ("PHASE2", ("phase 2", "phase ii", "phase2")),
    ("PHASE1", ("phase 1", "phase i", "phase1", "first-in-human", "dose escalation", "dose optimization")),
    ("PRECLINICAL", ("preclinical", "nonclinical", "in vivo", "xenograft")),
)

ACTION_PATTERNS = (
    ("RELEASED_DATA", ("released", "reported", "announced", "presented", "demonstrated", "showed", "updated results", "data from")),
    ("PLANNED_PRESENTATION", ("to present", "will present", "presenting", "accepted abstract", "poster presentation", "oral presentation", "late-breaking", "unveil")),
    ("TOPLINE_READOUT", ("topline", "top-line", "readout")),
    ("INTERIM_DATA", ("interim", "initial data", "preliminary")),
    ("SAFETY_DATA", ("safety", "tolerability", "adverse event")),
)


@dataclass(frozen=True)
class SponsorEvidenceItem:
    sponsor: str
    ticker: str
    title: str
    publisher: str
    published_at: str
    url: str
    evidence_state: str
    matched_terms: tuple[str, ...]
    relevance_score: int
    overlap_terms: tuple[str, ...] = ()
    provenance: str = "media/news article"
    relevance_tier: str = "low"
    evidence_route: str = "ticker_news"
    freshness_state: str = "unknown"
    freshness_score: float = 0.0
    catalyst_year: int | None = None
    catalyst_class: str = "UNCLASSIFIED"
    data_stage: str = "UNKNOWN_STAGE"
    evidence_action: str = "UNKNOWN_ACTION"
    source_quality: str = "medium"
    confidence: str = "limited"
    suppression_reason: str = ""


@dataclass(frozen=True)
class SponsorEvidenceAudit:
    sponsors_discovered: int
    sponsors_searched: int
    mapped_sources_used: int
    unmapped_sponsors: int
    raw_items_seen: int
    candidate_items: int
    accepted_items: int
    stale_items_removed: int
    low_lane_relevance_removed: int
    source_errors: int
    source_routes_checked: tuple[str, ...]
    fast_screen_sponsors: int = 0
    fast_screen_items_seen: int = 0
    promoted_items: int = 0
    deep_parsed_items: int = 0
    screened_sponsor_universe: int = 0
    unscreened_sponsors: int = 0
    unscreened_high_priority: tuple[str, ...] = ()
    focus_company_screen_status: str = "not_configured"
    sponsor_grade_universe: int = 0
    non_sponsor_entities_deprioritized: int = 0
    freshness_model: str = "publication_date_plus_catalyst_timing"


@dataclass(frozen=True)
class SponsorEvidenceSummary:
    source_status: str
    fetched_at_utc: str
    sponsors_checked: tuple[str, ...]
    items: tuple[SponsorEvidenceItem, ...]
    source_errors: tuple[str, ...]
    discovered_sponsors: tuple[str, ...] = ()
    unmapped_sponsors: tuple[str, ...] = ()
    evidence_search_links: tuple[str, ...] = ()
    stale_items: tuple[SponsorEvidenceItem, ...] = ()
    audit: SponsorEvidenceAudit | None = None

    @property
    def result_items(self) -> list[SponsorEvidenceItem]:
        return [i for i in self.items if i.evidence_state == "reported_data_signal"]

    @property
    def timing_items(self) -> list[SponsorEvidenceItem]:
        return [i for i in self.items if i.evidence_state == "future_data_timing_signal"]

    @property
    def clinical_items(self) -> list[SponsorEvidenceItem]:
        return [i for i in self.items if i.evidence_state in {"reported_data_signal", "future_data_timing_signal", "clinical_context_signal"}]

    @property
    def active_catalyst_items(self) -> list[SponsorEvidenceItem]:
        return [i for i in self.items if i.freshness_state in {"upcoming_catalyst", "active_window", "recent"}]


def _norm(text: str) -> str:
    return " ".join((text or "").lower().replace("–", "-").replace("—", "-").split())


def _matches_any(text: str, terms: Iterable[str]) -> list[str]:
    haystack = f" {_norm(text)} "
    out: list[str] = []
    for term in terms:
        t = _norm(term)
        if t and t in haystack:
            out.append(term.strip())
    return out


def _source_for_sponsor(sponsor: str) -> SponsorEvidenceSource | None:
    sponsor_l = _norm(sponsor)
    candidates: list[tuple[int, SponsorEvidenceSource]] = []
    for source in SPONSOR_EVIDENCE_LOOKUP:
        names = (source.sponsor, *source.aliases)
        if any(_norm(name) in sponsor_l or sponsor_l in _norm(name) for name in names):
            candidates.append((source.priority, source))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item[0])[0][1]


def _select_sponsor_sources(sponsors: Iterable[str]) -> list[SponsorEvidenceSource]:
    selected: dict[str, SponsorEvidenceSource] = {}
    for sponsor in sponsors:
        source = _source_for_sponsor(sponsor)
        if source is not None:
            selected[source.sponsor] = source
    return sorted(selected.values(), key=lambda s: s.priority)[:MAX_SPONSORS_PER_RUN]


def _dynamic_sources_for_discovered(discovered_sponsors: Iterable[DiscoveredSponsorLike] | None) -> tuple[list[SponsorEvidenceSource], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    if discovered_sponsors is None:
        return [], (), (), ()

    resolved: dict[str, SponsorEvidenceSource] = {}
    discovered_names: list[str] = []
    unmapped: list[str] = []
    links: list[str] = []

    for sponsor in sorted(discovered_sponsors, key=lambda s: getattr(s, "relevance_score", 0), reverse=True):
        name = getattr(sponsor, "sponsor_name", "") or getattr(sponsor, "normalized_name", "")
        if not name or name in discovered_names:
            continue
        discovered_names.append(name)
        mapped = _source_for_sponsor(name)
        if mapped is not None:
            terms = tuple(dict.fromkeys((*mapped.evidence_terms, *getattr(sponsor, "program_terms", ()))))
            resolved[mapped.sponsor] = SponsorEvidenceSource(
                sponsor=mapped.sponsor,
                tickers=mapped.tickers,
                aliases=tuple(dict.fromkeys((*mapped.aliases, name))),
                priority=mapped.priority,
                evidence_terms=terms,
            )
        else:
            unmapped.append(name)
            for link in getattr(sponsor, "evidence_queries", ())[:4]:
                if link not in links:
                    links.append(link)

    return sorted(resolved.values(), key=lambda s: s.priority), tuple(discovered_names), tuple(unmapped), tuple(links[:36])


def _news_items_for_ticker(ticker: str) -> list[dict[str, Any]]:
    if yf is None:
        raise RuntimeError("yfinance is not available")
    raw = yf.Ticker(ticker).news or []  # type: ignore[union-attr]
    return raw[:MAX_NEWS_ITEMS_PER_TICKER]


def _extract_title(item: dict[str, Any]) -> str:
    return str(item.get("title") or item.get("content", {}).get("title") or "").strip()


def _extract_publisher(item: dict[str, Any]) -> str:
    return str(item.get("publisher") or item.get("content", {}).get("provider", {}).get("displayName") or "").strip()


def _extract_url(item: dict[str, Any]) -> str:
    return str(item.get("link") or item.get("content", {}).get("canonicalUrl", {}).get("url") or "").strip()


def _extract_published_at(item: dict[str, Any]) -> str:
    ts = item.get("providerPublishTime") or item.get("content", {}).get("pubDate")
    if isinstance(ts, (int, float)):
        try:
            return datetime.fromtimestamp(ts, UTC).date().isoformat()
        except Exception:
            return ""
    return str(ts or "").strip()[:10]


def _parse_date(value: str) -> date | None:
    if not value:
        return None
    cleaned = str(value).strip()
    try:
        return date.fromisoformat(cleaned[:10])
    except Exception:
        pass
    for fmt, length in (("%Y-%m", 7), ("%Y", 4)):
        try:
            return datetime.strptime(cleaned[:length], fmt).date()
        except Exception:
            pass
    return None


def _conference_year(text: str) -> int | None:
    low = _norm(text)
    if not any(c in low for c in CONFERENCE_TERMS):
        return None
    years = [int(y) for y in re.findall(r"\b(20[2-4][0-9])\b", text)]
    if not years:
        return None
    # Prefer years close to current or future over old years in citations/URLs.
    years_sorted = sorted(years, key=lambda y: (abs(y - CURRENT_YEAR), -y))
    return years_sorted[0]


def _classify_provenance(title: str, publisher: str, url: str) -> str:
    text = _norm(" ".join([title, publisher, url]))
    if any(term in text for term in ("businesswire", "prnewswire", "globenewswire", "investor relations", "press release", "newsroom", "ir/newsroom", "/news-releases", "/press-releases")):
        return "press release / IR"
    if any(term in text for term in ("asco", "aacr", "esmo", "sitc", "sabcs", "abstract", "oral presentation", "poster", "late-breaking")):
        return "conference / abstract"
    if any(term in text for term in ("sec", "10-k", "10-q", "8-k", "annual report")):
        return "filing / investor update"
    if any(term in text for term in ("pubmed", "journal", "nejm", "lancet", "jama")):
        return "publication / journal"
    return "media/news article"


def _source_quality(provenance: str, publisher: str, url: str) -> str:
    text = _norm(" ".join([publisher, url]))
    if provenance in {"press release / IR", "conference / abstract", "filing / investor update", "publication / journal"}:
        return "high"
    if any(t in text for t in ("businesswire", "globenewswire", "prnewswire", "sec.gov", "asco", "aacr", "esmo")):
        return "high"
    if any(t in text for t in ("yahoo", "benzinga", "zacks", "seeking alpha")):
        return "medium"
    return "medium"


def _publication_freshness(published_at: str) -> tuple[str, float, str, int | None]:
    """Classify evidence freshness by publication date only.

    Publication date answers: "Is this source itself recent enough to use?"
    It does not answer whether a conference or data event is current. That is
    handled separately by _catalyst_timing below.
    """
    pub_date = _parse_date(published_at)
    today = datetime.now(UTC).date()
    if pub_date is None:
        return "unknown_publication_date", 0.35, "no reliable publication date", None
    age = (today - pub_date).days
    if age < 0:
        return "future_dated_publication", 0.95, "source is dated in the future", age
    if age <= 30:
        return "recent", 1.0, "published within 30 days", age
    if age <= 90:
        return "recent", 0.85, "published within 90 days", age
    if age <= RECENT_PUBLICATION_DAYS:
        return "recent", 0.65, f"published within {RECENT_PUBLICATION_DAYS} days", age
    if age <= AGING_PUBLICATION_DAYS:
        return "aging", 0.35, f"published within {AGING_PUBLICATION_DAYS} days", age
    return "stale_publication", 0.08, f"publication older than {AGING_PUBLICATION_DAYS} days", age


def _catalyst_timing(text: str) -> tuple[str, float, int | None, str]:
    """Classify catalyst timing/event year separately from source freshness."""
    year = _conference_year(text)
    if year is None:
        return "no_explicit_event_year", 1.0, None, "no explicit conference/event year"
    if year < CURRENT_YEAR:
        return "expired_event_year", 0.15, year, f"conference/event year {year} is older than current year {CURRENT_YEAR}"
    if year == CURRENT_YEAR:
        return "current_event_year", 1.0, year, f"conference/event year {year} is current"
    return "future_event_year", 0.95, year, f"conference/event year {year} is future"


def _freshness(published_at: str, text: str) -> tuple[str, float, int | None, str]:
    """Return final freshness after combining publication and catalyst timing.

    Suppression is driven by stale publication age or expired event timing, but
    the two are not collapsed into one field internally. This prevents a recent
    source discussing a future catalyst from being mistakenly stale, and prevents
    an old conference year from masquerading as active just because a headline is
    semantically relevant.
    """
    pub_state, pub_score, pub_reason, _age = _publication_freshness(published_at)
    timing_state, timing_score, catalyst_year, timing_reason = _catalyst_timing(text)

    if pub_state == "stale_publication":
        return "stale_publication", pub_score, catalyst_year, pub_reason
    if timing_state == "expired_event_year":
        return "stale_historical_event", min(pub_score, timing_score), catalyst_year, timing_reason
    if timing_state in {"current_event_year", "future_event_year"}:
        if pub_state in {"recent", "future_dated_publication", "unknown_publication_date"}:
            return "upcoming_catalyst", min(pub_score, timing_score), catalyst_year, f"{pub_reason}; {timing_reason}"
        return "aging_upcoming_catalyst", min(pub_score, timing_score), catalyst_year, f"{pub_reason}; {timing_reason}"
    return pub_state, pub_score, catalyst_year, pub_reason

def _data_stage(text: str) -> str:
    low = _norm(text)
    for label, patterns in DATA_STAGE_PATTERNS:
        if any(p in low for p in patterns):
            return label
    return "UNKNOWN_STAGE"


def _evidence_action(text: str) -> str:
    low = _norm(text)
    for label, patterns in ACTION_PATTERNS:
        if any(p in low for p in patterns):
            return label
    return "UNKNOWN_ACTION"


def _fast_screen_queries_for_sponsor(source: SponsorEvidenceSource) -> tuple[str, ...]:
    """Build compact evidence-discovery queries for the breadth pass.

    The old query combined sponsor aliases, program terms, and many evidence
    words into one very long RSS query. That was too brittle: small sponsors
    often returned nothing, so the fast pass promoted zero leads. We now use a
    small query cascade: sponsor + evidence action terms first, then sponsor +
    program/stage terms. Conference names are not the primary strategy; if a
    press release mentions ASCO/AACR/etc., the classifier extracts that later.
    """
    names = [name for name in dict.fromkeys((source.sponsor, *source.aliases)) if name]
    compact_names = names[:3]
    sponsor_terms = ' OR '.join(f'"{name}"' for name in compact_names) or f'"{source.sponsor}"'
    program_terms = ' OR '.join(dict.fromkeys(source.evidence_terms[:8])) or 'ADC OR oncology'
    evidence_terms = ' OR '.join((
        'data', 'results', 'readout', 'topline', 'interim', 'present',
        'presentation', 'poster', 'abstract', 'phase', 'preclinical',
        'safety', 'press release'
    ))
    return (
        f'({sponsor_terms}) ({evidence_terms})',
        f'({sponsor_terms}) ({program_terms}) (data OR results OR presentation OR phase OR preclinical)',
    )


def _fast_screen_query_for_sponsor(source: SponsorEvidenceSource) -> str:
    # Backward-compatible helper retained for tests/older imports.
    return _fast_screen_queries_for_sponsor(source)[0]


def _is_likely_ir_url(url: str) -> bool:
    low = _norm(url)
    return any(token in low for token in (
        "investor", "ir.", "news", "newsroom", "press", "release", "media", "events"
    )) and not any(bad in low for bad in (
        "linkedin.com", "facebook.com", "twitter.com", "x.com", "wikipedia.org",
        "clinicaltrials.gov", "sec.gov/Archives", "youtube.com"
    ))


def _web_rss_items(query_text: str, max_items: int = 8) -> list[dict[str, str]]:
    """Search the open web through a lightweight RSS endpoint.

    This is used only as a source-discovery mechanism for IR/newsroom pages,
    not as a sponsor whitelist. If the route throttles or fails, the dashboard
    continues with the other evidence routes.
    """
    query = quote_plus(query_text)
    url = f"https://www.bing.com/search?q={query}&format=rss"
    request = Request(url, headers={"User-Agent": "Mozilla/5.0 BuildWell Intelligence/0.9"})
    with urlopen(request, timeout=FAST_SCREEN_TIMEOUT_SECONDS) as response:  # nosec - public search metadata
        payload = response.read(512_000)
    root = ET.fromstring(payload)
    out: list[dict[str, str]] = []
    for node in root.findall(".//item")[:max_items]:
        out.append({
            "title": (node.findtext("title") or "").strip(),
            "link": (node.findtext("link") or "").strip(),
            "description": (node.findtext("description") or "").strip(),
            "pubDate": _rss_date_to_iso(node.findtext("pubDate") or ""),
        })
    return out


def _discover_ir_candidate_urls(source: SponsorEvidenceSource) -> list[str]:
    """Find likely sponsor IR/newsroom pages without hardcoding pages.

    The search queries are sponsor-name based. Candidate URLs are retained only
    when their path/domain looks like IR, newsroom, press-release, or events
    content. The release parser then extracts specific recent titles from those
    pages and the normal classifier decides whether any item is useful.
    """
    names = [n for n in dict.fromkeys((source.sponsor, *source.aliases)) if n]
    primary = names[0] if names else source.sponsor
    queries = (
        f'"{primary}" investor relations news releases',
        f'"{primary}" press releases data phase presentation',
        f'"{primary}" newsroom clinical data presentation',
    )
    urls: list[str] = []
    for query in queries:
        try:
            for result in _web_rss_items(query, max_items=8):
                link = result.get("link", "")
                title = result.get("title", "")
                blob = f"{title} {link}"
                if link and _is_likely_ir_url(blob) and link not in urls:
                    urls.append(link)
                if len(urls) >= MAX_IR_CANDIDATE_URLS:
                    return urls
        except Exception:
            continue
    return urls


def _strip_html(raw: str) -> str:
    text = re.sub(r"<script[\s\S]*?</script>", " ", raw, flags=re.I)
    text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    return html.unescape(re.sub(r"\s+", " ", text)).strip()


def _extract_date_near(html_text: str, start: int, end: int) -> str:
    window = html.unescape(html_text[max(0, start - 500): min(len(html_text), end + 500)])
    # ISO-ish first, then common press-release date styles.
    match = re.search(r"\b(20[2-4][0-9]-[01][0-9]-[0-3][0-9])\b", window)
    if match:
        return match.group(1)
    match = re.search(r"\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?\s+([0-3]?\d),\s+(20[2-4][0-9])\b", window, flags=re.I)
    if match:
        month_name, day, year = match.groups()
        try:
            return datetime.strptime(f"{month_name} {day} {year}", "%B %d %Y").date().isoformat()
        except Exception:
            try:
                return datetime.strptime(f"{month_name} {day} {year}", "%b %d %Y").date().isoformat()
            except Exception:
                return ""
    return ""


def _release_items_from_ir_page(source: SponsorEvidenceSource, page_url: str) -> list[dict[str, Any]]:
    """Lightly parse a sponsor IR/newsroom page into release candidates.

    The parser is intentionally generic: extract links/titles from the page,
    keep release-like or data/catalyst-like titles, infer nearby dates, and pass
    the result into the same evidence classifier used elsewhere.
    """
    request = Request(page_url, headers={"User-Agent": "Mozilla/5.0 BuildWell Intelligence/0.9"})
    with urlopen(request, timeout=IR_SCREEN_TIMEOUT_SECONDS) as response:  # nosec - public IR/newsroom page
        raw_bytes = response.read(900_000)
    raw = raw_bytes.decode("utf-8", errors="ignore")
    base = page_url
    items: list[dict[str, Any]] = []
    seen: set[str] = set()

    # Link text candidates catch most IR platforms. We do not require a known
    # conference name; generic data/release/stage language is enough to promote.
    link_pattern = re.compile(r'<a\b[^>]*href=["\']([^"\']+)["\'][^>]*>([\s\S]*?)</a>', flags=re.I)
    for match in link_pattern.finditer(raw):
        href, inner = match.groups()
        title = _strip_html(inner)
        if not title or len(title) < 18:
            continue
        if len(title) > 260:
            title = title[:260].strip()
        text = f"{source.sponsor} {title} {href}"
        if not (_matches_any(text, PROMOTION_TERMS) or _matches_any(text, DATA_TIMING_TERMS) or _matches_any(text, RESULT_TERMS)):
            continue
        full_url = urljoin(base, html.unescape(href))
        key = _norm(f"{title} {full_url}")
        if key in seen:
            continue
        seen.add(key)
        items.append({
            "title": title,
            "publisher": f"{source.sponsor} IR/newsroom",
            "providerPublishTime": _extract_date_near(raw, match.start(), match.end()),
            "link": full_url,
            "route": "ir_newsroom_screen",
        })
        if len(items) >= MAX_IR_LINKS_PER_PAGE:
            break

    # Fallback: some pages render cards without useful anchor text. Use plain
    # text sentence fragments containing evidence terms, still sponsor-scoped.
    if not items:
        plain = _strip_html(raw)
        for sentence in re.split(r"(?<=[.!?])\s+", plain):
            if len(sentence) < 35 or len(sentence) > 260:
                continue
            text = f"{source.sponsor} {sentence}"
            if _matches_any(text, PROMOTION_TERMS) and (_matches_any(text, source.evidence_terms) or _matches_any(text, CLINICAL_CONTEXT_TERMS)):
                items.append({
                    "title": sentence.strip(),
                    "publisher": f"{source.sponsor} IR/newsroom",
                    "providerPublishTime": _extract_date_near(raw, 0, min(len(raw), 2000)),
                    "link": page_url,
                    "route": "ir_newsroom_screen",
                })
                if len(items) >= MAX_IR_LINKS_PER_PAGE:
                    break
    return items


def _ir_newsroom_screen_items(source: SponsorEvidenceSource) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen_links: set[str] = set()
    for page_url in _discover_ir_candidate_urls(source):
        try:
            for item in _release_items_from_ir_page(source, page_url):
                link = _extract_url(item)
                title = _extract_title(item)
                if not title or link in seen_links:
                    continue
                seen_links.add(link)
                out.append(item)
                if len(out) >= MAX_FAST_SCREEN_ITEMS_PER_SPONSOR:
                    return out
        except Exception:
            continue
    return out


def _rss_date_to_iso(value: str) -> str:
    if not value:
        return ""
    try:
        return parsedate_to_datetime(value).astimezone(UTC).date().isoformat()
    except Exception:
        return str(value).strip()[:10]


def _fast_screen_news_items(source: SponsorEvidenceSource) -> list[dict[str, Any]]:
    """Lightweight, metadata-only news scan for sponsor evidence leads.

    This is the generic news/RSS breadth route. IR/newsroom screening is handled
    by _ir_newsroom_screen_items and merged in the fast pass for sponsor-grade
    entities. Accepted/promising items are then classified by the same recency
    and lane-specific logic as ticker news.
    """
    items: list[dict[str, Any]] = []
    seen_links: set[str] = set()
    for query_text in _fast_screen_queries_for_sponsor(source):
        query = quote_plus(query_text)
        url = f"https://news.search.yahoo.com/rss?p={query}"
        request = Request(url, headers={"User-Agent": "Mozilla/5.0 BuildWell Intelligence/0.9"})
        with urlopen(request, timeout=FAST_SCREEN_TIMEOUT_SECONDS) as response:  # nosec - read-only public RSS route
            payload = response.read(512_000)
        root = ET.fromstring(payload)
        for node in root.findall(".//item")[:MAX_FAST_SCREEN_ITEMS_PER_SPONSOR]:
            title = (node.findtext("title") or "").strip()
            link = (node.findtext("link") or "").strip()
            if link in seen_links:
                continue
            seen_links.add(link)
            pub_date = _rss_date_to_iso(node.findtext("pubDate") or "")
            source_name = "Yahoo News RSS"
            source_node = node.find("source")
            if source_node is not None and source_node.text:
                source_name = source_node.text.strip()
            if title:
                items.append({
                    "title": title,
                    "publisher": source_name,
                    "providerPublishTime": pub_date,
                    "link": link,
                    "route": "fast_news_screen",
                })
            if len(items) >= MAX_FAST_SCREEN_ITEMS_PER_SPONSOR:
                break
        if len(items) >= MAX_FAST_SCREEN_ITEMS_PER_SPONSOR:
            break
    return items


def _looks_promising_for_promotion(source: SponsorEvidenceSource, raw_item: dict[str, Any]) -> bool:
    text = " ".join([_extract_title(raw_item), _extract_publisher(raw_item), _extract_url(raw_item)])
    promo_terms = _matches_any(text, PROMOTION_TERMS)
    if not promo_terms:
        return False
    # The fast-pass query itself is sponsor-scoped, but search/RSS can still
    # return noisy market articles. Prefer explicit alias overlap; for small
    # sponsors, allow strong press-release/data-stage language even if the feed
    # title omits a formal alias because the URL/source may carry the sponsor.
    alias_terms = _matches_any(text, (source.sponsor, *source.aliases))
    program_terms = _matches_any(text, source.evidence_terms)
    context_terms = _matches_any(text, CLINICAL_CONTEXT_TERMS)
    press_terms = _matches_any(text, PRESS_RELEASE_TERMS)
    stage_terms = _matches_any(text, ("phase 1", "phase 2", "phase 3", "preclinical", "clinical data", "topline", "readout"))
    if alias_terms and (program_terms or context_terms or press_terms or stage_terms):
        return True
    if alias_terms and len(promo_terms) >= 2 and ("data" in [p.lower() for p in promo_terms] or stage_terms):
        return True
    if press_terms and stage_terms and (program_terms or context_terms):
        return True
    return False


def _strategic_source_from_discovered(sponsor: DiscoveredSponsorLike) -> SponsorEvidenceSource:
    name = getattr(sponsor, "sponsor_name", "") or getattr(sponsor, "normalized_name", "")
    aliases = tuple(dict.fromkeys((name, getattr(sponsor, "normalized_name", ""))))
    terms = tuple(dict.fromkeys((*getattr(sponsor, "program_terms", ()), *getattr(sponsor, "matched_lanes", ()))))
    return SponsorEvidenceSource(
        sponsor=name,
        tickers=(),
        aliases=aliases,
        priority=max(10, 100 - int(getattr(sponsor, "relevance_score", 0))),
        evidence_terms=terms or ("ADC", "oncology", "ovarian"),
    )


def _entity_grade(source: SponsorEvidenceSource) -> str:
    text = _norm(" ".join((source.sponsor, *source.aliases)))
    institutional = ("hospital", "university", "universit", "institute", "institut", "center", "centre",
                     "clinic", "ziekenhuis", "trial group", "research group", "network", "foundation", "fundación",
                     "national cancer institute", "nci", "alliance", "swog", "ecog", "nrgg", "hospital")
    company_like = ("therapeutics", "biopharma", "biotech", "oncology", "pharmaceutical", "pharma",
                    "medicines", "bioscience", "biomedical", "bio", "laboratories", "labs", "inc",
                    "ltd", "limited", "corp", "corporation", "plc", "se", "ag", "gmbh")
    if source.tickers:
        return "public_company"
    if any(t in text for t in company_like):
        return "sponsor_grade_company"
    if any(t in text for t in institutional):
        return "institutional_or_consortium"
    # Names with comma-heavy legal/geographic fragments are often site entities.
    if len(text.split()) >= 5 and not any(t in text for t in ("biotech", "pharma", "therapeutics", "oncology")):
        return "low_signal_entity"
    return "possible_company"


def _source_rank_score(src: SponsorEvidenceSource) -> tuple[int, int, str]:
    text = _norm(" ".join((src.sponsor, *src.aliases, *src.evidence_terms)))
    grade = _entity_grade(src)
    grade_weight = {
        "public_company": -35,
        "sponsor_grade_company": -22,
        "possible_company": -8,
        "institutional_or_consortium": 30,
        "low_signal_entity": 45,
    }.get(grade, 0)
    biotech_bonus = -10 if any(t in text for t in ("therapeutics", "biopharma", "biotech", "oncology", "pharmaceutical", "pharma", "medicines", "bioscience", "biomedical")) else 0
    lane_bonus = -10 if any(t in text for t in ("cdh6", "b7-h4", "b7h4", "vtcn1", "ovarian", "gynecologic", "adc", "antibody-drug", "sim0505")) else 0
    return (src.priority + grade_weight + biotech_bonus + lane_bonus, 0 if src.tickers else 1, src.sponsor.lower())


def _source_universe(
    legacy_sources: list[SponsorEvidenceSource],
    dynamic_sources: list[SponsorEvidenceSource],
    discovered_sponsors: Iterable[DiscoveredSponsorLike] | None,
) -> list[SponsorEvidenceSource]:
    by_name: dict[str, SponsorEvidenceSource] = {}
    for src in [*legacy_sources, *dynamic_sources]:
        by_name[_norm(src.sponsor)] = src
    if discovered_sponsors is not None:
        for sponsor in discovered_sponsors:
            name = getattr(sponsor, "sponsor_name", "") or getattr(sponsor, "normalized_name", "")
            if not name:
                continue
            key = _norm(name)
            if key not in by_name:
                by_name[key] = _strategic_source_from_discovered(sponsor)
    # Rank by strategic evidence utility, not just ticker availability. Sponsor-grade
    # company entities are screened before hospitals/consortia/site records so
    # the runtime budget is spent on likely press-release owners.
    return sorted(by_name.values(), key=_source_rank_score)


def _catalyst_class(result_terms: list[str], safety_terms: list[str], timing_terms: list[str], provenance: str, text: str) -> str:
    low = _norm(text)
    if result_terms or safety_terms:
        if "phase 3" in low or "topline" in low:
            return "PHASE3_TOPLINE_OR_SAFETY"
        if "phase 2" in low:
            return "PHASE2_DATA"
        return "CLINICAL_DATA_OR_SAFETY"
    if timing_terms:
        if "oral" in low or "late-breaking" in low or "plenary" in low:
            return "CONFERENCE_ORAL_OR_LATE_BREAKER"
        if "poster" in low or "abstract" in low:
            return "CONFERENCE_ABSTRACT_OR_POSTER"
        if "preclinical" in low:
            return "PRECLINICAL_CONFERENCE_SIGNAL"
        if provenance == "press release / IR":
            return "IR_DATA_TIMING_SIGNAL"
        return "DATA_TIMING_SIGNAL"
    return "CLINICAL_CONTEXT"


def _confidence(tier: str, freshness_state: str, provenance: str, source_quality: str, overlap_terms: tuple[str, ...]) -> str:
    if freshness_state == "stale_historical_event":
        return "stale"
    if tier == "high" and source_quality == "high" and len(overlap_terms) >= 1:
        return "high"
    if tier in {"high", "moderate"} and freshness_state in {"recent", "upcoming_catalyst", "active_window"}:
        return "moderate"
    if provenance == "media/news article":
        return "limited until reconciled"
    return "limited"


def _classify_item(source: SponsorEvidenceSource, ticker: str, item: dict[str, Any]) -> SponsorEvidenceItem | None:
    title = _extract_title(item)
    if not title:
        return None
    publisher = _extract_publisher(item)
    url = _extract_url(item)
    published_at = _extract_published_at(item)
    text = " ".join([title, publisher, url])

    result_terms = _matches_any(text, RESULT_TERMS)
    safety_terms = _matches_any(text, SAFETY_TERMS)
    timing_terms = _matches_any(text, DATA_TIMING_TERMS)
    context_terms = _matches_any(text, CLINICAL_CONTEXT_TERMS)
    sponsor_program_terms = _matches_any(text, source.evidence_terms)
    overlap_terms = tuple(dict.fromkeys(context_terms + sponsor_program_terms))

    if not overlap_terms:
        return SponsorEvidenceItem(
            sponsor=source.sponsor, ticker=ticker, title=title, publisher=publisher,
            published_at=published_at, url=url, evidence_state="rejected_low_lane_relevance",
            matched_terms=tuple(dict.fromkeys(result_terms + safety_terms + timing_terms)),
            relevance_score=0, suppression_reason="no monitored-lane or sponsor-program overlap",
        )
    if not any([result_terms, safety_terms, timing_terms, context_terms, sponsor_program_terms]):
        return None

    provenance = _classify_provenance(title, publisher, url)
    source_quality = _source_quality(provenance, publisher, url)
    freshness_state, freshness_score, catalyst_year, freshness_reason = _freshness(published_at, text)
    catalyst_class = _catalyst_class(result_terms, safety_terms, timing_terms, provenance, text)
    data_stage = _data_stage(text)
    evidence_action = _evidence_action(text)

    base_relevance = (
        len(result_terms) * 5
        + len(safety_terms) * 4
        + len(timing_terms) * 3
        + len(context_terms) * 2
        + len(sponsor_program_terms) * 3
    )
    if source_quality == "high":
        base_relevance += 4
    elif provenance == "media/news article":
        base_relevance -= 1
    if catalyst_class in {"PHASE3_TOPLINE_OR_SAFETY", "PHASE2_DATA", "CLINICAL_DATA_OR_SAFETY"}:
        base_relevance += 4
    if data_stage in {"PHASE1", "PHASE2", "PHASE3", "PRECLINICAL"}:
        base_relevance += 3
    if evidence_action in {"RELEASED_DATA", "PLANNED_PRESENTATION", "TOPLINE_READOUT", "INTERIM_DATA", "SAFETY_DATA"}:
        base_relevance += 3
    elif catalyst_class in {"CONFERENCE_ORAL_OR_LATE_BREAKER", "CONFERENCE_ABSTRACT_OR_POSTER", "IR_DATA_TIMING_SIGNAL"}:
        base_relevance += 3

    relevance = max(0, int(round(base_relevance * max(0.05, freshness_score))))

    if (result_terms or safety_terms) and len(overlap_terms) >= 1:
        state = "reported_data_signal"
    elif timing_terms and len(overlap_terms) >= 1:
        state = "future_data_timing_signal"
    else:
        state = "clinical_context_signal"

    if freshness_state in {"stale_historical_event", "stale_publication"}:
        state = "stale_historical_event"

    if relevance >= 15:
        tier = "high"
    elif relevance >= 8:
        tier = "moderate"
    else:
        tier = "low"

    terms = tuple(dict.fromkeys(result_terms + safety_terms + timing_terms + context_terms + sponsor_program_terms))
    confidence = _confidence(tier, freshness_state, provenance, source_quality, overlap_terms)
    return SponsorEvidenceItem(
        sponsor=source.sponsor,
        ticker=ticker,
        title=title,
        publisher=publisher,
        published_at=published_at,
        url=url,
        evidence_state=state,
        matched_terms=terms,
        relevance_score=relevance,
        overlap_terms=overlap_terms,
        provenance=provenance,
        relevance_tier=tier,
        freshness_state=freshness_state,
        freshness_score=freshness_score,
        catalyst_year=catalyst_year,
        catalyst_class=catalyst_class,
        data_stage=data_stage,
        evidence_action=evidence_action,
        source_quality=source_quality,
        confidence=confidence,
        suppression_reason=freshness_reason if state == "stale_historical_event" else "",
        evidence_route=str(item.get("route") or "ticker_news"),
    )


def _keep_executive_item(item: SponsorEvidenceItem) -> bool:
    if item.evidence_state in {"stale_historical_event", "rejected_low_lane_relevance"}:
        return False
    if item.freshness_state in {"stale_historical_event", "stale_publication"}:
        return False
    if item.relevance_tier == "low" and item.provenance == "media/news article" and item.evidence_state == "clinical_context_signal":
        return False
    if item.relevance_score < 5:
        return False
    return True


def build_sponsor_evidence_summary(
    sponsors: Iterable[str],
    discovered_sponsors: Iterable[DiscoveredSponsorLike] | None = None,
) -> SponsorEvidenceSummary:
    fetched_at = datetime.now(UTC).isoformat(timespec="seconds")

    dynamic_sources, discovered_names, unmapped_sponsors, search_links = _dynamic_sources_for_discovered(discovered_sponsors)
    legacy_sources = _select_sponsor_sources(sponsors)

    universe = _source_universe(legacy_sources, dynamic_sources, discovered_sponsors)

    # Deep mapped/ticker pass: keep this small and deterministic for dashboard speed.
    sources = [s for s in universe if s.tickers][:MAX_SPONSORS_PER_RUN]

    checked: list[str] = []
    accepted: list[SponsorEvidenceItem] = []
    stale: list[SponsorEvidenceItem] = []
    errors: list[str] = []
    raw_seen = 0
    candidates = 0
    low_lane_removed = 0
    promoted_items = 0
    fast_screen_items_seen = 0
    deep_parsed_items = 0
    fast_screened_names: list[str] = []
    promoted_raw_items: list[tuple[SponsorEvidenceSource, dict[str, Any]]] = []

    for source in sources:
        checked.append(source.sponsor)
        for ticker in source.tickers:
            try:
                raw_news = _news_items_for_ticker(ticker)
                raw_seen += len(raw_news)
                for raw_item in raw_news:
                    item = _classify_item(source, ticker, raw_item)
                    if item is None:
                        continue
                    deep_parsed_items += 1
                    if item.evidence_state == "rejected_low_lane_relevance":
                        low_lane_removed += 1
                        continue
                    candidates += 1
                    if item.evidence_state == "stale_historical_event":
                        stale.append(item)
                    elif _keep_executive_item(item):
                        accepted.append(item)
            except Exception as exc:  # upstream news failure should not break analysis
                errors.append(f"{source.sponsor} / {ticker}: {type(exc).__name__}: {exc}")
            time.sleep(0.03)

    # Fast breadth pass: screen many discovered sponsors through a lightweight
    # title/date RSS route. Only promising hits get promoted into the classifier.
    # The universe has already been ranked by strategic evidence utility.
    started = time.monotonic()
    fast_screen_budget_slice = universe[:MAX_FAST_SCREEN_SPONSORS]
    for source in fast_screen_budget_slice:
        if time.monotonic() - started > MAX_FAST_SCREEN_SECONDS:
            break
        if source.sponsor not in fast_screened_names:
            fast_screened_names.append(source.sponsor)
        try:
            raw_screen_items = _fast_screen_news_items(source)
            # For sponsor-grade companies, add a generic IR/newsroom route. This
            # avoids hardcoded company pages while catching releases that do not
            # appear in finance/news feeds. Keep it bounded by grade and global
            # source rank so runtime remains usable.
            if len(fast_screened_names) <= MAX_IR_SCREEN_SPONSORS and (source.tickers or _entity_grade(source) == "public_company"):
                raw_screen_items = [*raw_screen_items, *_ir_newsroom_screen_items(source)]
            fast_screen_items_seen += len(raw_screen_items)
            for raw_item in raw_screen_items:
                if _looks_promising_for_promotion(source, raw_item):
                    promoted_raw_items.append((source, raw_item))
                    if len(promoted_raw_items) >= MAX_PROMOTED_SCREEN_ITEMS:
                        break
            if len(promoted_raw_items) >= MAX_PROMOTED_SCREEN_ITEMS:
                break
        except Exception as exc:
            # Keep errors compact; fast screening is opportunistic and should not
            # break the core dashboard if public news search throttles.
            if len(errors) < 12:
                errors.append(f"fast screen {source.sponsor}: {type(exc).__name__}: {exc}")
        time.sleep(0.02)

    for source, raw_item in promoted_raw_items:
        promoted_items += 1
        raw_seen += 1
        item = _classify_item(source, "SCREEN", raw_item)
        if item is None:
            continue
        deep_parsed_items += 1
        if item.evidence_state == "rejected_low_lane_relevance":
            low_lane_removed += 1
            continue
        candidates += 1
        if item.evidence_state == "stale_historical_event":
            stale.append(item)
        elif _keep_executive_item(item):
            accepted.append(item)

    # Coverage should reflect both ticker-deep sources and the fast-screened
    # breadth pass, not just the old mapped ticker subset.
    checked = list(dict.fromkeys([*checked, *fast_screened_names]))
    screened_keys = {_norm(name) for name in fast_screened_names}
    unscreened_sources = [src for src in universe if _norm(src.sponsor) not in screened_keys]
    unscreened_high_priority = tuple(src.sponsor for src in unscreened_sources[:12])

    sponsor_grade_sources = [src for src in universe if _entity_grade(src) in {"public_company", "sponsor_grade_company", "possible_company"}]
    non_sponsor_entities_deprioritized = max(0, len(universe) - len(sponsor_grade_sources))

    focus_aliases = ("nextcure", "nextcure inc", "nxtc", "sim0505")
    focus_sources = [src for src in universe if any(alias in _norm(" ".join((src.sponsor, *src.aliases, *src.tickers, *src.evidence_terms))) for alias in focus_aliases)]
    focus_checked = any(src.sponsor in checked or src.sponsor in fast_screened_names for src in focus_sources)
    focus_accepted = any(any(alias in _norm(" ".join((item.sponsor, item.ticker, item.title, *item.overlap_terms))) for alias in focus_aliases) for item in accepted)
    if not focus_sources:
        focus_company_status = "NextCure not present in evidence universe"
    elif focus_accepted:
        focus_company_status = "NextCure screened and active evidence accepted"
    elif focus_checked:
        focus_company_status = "NextCure screened; no active promoted evidence accepted"
    else:
        focus_company_status = "NextCure present but not screened within runtime budget"

    deduped: dict[tuple[str, str], SponsorEvidenceItem] = {}
    for item in accepted:
        key = (_norm(item.title), item.ticker)
        existing = deduped.get(key)
        if existing is None or item.relevance_score > existing.relevance_score:
            deduped[key] = item
    ordered = sorted(deduped.values(), key=lambda i: (i.relevance_score, i.freshness_score, i.published_at), reverse=True)[:12]

    stale_deduped: dict[tuple[str, str], SponsorEvidenceItem] = {}
    for item in stale:
        key = (_norm(item.title), item.ticker)
        if key not in stale_deduped:
            stale_deduped[key] = item
    stale_ordered = tuple(sorted(stale_deduped.values(), key=lambda i: (i.published_at, i.title), reverse=True)[:12])

    if ordered:
        status = "live"
    elif stale_ordered and checked:
        status = "stale_only"
    elif checked and errors and raw_seen == 0 and fast_screen_items_seen == 0:
        status = "degraded"
    elif checked:
        status = "empty"
    elif discovered_names or unmapped_sponsors:
        status = "discovered_unmapped"
    else:
        status = "unmapped"

    audit = SponsorEvidenceAudit(
        sponsors_discovered=len(discovered_names),
        sponsors_searched=len(checked),
        mapped_sources_used=len(sources),
        unmapped_sponsors=len(unmapped_sponsors),
        raw_items_seen=raw_seen,
        candidate_items=candidates,
        accepted_items=len(ordered),
        stale_items_removed=len(stale_ordered),
        low_lane_relevance_removed=low_lane_removed,
        source_errors=len(errors),
        source_routes_checked=("ticker_news", "fast_news_screen", "generic_ir_newsroom_discovery", "promoted_evidence_parse", "IR/PR/conference query links for unmapped sponsors"),
        fast_screen_sponsors=len(fast_screened_names),
        fast_screen_items_seen=fast_screen_items_seen,
        promoted_items=promoted_items,
        deep_parsed_items=deep_parsed_items,
        screened_sponsor_universe=len(universe),
        unscreened_sponsors=max(0, len(universe) - len(fast_screened_names)),
        unscreened_high_priority=unscreened_high_priority,
        focus_company_screen_status=focus_company_status,
        sponsor_grade_universe=len(sponsor_grade_sources),
        non_sponsor_entities_deprioritized=non_sponsor_entities_deprioritized,
        freshness_model="publication_date_plus_catalyst_timing",
    )

    return SponsorEvidenceSummary(
        source_status=status,
        fetched_at_utc=fetched_at,
        sponsors_checked=tuple(checked),
        items=tuple(ordered),
        source_errors=tuple(errors),
        discovered_sponsors=tuple(discovered_names),
        unmapped_sponsors=tuple(unmapped_sponsors),
        evidence_search_links=tuple(search_links),
        stale_items=stale_ordered,
        audit=audit,
    )


def sponsor_evidence_table(summary: SponsorEvidenceSummary):
    import pandas as pd

    return pd.DataFrame([
        {
            "Sponsor": item.sponsor,
            "Ticker": item.ticker,
            "Evidence State": item.evidence_state,
            "Catalyst Class": item.catalyst_class,
            "Data Stage": item.data_stage,
            "Evidence Action": item.evidence_action,
            "Freshness": item.freshness_state,
            "Catalyst Year": item.catalyst_year or "",
            "Confidence": item.confidence,
            "Title": item.title,
            "Publisher": item.publisher,
            "Published": item.published_at,
            "Matched Terms": ", ".join(item.matched_terms),
            "Overlap Terms": ", ".join(item.overlap_terms),
            "Provenance": item.provenance,
            "Source Quality": item.source_quality,
            "Relevance Tier": item.relevance_tier,
            "Relevance Score": item.relevance_score,
            "Evidence Route": item.evidence_route,
            "Suppression Reason": item.suppression_reason,
            "URL": item.url,
        }
        for item in summary.items
    ])


def stale_sponsor_evidence_table(summary: SponsorEvidenceSummary):
    import pandas as pd

    return pd.DataFrame([
        {
            "Sponsor": item.sponsor,
            "Ticker": item.ticker,
            "Title": item.title,
            "Published": item.published_at,
            "Catalyst Year": item.catalyst_year or "",
            "Suppression Reason": item.suppression_reason,
            "Publisher": item.publisher,
            "URL": item.url,
        }
        for item in summary.stale_items
    ])


def sponsor_evidence_audit_table(summary: SponsorEvidenceSummary):
    import pandas as pd

    if summary.audit is None:
        return pd.DataFrame()
    audit = summary.audit
    return pd.DataFrame([{
        "Sponsors Discovered": audit.sponsors_discovered,
        "Sponsors Searched": audit.sponsors_searched,
        "Screened Sponsor Universe": audit.screened_sponsor_universe,
        "Fast Screen Sponsors": audit.fast_screen_sponsors,
        "Unscreened Sponsors": audit.unscreened_sponsors,
        "High-Priority Unscreened": ", ".join(audit.unscreened_high_priority),
        "Focus Company Screen Status": audit.focus_company_screen_status,
        "Sponsor-Grade Universe": audit.sponsor_grade_universe,
        "Non-Sponsor Entities Deprioritized": audit.non_sponsor_entities_deprioritized,
        "Freshness Model": audit.freshness_model,
        "Fast Screen Items Seen": audit.fast_screen_items_seen,
        "Promoted Items": audit.promoted_items,
        "Deep Parsed Items": audit.deep_parsed_items,
        "Mapped Sources Used": audit.mapped_sources_used,
        "Unmapped Sponsors": audit.unmapped_sponsors,
        "Raw Items Seen": audit.raw_items_seen,
        "Candidate Items": audit.candidate_items,
        "Accepted Items": audit.accepted_items,
        "Stale Items Removed": audit.stale_items_removed,
        "Low-Lane Items Removed": audit.low_lane_relevance_removed,
        "Source Errors": audit.source_errors,
        "Routes Checked": ", ".join(audit.source_routes_checked),
    }])
