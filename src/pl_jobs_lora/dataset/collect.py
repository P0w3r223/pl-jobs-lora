"""theprotocol collector for the probe slice: offer HTML -> (prose input, platform-gold JSON).

Parsing (``build_prose`` / ``build_gold`` / ``parse_offer_page``) is pure and offline-testable;
network I/O lives in ``fetch_*`` / ``collect_dev_slice``. Etiquette knobs come from config
(browser UA, throttle, bounded spread sample) — mirrors it-job-radar.

Leakage guard (ADR-0002): the model INPUT is built only from the free-text sections
(``textSections`` / ``jsonSections``); the structured label widgets (``technologies``,
``attributes``) feed GOLD, never the input. Raw HTML is never persisted — only prose-derived
fields leave the machine.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass

import requests

from pl_jobs_lora import normalize, vocab
from pl_jobs_lora.config import Config

_NEXT_DATA_RE = re.compile(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', re.DOTALL)
_OFFER_LOC_RE = re.compile(r"<loc>(https://theprotocol\.it/szczegoly/[^<]+)</loc>")

# theprotocol salary kindCode -> our contract kind.
_KIND_MAP = {"gross": vocab.CONTRACT_EMPLOYMENT, "netto (+ vat)": vocab.CONTRACT_B2B}

# Section titles that echo the structured technologies widget verbatim — dropped from the
# model INPUT (leakage guard, ADR-0002); the same techs still occur naturally in prose bullets.
_TECH_TITLES = frozenset({
    "technologies", "technologies we use", "technology stack", "tech stack",
    "technologie", "nasze technologie", "wymagania techniczne", "expected", "optional",
})


@dataclass(frozen=True)
class DevExample:
    """One probe example: prose input + platform-gold labels + publication date."""

    offer_id: str
    url: str
    pub_date: str | None
    prose: str
    gold: dict  # JobPosting-shaped platform-gold


def extract_next_data(html: str) -> dict | None:
    m = _NEXT_DATA_RE.search(html)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def _offer_of(data: dict) -> dict | None:
    offer = data.get("props", {}).get("pageProps", {}).get("offer")
    return offer if isinstance(offer, dict) and offer.get("id") else None


def build_prose(offer: dict) -> str:
    """Model input from the titled ``jsonSections`` (structured prose), excluding the
    technologies widget echo. ``textSections`` are dropped: they mirror this content but also
    render the raw tech list as leading text — a direct label leak."""
    parts: list[str] = []
    for sec in offer.get("jsonSections") or []:
        title = (sec.get("title") or "").strip()
        if title.lower() in _TECH_TITLES:
            continue  # leakage guard
        model = sec.get("model") or {}
        lines = [ln for ln in (model.get("paragraphs") or []) if ln and ln.strip()]
        lines += [f"- {b}" for b in model.get("bullets") or [] if b and b.strip()]
        if lines:
            body = "\n".join(lines)
            parts.append(f"## {title}\n{body}" if title else body)
    return "\n\n".join(parts).strip()


def _canon_list(values, mapper, allowed: tuple[str, ...]) -> list[str]:
    out: list[str] = []
    for v in values or []:
        c = mapper(v)
        if c in allowed and c not in out:
            out.append(c)
    return out


def _build_salary(employment: dict) -> dict | None:
    for contract in employment.get("typesOfContracts") or []:
        salary = contract.get("salary")
        if not isinstance(salary, dict) or salary.get("from") is None:
            continue
        kind_raw = (salary.get("kindCode") or "").strip().lower()
        time_unit = (salary.get("timeUnit") or {}).get("longForm")
        return {
            "kind": _KIND_MAP.get(kind_raw),
            "currency": normalize.normalize_currency(salary.get("currencyCode")),
            "amount_from": salary.get("from"),
            "amount_to": salary.get("to"),
            "period": "hour" if (time_unit or "").lower().startswith("godzin") else "month",
        }
    return None


def build_gold(offer: dict, alias_index: dict[str, str]) -> dict:
    """Platform-gold JobPosting-shaped dict, normalized (leakage widgets, never fed to model)."""
    attrs = offer.get("attributes") or {}
    employment = attrs.get("employment") or {}
    tech = offer.get("technologies") or {}

    def _tech(items):
        seen: list[str] = []
        for t in items or []:
            name = t.get("name")
            c = normalize.normalize_technology(name, alias_index) if name else ""
            if c and c not in seen:
                seen.append(c)
        return seen

    return {
        "title": (attrs.get("title") or {}).get("value"),
        "seniority": _canon_list(
            employment.get("positionLevelIds"), normalize.normalize_seniority, vocab.SENIORITY_ORDER
        ),
        "work_mode": _canon_list(
            [w.get("code") for w in employment.get("detailedWorkModes") or []],
            normalize.normalize_work_mode, vocab.WORK_MODE_CANON,
        ),
        "tech_expected": _tech(tech.get("expected")),
        "tech_optional": _tech(tech.get("optional")),
        "contract_types": [
            (c.get("name") or "").strip().lower()
            for c in employment.get("typesOfContracts") or [] if c.get("name")
        ],
        "salary": _build_salary(employment),
    }


def parse_offer_page(html: str, alias_index: dict[str, str]) -> DevExample | None:
    """Pure: offer HTML -> DevExample, or None if prose is missing (unusable for the probe)."""
    data = extract_next_data(html)
    offer = _offer_of(data) if data else None
    if not offer:
        return None
    prose = build_prose(offer)
    if not prose:
        return None  # no input text -> not a probe example
    return DevExample(
        offer_id=str(offer["id"]),
        url="",
        pub_date=(offer.get("publicationDetails") or {}).get("dateOfInitialPublicationUtc"),
        prose=prose,
        gold=build_gold(offer, alias_index),
    )


# --- Network I/O -------------------------------------------------------------

def _session(cfg: Config) -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": cfg.collection.user_agent})
    return s


def fetch_sitemap_urls(cfg: Config, session: requests.Session) -> list[str]:
    resp = session.get(cfg.collection.sitemap_url, timeout=cfg.collection.request_timeout_s)
    resp.raise_for_status()
    return _OFFER_LOC_RE.findall(resp.text)


def _spread_sample(urls: list[str], n: int) -> list[str]:
    if n >= len(urls):
        return urls
    step = max(1, len(urls) // n)
    return urls[::step][:n]


def collect_dev_slice(cfg: Config, alias_index: dict[str, str]) -> list[DevExample]:
    """Fetch a bounded, throttled, spread slice of offers with prose + gold (raw HTML dropped)."""
    session = _session(cfg)
    urls = _spread_sample(fetch_sitemap_urls(cfg, session), cfg.probe.dev_slice_size)
    out: list[DevExample] = []
    for url in urls:
        try:
            resp = session.get(url, timeout=cfg.collection.request_timeout_s)
            resp.raise_for_status()
            ex = parse_offer_page(resp.text, alias_index)
            if ex:
                out.append(DevExample(**{**asdict(ex), "url": url}))
        except requests.RequestException:
            continue
        finally:
            time.sleep(cfg.collection.request_delay_s)
    print(f"[collect] {len(urls)} urls -> {len(out)} usable")
    return out
