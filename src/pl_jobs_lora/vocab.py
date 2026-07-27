"""Canonical vocabulary for the extraction target — seniority, work mode, salary.

VENDORED from P0w3r223/it-job-radar @ 2e051cdb (src/it_job_radar/config.py). ADR-0002
chose vendoring over a git dependency for S1 so P4 is self-contained and unblocked; keep
these maps in sync if the upstream normalization changes. The point of reusing the exact
maps is *scoring fairness*: predictions and gold labels normalize through the same
vocabulary, so ``ReactJS`` vs ``react`` or the Polish ``regular`` → mid quirk are matches,
not errors.

No I/O here — only constants.
"""

from __future__ import annotations

# --- Seniority: theprotocol positionLevelIds + Polish free-text fallbacks ----
# The Polish quirk: "regular" means mid, not entry-level.
SENIORITY_MAP: dict[str, str] = {
    "trainee": "intern", "stażysta": "intern", "praktykant": "intern", "intern": "intern",
    "junior": "junior", "młodszy": "junior", "assistant": "junior",
    "mid": "mid", "regular": "mid", "middle": "mid",
    "senior": "senior", "starszy": "senior",
    "expert": "expert", "lead": "lead", "principal": "principal", "manager": "manager",
}
# Canonical seniority levels, low → high — the allowed enum values.
SENIORITY_ORDER: tuple[str, ...] = (
    "intern", "junior", "mid", "senior", "expert", "lead", "principal", "manager",
)

# --- Work mode (theprotocol detailedWorkModes codes) -------------------------
WORK_MODE_MAP: dict[str, str] = {
    "remote": "remote", "home-office": "remote",
    "hybrid": "hybrid",
    "office": "office", "full-office": "office", "stationary": "office",
    "mobile": "mobile",
}
# Canonical work modes — the allowed enum values.
WORK_MODE_CANON: tuple[str, ...] = ("remote", "hybrid", "office", "mobile")

# --- Salary ------------------------------------------------------------------
# B2B is net-on-invoice, employment (UoP) is gross — kept separate, never averaged.
CONTRACT_B2B = "b2b"
CONTRACT_EMPLOYMENT = "employment"
KNOWN_CURRENCIES: tuple[str, ...] = ("PLN", "EUR", "USD", "GBP")

CURRENCY_MAP: dict[str, str] = {
    "zł": "PLN", "zl": "PLN", "pln": "PLN",
    "€": "EUR", "eur": "EUR",
    "$": "USD", "usd": "USD",
    "£": "GBP", "gbp": "GBP",
}

# --- Fuzzy matching ----------------------------------------------------------
FUZZY_THRESHOLD = 88
