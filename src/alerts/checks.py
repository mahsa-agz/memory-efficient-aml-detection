import re
import unicodedata

from src.alerts.evidence import numbers_in

# each check: (exp, pack) -> (passed, reason); plain code, no model judges a model

FIELDS = {
    "summary": str,
    "key_signals": list,
    "sufficient_evidence": bool,
    "cited_typologies": list,
}

MIN_SUMMARY = 40
MIN_SIGNAL = 15

MIN_HISTORY = 5     # fewer past transactions: too little to judge
RICH_HISTORY = 20   # this many on both sides: enough to judge

# guilt instead of a pattern; a word list is a tripwire, not a guarantee
VERDICT = [
    "money laundering", "laundering", "launderer", "is a criminal", "criminal proceeds",
    "is fraud", "is fraudulent", "committed", "guilty", "culpable", "perpetrator",
    "proves", "proven", "confirms", "conclusive", "conclusively", "establishes intent",
    "beyond doubt", "beyond any reasonable doubt", "no doubt", "undoubtedly",
    "definitely", "without question", "must be reported to law enforcement",
    "file a suspicious activity report", "must be filed",
]

# words that close an alert; a high score must never be called routine
CLOSING = [
    "routine", "no action", "no further action", "low risk", "can be closed",
    "false positive", "nothing here warrants", "no cause for concern",
    "previously cleared", "already cleared", "no escalation",
]
ALERT_SCORE = 0.50

NUMBER = re.compile(r"\d[\d,]*\.?\d*")

# numbers must be digits so they can be checked
NUMBER_WORDS = re.compile(
    r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty|thirty|"
    r"forty|fifty|hundred|thousand|million|billion|dozen|half|triple|quadruple)\b"
)

# facts the pack can never contain, so any claim about them is made up
UNSUPPORTED = [
    "shell company", "shell entity", "jurisdiction", "sanctions", "sanctioned",
    "sar filing", "suspicious activity report", "watchlist", "blacklist",
    "ip address", "geolocat", "kyc", "know your customer", "beneficial owner",
    "law enforcement", "other institutions", "prior alert", "previous alert",
    "mule network", "cash withdrawal", "withdrawn in cash", "offshore",
    "criminal record", "politically exposed",
]

# protected attributes
PROHIBITED = [
    "nationality", "national origin", "ethnic", "ethnicity", "race", "racial",
    "religion", "religious", "muslim", "christian", "jewish",
    "immigrant", "migrant", "foreigner", "refugee", "asylum",
    "gender", "sex of", "marital status", "age of the customer",
    "their name suggests", "name is typical", "remittance corridor",
]


def normalize(text):
    # remove tricks that hide a phrase: invisible characters, hyphens, spaces, case
    text = unicodedata.normalize("NFKC", text)
    text = re.sub("[​-‏⁠﻿]", "", text)
    text = text.replace("-", " ")
    return re.sub(r"\s+", " ", text).lower().strip()


def text_of(exp):
    # all the prose the model wrote, extra fields included
    parts = [str(exp.get("summary", ""))]
    parts += [str(s) for s in exp.get("key_signals", []) or []]
    parts += [v for k, v in exp.items() if k not in FIELDS and isinstance(v, str)]
    return " ".join(parts)


def numbers_in_text(text):
    found = set()
    for token in NUMBER.findall(text):
        try:
            found.add(round(float(token.replace(",", "")), 2))
        except ValueError:
            continue
    return found


def allowed_numbers(pack):
    # pack numbers in full and at 2 decimals, plus the alert's date
    allowed = set()
    for v in numbers_in(pack, digits=4):
        allowed.add(v)
        allowed.add(round(v, 2))
    date = str(pack.get("transaction", {}).get("occurred_at", ""))
    allowed |= {float(part) for part in re.findall(r"\d+", date)[:4]}
    return allowed


def currencies(pack):
    t = pack.get("transaction", {})
    return {str(t.get("payment_currency", "")).lower(),
            str(t.get("receiving_currency", "")).lower()} - {""}


def check_schema(exp, pack):
    if not isinstance(exp, dict):
        return False, f"expected an object, got {type(exp).__name__}"
    for field, kind in FIELDS.items():
        if field not in exp:
            return False, f"missing field '{field}'"
        if not isinstance(exp[field], kind):
            return False, f"'{field}' should be {kind.__name__}, got {type(exp[field]).__name__}"
    extra = set(exp) - set(FIELDS)
    if extra:
        return False, f"unexpected fields: {sorted(extra)}"
    if len(normalize(exp["summary"])) < MIN_SUMMARY:
        return False, f"summary under {MIN_SUMMARY} characters"
    for s in exp["key_signals"]:
        if not isinstance(s, str):
            return False, f"key_signals contains a {type(s).__name__}, expected str"
    return True, ""


def check_informative(exp, pack):
    # at least one real signal, and the risk score
    signals = [s for s in exp.get("key_signals", []) or []
               if isinstance(s, str) and len(s.strip()) >= MIN_SIGNAL]
    if not signals:
        return False, f"no key_signal of at least {MIN_SIGNAL} characters"
    score = pack.get("risk", {}).get("score")
    if score is not None and round(float(score), 2) not in numbers_in_text(text_of(exp)):
        return False, "does not state the risk score"
    return True, ""


def check_grounded(exp, pack):
    # every number written must be in the pack
    text = text_of(exp)
    spelled = NUMBER_WORDS.search(normalize(text))
    if spelled:
        return False, f"quantity spelled out ('{spelled.group()}'); write figures as digits"
    made_up = sorted(numbers_in_text(text) - allowed_numbers(pack))
    if made_up:
        return False, f"numbers not in evidence: {made_up[:5]}"
    return True, ""


def check_no_unsupported_claims(exp, pack):
    text = normalize(text_of(exp))
    for topic in UNSUPPORTED:
        if normalize(topic) in text:
            return False, f"claims something the evidence cannot contain: '{topic}'"
    return True, ""


def check_no_prohibited_basis(exp, pack):
    text = normalize(text_of(exp))
    for word in PROHIBITED:
        if normalize(word) in text:
            return False, f"reasons from a protected attribute: '{word}'"
    return True, ""


def check_categorical_consistency(exp, pack):
    # currencies and true/false facts must match the pack
    text = normalize(text_of(exp))
    known = {normalize(c) for c in currencies(pack)}
    for c in ("euro", "dollar", "yen", "pound", "franc", "rupee", "yuan", "bitcoin"):
        if c in text and not any(c in k for k in known):
            return False, f"names a currency not in the evidence: '{c}'"

    if not pack.get("transaction", {}).get("cross_currency", False):
        for phrase in ("cross currency", "currency conversion", "converting", "converted into"):
            if phrase in text:
                return False, f"says '{phrase}' but the payment is single-currency"
    if pack.get("sender", {}).get("counterparty_is_new"):
        for phrase in ("established relationship", "used for years", "regular counterparty",
                       "longstanding", "familiar counterparty"):
            if phrase in text:
                return False, f"says '{phrase}' but the counterparty is new to the sender"
    return True, ""


def check_refusal(exp, pack):
    # don't judge what you can't see, don't refuse what you can
    sender = pack["sender"]["transactions_seen_before"]
    receiver = pack["receiver"]["transactions_seen_before"]
    claimed = bool(exp.get("sufficient_evidence", True))

    if sender < MIN_HISTORY and claimed:
        return False, f"sender has {sender} prior transactions but evidence claimed sufficient"
    if sender >= RICH_HISTORY and receiver >= RICH_HISTORY and not claimed:
        return False, f"both sides well established ({sender}/{receiver}) but evidence claimed insufficient"
    return True, ""


def check_no_verdict(exp, pack):
    text = normalize(text_of(exp))
    for phrase in VERDICT:
        if re.search(rf"\b{re.escape(normalize(phrase))}\b", text):
            return False, f"asserts a verdict: '{phrase}'"
    return True, ""


def check_conclusion_matches_score(exp, pack):
    score = pack.get("risk", {}).get("score")
    if score is None or float(score) < ALERT_SCORE:
        return True, ""
    text = normalize(text_of(exp))
    for phrase in CLOSING:
        if normalize(phrase) in text:
            return False, f"score {score} but explanation says '{phrase}'"
    return True, ""


def check_citations(exp, pack):
    # every cited pattern must have been retrieved for this alert
    available = {p["id"] for p in pack.get("typologies", [])}
    cited = exp.get("cited_typologies", []) or []
    if not isinstance(cited, list):
        return False, "cited_typologies is not a list"
    unknown = [c for c in cited if c not in available]
    if unknown:
        return False, f"cites typologies that were not retrieved: {unknown}"
    if available and not cited and exp.get("sufficient_evidence"):
        return False, "typologies were retrieved but none were cited"
    return True, ""


CHECKS = {
    "schema": check_schema,
    "citations": check_citations,
    "informative": check_informative,
    "grounded": check_grounded,
    "unsupported": check_no_unsupported_claims,
    "categorical": check_categorical_consistency,
    "prohibited_basis": check_no_prohibited_basis,
    "refusal": check_refusal,
    "no_verdict": check_no_verdict,
    "score_match": check_conclusion_matches_score,
}


def run_checks(exp, pack):
    # a check that crashes counts as failed, never as passed
    results = {}
    for name, check in CHECKS.items():
        try:
            results[name] = check(exp, pack)
        except Exception as error:
            results[name] = (False, f"CHECK ERRORED: {type(error).__name__}: {error}")
    return results
