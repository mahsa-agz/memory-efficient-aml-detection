import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

CORPUS_PATH = Path(__file__).resolve().parents[2] / "docs" / "typologies" / "typologies.md"


def load_corpus(path=CORPUS_PATH):
    # one dict per pattern; patterns are separated by "---"
    text = Path(path).read_text(encoding="utf-8")
    corpus = []
    for block in text.split("\n---\n"):
        found_id = re.search(r"^## id:\s*(\S+)", block, re.MULTILINE)
        found_name = re.search(r"^### name:\s*(.+)$", block, re.MULTILINE)
        if not found_id or not found_name:
            continue
        description = block.split(found_name.group(0), 1)[1].strip()
        corpus.append({"id": found_id.group(1), "name": found_name.group(1).strip(),
                       "description": description})
    if not corpus:
        raise SystemExit(f"no patterns found in {path}")
    return corpus


def query_from(evidence):
    # alert facts -> words; never payment_format (the account holder writes it)
    terms = []
    comparison = evidence.get("comparison", {})
    sender = evidence.get("sender", {})
    transaction = evidence.get("transaction", {})

    ratio = comparison.get("amount_vs_sender_usual")
    if ratio is not None and ratio >= 3:
        terms += ["far above the account's own running average", "out of character", "deviation"]
    if sender.get("transactions_seen_before", 0) < 5:
        terms += ["new account", "no behavioural history", "no baseline"]
    if sender.get("counterparty_is_new"):
        terms += ["first-time counterparty", "new counterparty"]
    if transaction.get("cross_currency"):
        terms += ["payment currency differs from receiving currency", "conversion"]
    if (transaction.get("hour_of_day") or 12) <= 5:
        terms += ["outside its usual hours", "time of day"]
    if (sender.get("distinct_counterparties") or 0) > 20:
        terms += ["many counterparties", "counterparty expansion"]
    return " ".join(terms) or "transaction risk"


def retrieve(evidence, top=3, corpus=None):
    # top patterns by TF-IDF similarity, best first
    corpus = corpus if corpus is not None else load_corpus()
    documents = [f"{p['name']}. {p['description']}" for p in corpus]

    matrix = TfidfVectorizer(stop_words="english").fit_transform(documents + [query_from(evidence)])
    similarity = (matrix[:-1] @ matrix[-1].T).toarray().ravel()
    ranked = sorted(zip(similarity, corpus), key=lambda pair: -pair[0])
    return [pattern for score, pattern in ranked[:top] if score > 0]


def as_context(patterns):
    # what goes into the pack
    return [{"id": p["id"], "name": p["name"], "description": p["description"]} for p in patterns]
