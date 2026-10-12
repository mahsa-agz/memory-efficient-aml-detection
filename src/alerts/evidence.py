import numpy as np

# one alert -> one small dictionary of facts; the explainer sees only this

RECENT = 5           # last transactions shown per account
MAX_FLAGGED = 10     # earlier flagged transactions shown per account
FLAGGED = 0.70       # flagged score: catches 90% of laundering on validation


def rounded(value, digits=2):
    # NaN becomes None (null in JSON)
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    return round(float(value), digits)


def side(values, prefix):
    history = values.get(f"{prefix}_count", 0.0)
    return {
        "transactions_seen_before": int(history) if np.isfinite(history) else 0,
        "usual_amount": rounded(values.get(f"{prefix}_mean_amount")),
        "recent_mean_amount": rounded(values.get(f"{prefix}_ema_fast")),
        "distinct_counterparties": rounded(values.get(f"{prefix}_distinct_counterparties"), 1),
        "counterparty_is_new": bool(values.get(f"{prefix}_counterparty_frequency", 0.0) == 0.0),  # 0 = first time
    }


def past_transactions(df, scores, account, i, trained_until):
    # only transactions before the alert; no score on training rows
    before = df.iloc[:i]
    before = before[before["timestamp"] < df["timestamp"].iat[i]]
    mine = before[(before["sender_account"] == account) | (before["receiver_account"] == account)]

    def describe(j, r):
        return {
            "transaction_id": int(j),
            "occurred_at": r["timestamp"].isoformat(),
            "amount_paid": rounded(r["amount_paid"]),
            "direction": "out" if r["sender_account"] == account else "in",
            "score": rounded(scores[j], 4) if j >= trained_until else None,
        }

    recent_rows = mine.tail(RECENT)
    flagged_ids = [j for j in mine.index if j >= trained_until and scores[j] >= FLAGGED]
    older_flagged = [j for j in flagged_ids if j not in recent_rows.index][-MAX_FLAGGED:]

    recent = [describe(j, r) for j, r in recent_rows.iterrows()]
    flagged = [describe(j, mine.loc[j]) for j in older_flagged]
    return recent, flagged, len(flagged_ids)


def build_evidence(df, X, names, scores, i, trained_until):
    row = df.iloc[i]
    values = {name: float(X[i, j]) for j, name in enumerate(names)}
    sender = side(values, "sender")
    receiver = side(values, "receiver")

    for pack_side, account in ((sender, row["sender_account"]), (receiver, row["receiver_account"])):
        recent, flagged, count = past_transactions(df, scores, account, i, trained_until)
        pack_side["recent_transactions"] = recent
        pack_side["earlier_flagged_transactions"] = flagged
        pack_side["flagged_count"] = count

    amount = float(row["amount_paid"])
    usual = sender["usual_amount"]

    return {
        "transaction": {
            "occurred_at": row["timestamp"].isoformat(),
            "amount_paid": rounded(amount),
            "payment_currency": str(row["payment_currency"]),
            "receiving_currency": str(row["receiving_currency"]),
            "payment_format": str(row["payment_format"]),
            "cross_currency": str(row["payment_currency"]) != str(row["receiving_currency"]),
            "hour_of_day": int(row["timestamp"].hour),
        },
        "risk": {
            "score": rounded(float(scores[i]), 4),
            "model": "xgboost_bounded_memory",
        },
        "sender": sender,
        "receiver": receiver,
        "comparison": {
            "amount_vs_sender_usual": rounded(amount / usual, 1) if usual else None,
            "sender_has_no_history": sender["transactions_seen_before"] == 0,
            "receiver_has_no_history": receiver["transactions_seen_before"] == 0,
        },
    }


def numbers_in(evidence, digits=2):
    # every number in the pack: the only numbers an explanation may use
    found = set()

    def walk(node):
        if isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
        elif isinstance(node, bool):
            pass  # True is also 1 in Python
        elif isinstance(node, (int, float)) and np.isfinite(node):
            found.add(round(float(node), digits))

    walk(evidence)
    return found
