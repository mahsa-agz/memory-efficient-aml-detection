"""
Features for each transaction.

TWO PROTOCOLS
    paper   all transaction fields, like the TransXion paper. Use it to compare with their numbers.
    clean   without amount_received / receiving_currency. Use it to show the memory works
            without the simulator's shortcut: amount_paid == amount_received for 100% of
            laundering rows but only 8.4% of clean rows.

CURRENT: from the transaction itself (9 columns, both protocols)
    log_amount_paid          log(1 + amount)
    hour_sin, hour_cos       hour of day, on a circle (23:00 sits next to 00:00)
    weekday_sin, weekday_cos day of week, on a circle (Sunday sits next to Monday)
    from_bank_code           sender bank, as an integer code
    to_bank_code             receiver bank, as an integer code
    payment_format_code      Cash / Mobile / Wire ..., as an integer code
    payment_currency_code    currency, as an integer code

ARTIFACT: the received side of the transaction (4 columns, paper protocol only)
    log_amount_received      log(1 + amount received)
    paid_received_gap        |paid - received| / paid       (0 for every laundering row)
    cross_currency           1 if paid and received currencies differ
    receiving_currency_code  receiving currency, as an integer code

PROFILE: the account's static profile, for sender and receiver (9 + 9 = 18 columns, optional)
    is_merchant              1 = merchant, 0 = person
    age_years                person's age, or the merchant's years since founding (at 2025-01-01)
    education_code, marital_code, occupation_code                    people only (NaN for merchants)
    merchant_type_code, industry_code, log_capital, operating_code   merchants only (NaN for people)
    gender is left out (USE_GENDER = False): a protected attribute

MEMORY: from each account's past, for sender and receiver (27 + 27 + 1 = 55 columns)
    the receiver remembers the amount RECEIVED (paper) or the amount PAID (clean)

  LONG-TERM: what the account is normally like (all of its history)
    count                    transactions before this one
    mean_amount              average amount so far
    amount_to_mean           this amount / its average      (5 = five times usual)
    std_amount               how much its amounts usually vary (Welford)
    amount_z                 (amount - mean) / std           (0 = typical, 10 = very unusual)

  SHORT-TERM: what the account has been doing lately (old transactions fade)
    ema_fast                 average of roughly the last 3 amounts   (alpha 0.3)
    ema_slow                 average of roughly the last 33 amounts  (alpha 0.03)
    fast_to_slow             ema_fast / ema_slow  (>1 = amounts rising lately)
    log_seconds_since_last   log(1 + seconds since its previous transaction)

  DIRECTION
    incoming_fraction        share of its past transactions that were incoming

  WHO IT DEALS WITH: a vector of 64 counters, counterparty hashed into one slot
    counterparty_frequency   past transactions with this counterparty (0 = first time)
    distinct_counterparties  estimated number of different counterparties
    counterparty_entropy     spread over many (high) or concentrated on few (low)

  RECENT WINDOWS: the last 5 and the last 20 transactions (7 each, prefix r5_ / r20_)
    rN_count                 transactions in the window (fewer than N for young accounts)
    rN_mean_amount           average amount in the window
    rN_std_amount            how much amounts vary in the window
    amount_to_rN_mean        this amount / the window's average
    rN_distinct_counterparties  different counterparties in the window
    rN_outgoing_fraction     share of the window that was outgoing
    rN_log_span_seconds      log(1 + seconds from first to last in the window) (short = burst)

  BETWEEN THE TWO ACCOUNTS
    sender_receiver_similarity  cosine of their two vectors (1 = same circle of accounts)

  Summary numbers plus a buffer of the last 20 transactions; older ones drop out,
  so memory per account stays the same size however long its history.

  For every transaction: READ the state first, THEN update it.
  Swapping the two would let a transaction see itself.

Total: paper 9 + 4 + 55 = 68 columns, clean 9 + 55 = 64 columns; + 18 with the profile.
"""
import zlib
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


VECTOR_SIZE = 64
FAST, SLOW = 0.3, 0.03
WINDOWS = (5, 20)

# which amount the receiver's memory uses, per protocol
PROTOCOLS = {"paper": "amount_received", "clean": "amount_paid"}

CURRENT_FEATURES = [
    "log_amount_paid",
    "hour_sin", "hour_cos",
    "weekday_sin", "weekday_cos",
    "from_bank_code", "to_bank_code",
    "payment_format_code", "payment_currency_code",
]

ARTIFACT_FEATURES = [
    "log_amount_received",
    "paid_received_gap",
    "cross_currency",
    "receiving_currency_code",
]

USE_GENDER = False      # protected attribute; off by default, switch on only to test it

PROFILE_COLUMNS = [
    "is_merchant", "age_years",
    "education_code", "marital_code", "occupation_code",                      # people
    "merchant_type_code", "industry_code", "log_capital", "operating_code",   # merchants
] + (["gender_code"] if USE_GENDER else [])

PROFILE_FEATURES = [f"{side}_{c}" for side in ("sender", "receiver") for c in PROFILE_COLUMNS]

SIDE_FEATURES = [
    "count", "mean_amount", "amount_to_mean", "std_amount", "amount_z",
    "ema_fast", "ema_slow", "fast_to_slow", "log_seconds_since_last",
    "incoming_fraction",
    "counterparty_frequency", "distinct_counterparties", "counterparty_entropy",
] + [
    name
    for w in WINDOWS
    for name in (f"r{w}_count", f"r{w}_mean_amount", f"r{w}_std_amount", f"amount_to_r{w}_mean",
                 f"r{w}_distinct_counterparties", f"r{w}_outgoing_fraction", f"r{w}_log_span_seconds")
]

MEMORY_FEATURES = (
    [f"sender_{name}" for name in SIDE_FEATURES]
    + [f"receiver_{name}" for name in SIDE_FEATURES]
    + ["sender_receiver_similarity"]
)


def current_matrix(df):
    hour = df["timestamp"].dt.hour.to_numpy()
    weekday = df["timestamp"].dt.dayofweek.to_numpy()
    return np.column_stack([
        np.log1p(df["amount_paid"].to_numpy()),
        np.sin(2 * np.pi * hour / 24),
        np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * weekday / 7),
        np.cos(2 * np.pi * weekday / 7),
        pd.factorize(df["from_bank"].astype(str), sort=True)[0],
        pd.factorize(df["to_bank"].astype(str), sort=True)[0],
        pd.factorize(df["payment_format"].astype(str), sort=True)[0],
        pd.factorize(df["payment_currency"].astype(str), sort=True)[0],
    ]).astype(np.float32)


def artifact_matrix(df):
    # the received side; it carries the simulator's paid == received shortcut
    paid = df["amount_paid"].to_numpy()
    received = df["amount_received"].to_numpy()
    return np.column_stack([
        np.log1p(received),
        np.abs(paid - received) / np.maximum(np.abs(paid), 1e-6),
        (df["payment_currency"].astype(str) != df["receiving_currency"].astype(str)).astype(float),
        pd.factorize(df["receiving_currency"].astype(str), sort=True)[0],
    ]).astype(np.float32)


def code(series):
    return pd.Series(pd.factorize(series.astype(str), sort=True)[0], index=series.index).astype(float)


def profile_table(folder="data/raw"):
    # one row per account: people and merchants in one table; fields that do not apply are NaN
    p = pd.read_csv(Path(folder) / "person.csv")
    m = pd.read_csv(Path(folder) / "merchant.csv")

    people = pd.DataFrame({
        "is_merchant": 0.0,
        "age_years": p["person_age"].astype(float),
        "education_code": code(p["person_education"]),
        "marital_code": code(p["person_marital_status"]),
        "occupation_code": code(p["person_occupation"]),
        "gender_code": code(p["person_gender"]),
    }).set_index(p["bank_account_number"])

    firms = pd.DataFrame({
        "is_merchant": 1.0,
        "age_years": (pd.Timestamp("2025-01-01") - pd.to_datetime(m["establishment_date"])).dt.days / 365.25,
        "merchant_type_code": code(m["type"]),
        "industry_code": code(m["industry"]),
        "log_capital": np.log1p(m["registered_capital"].astype(float)),
        "operating_code": code(m["operating_status"]),
    }).set_index(m["bank_account_number"])

    return pd.concat([people, firms]).reindex(columns=PROFILE_COLUMNS)


def profile_matrix(df):
    # look up the sender's and the receiver's profile for every transaction
    table = profile_table()
    sender = table.reindex(df["sender_account"]).to_numpy()
    receiver = table.reindex(df["receiver_account"]).to_numpy()
    return np.hstack([sender, receiver]).astype(np.float32)


def slot(counterparty):
    # crc32, not hash(): Python's hash() changes every run, crc32 is always the same
    return zlib.crc32(str(counterparty).encode()) % VECTOR_SIZE


class AccountState:
    # summary numbers only, never the transactions themselves
    def __init__(self):
        # long-term
        self.count = 0
        self.mean_amount = 0.0
        self.m2_amount = 0.0
        # short-term
        self.ema_fast = 0.0
        self.ema_slow = 0.0
        self.last_time = 0
        # direction
        self.n_in = 0
        # who it deals with
        self.vector = np.zeros(VECTOR_SIZE, dtype=np.float32)
        # recent windows: the last 20 transactions as (amount, counterparty, direction, time)
        self.recent = deque(maxlen=max(WINDOWS))

    # READ: describe this transaction using only the past
    def features(self, amount, time, counterparty):
        if self.count == 0:
            return [0.0] * len(SIDE_FEATURES)

        std = (self.m2_amount / (self.count - 1)) ** 0.5 if self.count > 1 else 0.0
        zero_slots = int((self.vector == 0).sum())
        distinct = -VECTOR_SIZE * np.log(max(zero_slots, 1) / VECTOR_SIZE)
        p = self.vector[self.vector > 0] / self.count
        entropy = float(-(p * np.log(p)).sum())

        values = [
            self.count,
            self.mean_amount,
            amount / max(self.mean_amount, 1e-6),
            std,
            (amount - self.mean_amount) / std if std > 0 else 0.0,
            self.ema_fast,
            self.ema_slow,
            self.ema_fast / max(self.ema_slow, 1e-6),
            np.log1p(max(time - self.last_time, 0)),
            self.n_in / self.count,
            float(self.vector[slot(counterparty)]),
            distinct,
            entropy,
        ]

        recent = list(self.recent)
        for w in WINDOWS:
            items = recent[-w:]
            amounts = [item[0] for item in items]
            mean = sum(amounts) / len(amounts)
            window_std = (sum((a - mean) ** 2 for a in amounts) / len(amounts)) ** 0.5
            values += [
                len(items),
                mean,
                window_std,
                amount / max(mean, 1e-6),
                len({item[1] for item in items}),
                sum(item[2] == "out" for item in items) / len(items),
                np.log1p(max(items[-1][3] - items[0][3], 0)),
            ]
        return values

    # WRITE: add this transaction to the summary
    def update(self, amount, time, counterparty, direction):
        self.count += 1
        delta = amount - self.mean_amount
        self.mean_amount += delta / self.count
        self.m2_amount += delta * (amount - self.mean_amount)

        if self.count == 1:
            self.ema_fast = self.ema_slow = amount
        else:
            self.ema_fast = FAST * amount + (1 - FAST) * self.ema_fast
            self.ema_slow = SLOW * amount + (1 - SLOW) * self.ema_slow

        self.last_time = time
        self.n_in += direction == "in"
        self.vector[slot(counterparty)] += 1
        self.recent.append((amount, counterparty, direction, time))   # oldest drops out at 20


def similarity(a, b):
    # cosine of the two counterparty vectors: 1 = same circle, 0 = nothing in common
    norm = float(np.linalg.norm(a.vector) * np.linalg.norm(b.vector))
    return float(a.vector @ b.vector) / norm if norm > 0 else 0.0


def memory_matrix(df, receiver_amount="amount_received"):
    states = {}
    rows = np.zeros((len(df), len(MEMORY_FEATURES)), dtype=np.float32)
    seconds = df["timestamp"].astype("int64").to_numpy() // 10**9
    data = zip(df["sender_account"], df["receiver_account"],
               df["amount_paid"], df[receiver_amount], seconds)

    for i, (sender, receiver, paid, r_amount, t) in enumerate(tqdm(data, total=len(df), desc="memory")):
        s = states.setdefault(sender, AccountState())
        r = states.setdefault(receiver, AccountState())

        # 1. READ first: the state BEFORE this transaction
        rows[i] = s.features(paid, t, receiver) + r.features(r_amount, t, sender) + [similarity(s, r)]

        # 2. THEN WRITE: never swap these two blocks
        s.update(paid, t, receiver, "out")
        r.update(r_amount, t, sender, "in")

    return rows


def is_window(name):
    # "sender_r5_count", "receiver_amount_to_r20_mean", ... are window columns
    return name.split("_", 1)[1].startswith(("r5_", "r20_", "amount_to_r"))


def is_ema(name):
    # the decaying short-term part: "sender_ema_fast", "receiver_fast_to_slow", ...
    return name.split("_", 1)[1] in ("ema_fast", "ema_slow", "fast_to_slow")


def memory_columns(windows, ema=True):
    # indices into MEMORY_FEATURES, original order kept
    #   long-term only : ema=False, windows=False
    #   long + EMA     : ema=True,  windows=False   (the original 27 columns, "compact")
    #   long + window  : ema=False, windows=True
    #   long + both    : ema=True,  windows=True    ("full")
    return [i for i, name in enumerate(MEMORY_FEATURES)
            if (windows or not is_window(name)) and (ema or not is_ema(name))]


def build_features(df, protocol, memory=True, profile=False, windows=False, ema=True):
    # one place that decides the columns, so every script builds the same matrix
    if protocol not in PROTOCOLS:
        raise ValueError(f"protocol must be one of {list(PROTOCOLS)}, got {protocol!r}")
    blocks, names = [current_matrix(df)], list(CURRENT_FEATURES)
    if protocol == "paper":
        blocks.append(artifact_matrix(df))
        names += ARTIFACT_FEATURES
    if profile:
        blocks.append(profile_matrix(df))
        names += PROFILE_FEATURES
    if memory:
        keep = memory_columns(windows, ema)
        blocks.append(memory_matrix(df, PROTOCOLS[protocol])[:, keep])
        names += [MEMORY_FEATURES[i] for i in keep]
    return np.hstack(blocks), names