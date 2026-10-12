from pathlib import Path
import pandas as pd

"""
0%                60%          80%          100%
|------ train -----|-- valid ---|--- test ---|
       60%              20%          20%
"""

Train = 0.60
Validation = 0.80

RENAME = {
    "Timestamp": "timestamp",
    "From Bank": "from_bank",
    "From Account": "sender_account",
    "To Bank": "to_bank",
    "To Account": "receiver_account",
    "Amount Received": "amount_received",
    "Receiving Currency": "receiving_currency",
    "Amount Paid": "amount_paid",
    "Payment Currency": "payment_currency",
    "Payment Format": "payment_format",
    "Is Laundering": "is_laundering",
}

def load_transactions(path, max_rows = None):
    df = pd.read_csv(path, nrows = max_rows).rename(columns = RENAME)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors = "raise")
    return df.sort_values("timestamp", kind = "stable").reset_index(drop = True)

def split_data(rows):
    return int(rows * Train), int(rows * Validation)

