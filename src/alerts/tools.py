# read-only tools an agent may call while investigating one alert:
# only the alert's two accounts, only the past, at most MAX_ROWS rows

MAX_ROWS = 20

TOOL_SCHEMAS = [
    {
        "name": "get_account_profile",
        "description": "Profile and history summary of the alert's sender or receiver.",
        "input_schema": {
            "type": "object",
            "properties": {"account": {"type": "string", "description": "Account number."}},
            "required": ["account"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_recent_transactions",
        "description": f"The account's last transactions before the alert, at most {MAX_ROWS}.",
        "input_schema": {
            "type": "object",
            "properties": {
                "account": {"type": "string"},
                "limit": {"type": "integer", "description": f"1-{MAX_ROWS}."},
            },
            "required": ["account"],
            "additionalProperties": False,
        },
    },
    {
        "name": "check_typology_rule",
        "description": "Run one typology rule on the alert and return its inputs and result.",
        "input_schema": {
            "type": "object",
            "properties": {
                "typology_id": {
                    "type": "string",
                    "enum": ["rapid_movement", "behaviour_change", "counterparty_expansion"],
                }
            },
            "required": ["typology_id"],
            "additionalProperties": False,
        },
    },
]


class AlertToolbox:
    # one toolbox per alert

    def __init__(self, df, i, pack):
        self.df = df
        self.i = i
        self.pack = pack
        row = df.iloc[i]
        self.accounts = {str(row["sender_account"]), str(row["receiver_account"])}
        self.time = row["timestamp"]

    def out_of_scope(self, account):
        if str(account) in self.accounts:
            return None
        return {
            "error": "out_of_scope",
            "detail": f"account {account} is not part of this alert; only {sorted(self.accounts)}",
        }

    def get_account_profile(self, account):
        refused = self.out_of_scope(account)
        if refused:
            return refused
        row = self.df.iloc[self.i]
        side = "sender" if str(row["sender_account"]) == str(account) else "receiver"
        return {
            "account": account,
            "role_in_alert": side,
            **self.pack[side],
        }

    def get_recent_transactions(self, account, limit=5):
        refused = self.out_of_scope(account)
        if refused:
            return refused
        limit = max(1, min(int(limit), MAX_ROWS))
        # earlier in time, not only in row order: about 12% of rows share a timestamp
        before = self.df.iloc[:self.i]
        before = before[before["timestamp"] < self.time]
        mine = before[
            (before["sender_account"].astype(str) == str(account))
            | (before["receiver_account"].astype(str) == str(account))
        ].tail(limit)
        rows = []
        for _, r in mine.iterrows():
            rows.append({
                "occurred_at": r["timestamp"].isoformat(),
                "amount_paid": round(float(r["amount_paid"]), 2),
                "payment_currency": str(r["payment_currency"]),
                "direction": "out" if str(r["sender_account"]) == str(account) else "in",
            })
        return {
            "account": account,
            "returned": len(rows),
            "limit_applied": limit,
            "transactions": rows,
        }

    def check_typology_rule(self, typology_id):
        sender = self.pack["sender"]
        if typology_id == "behaviour_change":
            ratio = self.pack["comparison"].get("amount_vs_sender_usual")
            return {
                "typology_id": typology_id,
                "inputs": {"amount_vs_sender_usual": ratio},
                "matches": bool(ratio is not None and ratio >= 3),
            }
        if typology_id == "counterparty_expansion":
            distinct = sender.get("distinct_counterparties")
            is_new = sender.get("counterparty_is_new")
            return {
                "typology_id": typology_id,
                "inputs": {"distinct_counterparties": distinct, "counterparty_is_new": is_new},
                "matches": bool(is_new and (distinct or 0) > 20),
            }
        if typology_id == "rapid_movement":
            receiver = str(self.df.iloc[self.i]["receiver_account"])
            recent = self.get_recent_transactions(receiver, limit=MAX_ROWS)
            incoming = [t for t in recent.get("transactions", []) if t["direction"] == "in"]
            return {
                "typology_id": typology_id,
                "inputs": {"prior_incoming_seen": len(incoming)},
                "matches": len(incoming) > 0,
            }
        return {
            "error": "unknown_typology",
            "detail": typology_id,
        }

    def call(self, name, arguments):
        # an unknown tool or bad arguments come back as a result, never a crash
        tools = {
            "get_account_profile": self.get_account_profile,
            "get_recent_transactions": self.get_recent_transactions,
            "check_typology_rule": self.check_typology_rule,
        }
        if name not in tools:
            return {
                "error": "unknown_tool",
                "detail": name,
            }
        try:
            return tools[name](**arguments)
        except TypeError as error:
            return {
                "error": "bad_arguments",
                "detail": str(error),
            }
