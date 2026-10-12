# Money laundering typologies

A small reference corpus for grounding alert explanations. Each entry is a widely
documented pattern, described generically. These are descriptions of behaviour, not
determinations about any account.

Retrieval matches an alert's evidence pack against these entries; the explanation must
cite the id of any typology it names. An explanation that names a typology not in the
retrieved set fails the citation check.

---

## id: structuring
### name: Structuring (smurfing)
Value is broken into many payments that individually stay below a reporting or review
threshold, so no single transaction attracts attention. Indicators: a cluster of
similar amounts from one account over a short window; amounts sitting just under a
round figure; a sudden rise in transaction count without a matching rise in the usual
per-transaction size.
Distinguishing note: legitimate payroll, subscription billing and instalment plans also
produce repeated similar amounts. Regularity of timing argues for a legitimate
explanation; irregular bursts argue against one.

---

## id: layering
### name: Layering
Funds are moved through a chain of accounts to separate them from their origin. Each
hop is unremarkable alone; the pattern only appears across the chain. Indicators: an
account that receives and forwards similar value within a short interval; a
counterparty that is itself new; low balance retention.
Distinguishing note: payment processors, escrow and treasury sweep accounts show the
same pass-through shape as ordinary business.

---

## id: rapid_movement
### name: Rapid movement of funds
Money arrives and leaves within a short window, leaving little or nothing behind. The
account functions as a conduit rather than a store of value. Indicators: a short
interval between an incoming and an outgoing payment of comparable size.
Distinguishing note: a low time gap on its own is weak. It matters when combined with
an amount that is out of character for the account.

---

## id: behaviour_change
### name: Sudden change in account behaviour
An account that has been consistent begins transacting at a different scale, time of
day, or with a different type of counterparty. Indicators: a payment far above the
account's own running average; activity outside its usual hours; a first-time
counterparty after a long history of the same few.
Distinguishing note: genuine life events, seasonal business and one-off large purchases
produce the same signature. Magnitude of deviation matters more than deviation itself.

---

## id: new_account_activity
### name: High-value activity on a new account
An account transacts at significant value before establishing any behavioural history.
There is no baseline to compare against, so the usual deviation signals are unavailable.
Indicators: few or no prior transactions; a first transaction at a value that would be
an outlier for a comparable established account.
Distinguishing note: this is a statement about missing information, not about the
account holder. New accounts transacting normally are the overwhelming majority.

---

## id: cross_currency
### name: Cross-currency conversion
Value changes denomination as part of the transfer, which can complicate tracing.
Indicators: payment currency differs from receiving currency; a conversion that is not
explained by the parties' usual currencies.
Distinguishing note: importers, exporters, travel businesses and anyone with
cross-border suppliers convert currency routinely.

---

## id: counterparty_expansion
### name: Sudden counterparty expansion
An account that dealt with a small, stable set of counterparties begins dealing with
many new ones in a short period. Indicators: a rise in distinct counterparties relative
to the account's own history; a high share of first-time counterparties.
Distinguishing note: a growing business naturally expands its counterparty set. Rate of
change relative to the account's own past is the signal, not the absolute count.
