"""
Figures for the README. Reads saved results only; trains nothing.
Run: python -m scripts.figures     (after scripts.final)
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.data import load_transactions, split_data

OUT = Path("docs/figures")      # not in reports/: that folder is git-ignored, the README needs these
FINAL = Path("reports/final")

# the paper's test AP without profiles (arXiv:2604.17420, Tables 2 and 3); single runs
PAPER = {
    "XGBoost": 0.172, "Multi-GIN+EU": 0.269, "Multi-PNA+EU": 0.278,
    "Multi-GIN": 0.288, "Multi-PNA": 0.346, "Multi-GAT": 0.351,
}

# label in the figure (with memory per account), model name in test_summary.csv
OURS = [
    ("Baseline, transaction only", "baseline_paper"),
    ("Long-term memory, O(1)", "baseline_memory_noema_paper"),
    ("Long-term + decay, O(1)", "baseline_memory_paper"),
    ("Long-term + window, O(k)", "baseline_memory_noema_windows_paper"),
    ("Long-term + decay + window, O(k)", "baseline_memory_windows_paper"),
]


def data_over_time():
    # transactions and laundering per day, with the 60/20/20 split
    df = load_transactions(Path("data/raw/tx.csv"))
    train_end, valid_end = split_data(len(df))
    daily = df.set_index("timestamp").resample("D")["is_laundering"].agg(["size", "sum"])

    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    axes[0].plot(daily.index, daily["size"], color="grey", lw=0.8)
    axes[0].set_ylabel("transactions / day")
    axes[1].bar(daily.index, daily["sum"], color="#b2182b", width=1)
    axes[1].set_ylabel("laundering / day")
    for ax in axes:
        for cut in (train_end, valid_end):
            ax.axvline(df["timestamp"].iloc[cut], color="black", ls="--", lw=1)
    top = axes[0].get_ylim()[1]
    for x, label in ((df["timestamp"].iloc[0], "train"), (df["timestamp"].iloc[train_end], "validation"),
                     (df["timestamp"].iloc[valid_end], "test")):
        axes[0].text(x, top, f" {label}", va="top")
    fig.suptitle("One year of transactions, split by time")
    fig.tight_layout()
    fig.savefig(OUT / "data_over_time.png", dpi=150)
    plt.close(fig)


def model_comparison():
    # every model as one bar, sorted by test AP; this project in blue, the paper in grey
    # from the per-seed file, not the 4-decimal summary, so rounding matches the README
    stats = (pd.read_csv(FINAL / "test_per_seed.csv")
             .groupby("model")["average_precision"].agg(["mean", "std"]))
    rows = [(label, stats.loc[m, "mean"], stats.loc[m, "std"], "ours")
            for label, m in OURS if m in stats.index]
    rows += [(f"{name} (paper)", ap, 0.0, "paper") for name, ap in PAPER.items()]
    rows.sort(key=lambda r: r[1])

    colors = {"ours": "#2166ac", "paper": "#bdbdbd"}
    fig, ax = plt.subplots(figsize=(8, 5.5))
    y = range(len(rows))
    ax.barh(y, [r[1] for r in rows], color=[colors[r[3]] for r in rows], height=0.7)
    ours = [i for i, r in enumerate(rows) if r[3] == "ours"]
    ax.errorbar([rows[i][1] for i in ours], ours, xerr=[rows[i][2] for i in ours],
                fmt="none", ecolor="black", capsize=3)
    for i, (_, ap, sd, _) in enumerate(rows):
        ax.text(ap + sd + 0.004, i, f"{ap:.3f}", va="center", fontsize=9)
    ax.set_yticks(list(y), [r[0] for r in rows], fontsize=9)
    ax.set_xlim(0, 0.42)
    ax.set_xlabel("test average precision (this project: mean ± sd over 3 seeds)")
    ax.set_title("TransXion test set, without account profiles")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=colors["ours"]),
                       plt.Rectangle((0, 0), 1, 1, color=colors["paper"])],
              labels=["this project", "benchmark paper (single run)"],
              loc="lower right", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "model_comparison.png", dpi=150)
    plt.close(fig)


def how_it_works():
    # transactions -> main model -> coloured scores -> alert -> evidence -> ... -> analyst
    from matplotlib import colormaps
    from matplotlib.font_manager import FontProperties
    from matplotlib.patches import FancyBboxPatch

    icons = FontProperties(fname=r"C:\Windows\Fonts\seguisym.ttf")   # Windows symbol font
    blue, green, red, grey = "#2166ac", "#1b7837", "#b2182b", "#9e9e9e"
    risk = colormaps["RdYlGn_r"]                                       # 0 green -> 1 red

    def box(x, y, w, h, face, edge, lw=1.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                    facecolor=face, edgecolor=edge, lw=lw))

    def arrow(x1, y1, x2, y2, color):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5))

    fig, ax = plt.subplots(figsize=(13, 5.8))
    ax.set_xlim(0, 13)
    ax.set_ylim(0.4, 6.2)
    ax.axis("off")

    # 1. detect
    ax.text(0.3, 5.9, "Detect: every transaction", fontsize=12, weight="bold", color=blue)

    # incoming transactions, grey
    ax.text(1.0, 5.45, "transactions", fontsize=9, color="#555555", ha="center")
    for k in range(6):
        y = 4.85 - k * 0.38
        box(0.4, y, 1.2, 0.28, "#eeeeee", grey, lw=1)
        ax.text(0.55, y + 0.14, "💳", fontproperties=icons, fontsize=10,
                color=grey, va="center")
        ax.text(1.12, y + 0.14, "tx", fontsize=8, color="#777777", va="center", ha="center")
    arrow(1.7, 4.0, 2.25, 4.0, grey)

    # the main model
    box(2.3, 2.95, 3.6, 2.55, "#e8f1fb", blue, lw=2)
    ax.text(4.1, 5.22, "main model", fontsize=11, weight="bold", color=blue, ha="center")
    parts = [
        ("💾", "long-term memory", "O(1)"),
        ("🕒", "short-term: decay + last 20", "O(k)"),
    ]
    for k, (icon, label, size) in enumerate(parts):
        y = 4.45 - k * 0.62
        box(2.55, y, 3.1, 0.48, "white", blue, lw=1)
        ax.text(2.75, y + 0.24, icon, fontproperties=icons, fontsize=13, color=blue, va="center")
        ax.text(3.1, y + 0.24, label, fontsize=9, va="center")
        ax.text(5.58, y + 0.24, size, fontsize=8, color="#555555", va="center", ha="right")
    box(2.55, 3.15, 3.1, 0.48, blue, blue, lw=1)
    ax.text(2.75, 3.39, "⚙", fontproperties=icons, fontsize=13, color="white", va="center")
    ax.text(3.1, 3.39, "XGBoost", fontsize=10, weight="bold", color="white", va="center")
    arrow(6.0, 4.0, 6.55, 4.0, blue)

    # scored transactions, coloured by risk
    ax.text(7.3, 5.45, "risk score", fontsize=9, color="#555555", ha="center")
    scores = [0.02, 0.08, 0.31, 0.55, 0.90]
    for k, s in enumerate(scores):
        y = 4.85 - k * 0.45
        alert = s == 0.90
        box(6.6, y, 1.4, 0.34, risk(s), "#7f0000" if alert else "white", lw=2 if alert else 1)
        ax.text(7.3, y + 0.17, f"{s:.2f}", fontsize=9, ha="center", va="center",
                weight="bold" if alert else "normal",
                color="white" if s > 0.75 or s < 0.12 else "#222222")

    # colour scale
    ax.imshow([[i / 99 for i in range(100)]], cmap=risk, extent=(6.6, 8.0, 2.62, 2.74), aspect="auto")
    ax.text(6.6, 2.5, "0", fontsize=8, ha="center", va="top")
    ax.text(8.0, 2.5, "1", fontsize=8, ha="center", va="top")

    # alert
    arrow(8.1, 3.22, 8.65, 3.22, red)
    box(8.7, 2.97, 1.9, 0.5, "#fbe9e9", red, lw=1.5)
    ax.text(8.88, 3.22, "🚩", fontproperties=icons, fontsize=13, color=red, va="center")
    ax.text(9.2, 3.22, "alert", fontsize=10, weight="bold", color=red, va="center")
    ax.text(10.7, 3.22, "example: score 0.90", fontsize=8, color=red, va="center")

    # 2. explain, alerts only
    ax.text(0.3, 1.45, "Explain:\nalerts only", fontsize=12, weight="bold", color=green,
            va="center")
    steps = [
        ("📄", "evidence"),
        ("🔍", "retrieve patterns\n(RAG)"),
        ("💬", "write explanation\n(LLM)"),
        ("✔", "10 checks"),
        ("👤", "analyst"),
    ]
    xs = [3.0, 5.0, 7.0, 9.0, 11.0]
    for k, (icon, label) in enumerate(steps):
        x = xs[k]
        box(x - 0.75, 0.6, 1.5, 1.2, "#e7f4ea", green)
        ax.text(x, 1.42, icon, fontproperties=icons, fontsize=20, color=green,
                ha="center", va="center")
        ax.text(x, 0.92, label, fontsize=9, ha="center", va="center")
        if k < len(steps) - 1:
            arrow(x + 0.78, 1.2, xs[k + 1] - 0.78, 1.2, green)

    # alert -> evidence, through the gap between the rows
    ax.plot([9.65, 9.65, xs[0], xs[0]], [2.97, 2.2, 2.2, 1.95], color=red, lw=1.5)
    arrow(xs[0], 1.96, xs[0], 1.82, red)

    fig.tight_layout()
    fig.savefig(OUT / "how_it_works.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    data_over_time()
    model_comparison()
    how_it_works()
    print("saved:", *sorted(p.name for p in OUT.glob("*.png")))
