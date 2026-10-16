"""docs/make_figures.py — every figure used in the report and the deck."""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "docs", "figures")
os.makedirs(FIG, exist_ok=True)
M = json.load(open(os.path.join(ROOT, "results", "metrics.json")))

INK = "#1a1a1a"
ACCENT = "#1f4e79"
ACCENT2 = "#c55a11"
MUTED = "#8c8c8c"
GRID = "#d9d9d9"
OK = "#2e7d32"
BAD = "#b3261e"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": "#bfbfbf", "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def save(fig, name):
    path = os.path.join(FIG, name)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print("wrote", path)


# ---------------------------------------------------------------- Fig: architecture
def fig_architecture():
    fig, ax = plt.subplots(figsize=(9.2, 6.4))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    def box(x, y, w, h, label, sub="", fc="#eef3f8", ec=ACCENT, lw=1.2, fs=8.5, bold=True):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.5",
                                    facecolor=fc, edgecolor=ec, linewidth=lw))
        ax.text(x + w / 2, y + h / 2 + (1.6 if sub else 0), label, ha="center", va="center",
                fontsize=fs, fontweight="bold" if bold else "normal", color=INK)
        if sub:
            ax.text(x + w / 2, y + h / 2 - 2.6, sub, ha="center", va="center",
                    fontsize=6.8, color=MUTED, style="italic")

    def arrow(x1, y1, x2, y2, color=MUTED, style="-|>", lw=1.1, ls="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                     mutation_scale=11, color=color, linewidth=lw,
                                     linestyle=ls, shrinkA=0, shrinkB=0))

    box(3, 86, 20, 9, "User query", "natural language", fc="#f4f4f4", ec=MUTED)
    box(33, 85.5, 34, 10.5, "Orchestrator / Router Agent",
        "feature classifier · 94.7% exact match · 60.3% fewer invocations")
    arrow(23, 90.5, 33, 90.5)

    agents = [("Fundamental", "P/E · ROE · D/E · margins"),
              ("Technical", "RSI · MACD · SMA · bands"),
              ("Sentiment", "FinBERT scoring"),
              ("Risk", "volatility · drawdown · VaR")]
    xs = [2, 26, 50, 74]
    for x, (nm, sub) in zip(xs, agents):
        box(x, 64, 22, 11, nm + " Agent", sub)
        arrow(50, 85.5, x + 11, 75)

    box(20, 44, 60, 12, "VERIFICATION AGENT",
        "re-retrieve · numeric veto · semantic · citation · entity binding",
        fc="#fdf0e6", ec=ACCENT2, lw=2.0, fs=10)
    for x in xs:
        arrow(x + 11, 64, 50, 56, color="#9aa5b1")
    ax.text(50, 59.5, "claims + asserted citations", ha="center", fontsize=6.8,
            color=MUTED, style="italic")

    # revision loop
    arrow(20, 47, 6, 47, color=ACCENT2, ls="--")
    arrow(6, 47, 6, 69.5, color=ACCENT2, ls="--")
    arrow(6, 69.5, 12, 69.5, color=ACCENT2, ls="--")
    ax.text(9.5, 57.5, "confidence < θ\nbounded revision\n(≤ R retries)", fontsize=6.6,
            color=ACCENT2, ha="left", va="center", style="italic", linespacing=1.4)

    box(28, 26, 44, 10, "Report Generator Agent",
        "cited report · per-claim confidence · mandatory disclaimer")
    arrow(50, 44, 50, 36, color=ACCENT2, lw=1.5)
    ax.text(51.5, 40, "verified only", fontsize=6.8, color=ACCENT2, style="italic")

    box(28, 8, 44, 10, "Evidence store — ChromaDB",
        "filings · news · market data · derived records", fc="#f0f4ef", ec=OK)
    arrow(50, 26, 50, 18, color=MUTED, style="<|-|>")
    ax.text(74, 22, "independent\nre-retrieval", fontsize=6.6, color=OK,
            ha="center", style="italic")
    arrow(72, 13, 84, 13, color=OK, style="<|-")
    arrow(84, 13, 84, 50, color=OK)
    arrow(84, 50, 80, 50, color=OK, style="-|>")

    box(78, 84, 20, 9, "Data Collection\nAgent", fc="#f0f4ef", ec=OK, fs=8)
    arrow(88, 84, 88, 18, color=OK, ls=":")

    ax.text(50, 2, "Fig. 2  System architecture. Nothing reaches the user without "
                   "passing the Verification Agent.",
            ha="center", fontsize=7.2, color=MUTED)
    save(fig, "fig2_architecture.png")


# --------------------------------------------------------------------- Fig: DFD
def fig_dfd():
    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    ax.set_xlim(0, 100); ax.set_ylim(0, 62); ax.axis("off")

    R = 7.0

    def proc(x, y, label, num):
        ax.add_patch(plt.Circle((x, y), R, facecolor="#eef3f8",
                                edgecolor=ACCENT, linewidth=1.3))
        ax.text(x, y + 2.4, num, ha="center", fontsize=6.8, color=MUTED)
        ax.text(x, y - 1.6, label, ha="center", va="center", fontsize=7.2,
                fontweight="bold", linespacing=1.25)

    def ext(x, y, w, h, label):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor="#f4f4f4",
                                   edgecolor=INK, linewidth=1.1))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=7.2,
                fontweight="bold", linespacing=1.3)

    def datastore(x, y, w, label):
        ax.add_patch(plt.Rectangle((x, y), w, 4.6, facecolor="#f7f9fb",
                                   edgecolor=INK, linewidth=1.0))
        ax.plot([x + 7.5, x + 7.5], [y, y + 4.6], color=INK, lw=1.0)
        ax.text(x + 3.7, y + 2.3, label.split("|")[0], ha="center", va="center",
                fontsize=6.8, fontweight="bold")
        ax.text(x + 9, y + 2.3, label.split("|")[1], va="center", fontsize=6.6)

    def flow(x1, y1, x2, y2, label="", lx=None, ly=None, color=MUTED, ls="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=10, color=color, lw=1.05,
                                     linestyle=ls))
        if label:
            ax.text(lx if lx is not None else (x1 + x2) / 2,
                    ly if ly is not None else (y1 + y2) / 2 + 1.6,
                    label, ha="center", fontsize=6.3, color=color, style="italic",
                    linespacing=1.3)

    # --- entities -------------------------------------------------------
    ext(1, 40, 15, 9, "Investor /\nAnalyst")
    ext(1, 10, 15, 9, "Free data\nsources")
    proc(31, 50, "Route\nquery", "1.0")
    proc(55, 50, "Analyse", "3.0")
    proc(80, 50, "Verify\nclaims", "4.0")
    proc(31, 20, "Collect &\nindex", "2.0")
    proc(80, 20, "Generate\nreport", "5.0")
    datastore(44, 22, 26, "D1|Evidence store")
    datastore(44, 5, 26, "D2|Verdict log")

    # --- flows ----------------------------------------------------------
    flow(16, 46.5, 24.6, 50, "query", 19.5, 50.4)
    flow(38, 50, 48, 50, "agents\nselected", 43, 51.6)
    flow(62, 50, 73, 50, "claims +\ncitations", 67.5, 51.6)
    flow(80, 43, 80, 27, "verified\nclaims", 87.5, 35)
    flow(73.5, 16, 16.5, 42, "cited report +\nper-claim confidence", 30, 36.5)
    flow(16, 14.5, 24.5, 17, "prices · filings · news", 30, 11.5)
    flow(35, 24.5, 46, 24.5, "index", 40.5, 25.6)
    flow(57, 26.6, 55, 43, "retrieve", 51.4, 35)
    flow(69, 26.5, 76, 43.5, "independent\nre-retrieval", 74.5, 32.5, color=OK)
    flow(76, 14.5, 68, 8.5, "verdicts", 76, 9.5)

    ax.text(50, 59.5, "Fig. 3  Level-1 Data Flow Diagram", ha="center",
            fontsize=8.6, fontweight="bold")
    save(fig, "fig3_dfd.png")


# ---------------------------------------------------------------- Fig: use case
def fig_usecase():
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 62); ax.axis("off")

    def actor(x, y, name):
        ax.plot([x], [y + 4], marker="o", ms=6, color=INK)
        ax.plot([x, x], [y, y + 3.4], color=INK, lw=1.2)
        ax.plot([x - 2.6, x + 2.6], [y + 2.4, y + 2.4], color=INK, lw=1.2)
        ax.plot([x, x - 2.4], [y, y - 3.4], color=INK, lw=1.2)
        ax.plot([x, x + 2.4], [y, y - 3.4], color=INK, lw=1.2)
        ax.text(x, y - 6, name, ha="center", fontsize=7.5, fontweight="bold")

    ax.add_patch(plt.Rectangle((22, 4), 56, 52, facecolor="none",
                               edgecolor="#bfbfbf", linewidth=1.0))
    ax.text(50, 57.4, "Multi-Agent Financial Research Assistant", ha="center",
            fontsize=7.6, style="italic", color=MUTED)

    cases = [("Submit research query", 50), ("Receive cited report", 43),
             ("Inspect source evidence", 36), ("Adjust confidence threshold", 29),
             ("Verify an arbitrary claim", 22), ("Export report to PDF", 15),
             ("Refresh cached market data", 8)]
    for label, y in cases:
        ax.add_patch(plt.matplotlib.patches.Ellipse((50, y), 42, 5.4,
                                                    facecolor="#eef3f8", edgecolor=ACCENT,
                                                    linewidth=1.0))
        ax.text(50, y, label, ha="center", va="center", fontsize=7.2)

    actor(10, 34, "Investor /\nAnalyst")
    actor(90, 20, "Scheduler\n(GitHub Actions)")

    for y in [50, 43, 36, 29, 22, 15]:
        ax.plot([14, 29], [34, y], color=MUTED, lw=0.7)
    ax.plot([86, 71], [20, 8], color=MUTED, lw=0.7)

    ax.text(50, 1, "Fig. 4  Use case diagram", ha="center", fontsize=8.5,
            fontweight="bold")
    save(fig, "fig4_usecase.png")


# ------------------------------------------------------------------ Fig: Gantt
def fig_gantt():
    tasks = [
        ("Scoping & literature review", 0, 2, "#9db4c8"),
        ("Data pipeline & connectors", 2, 3, "#9db4c8"),
        ("Specialist agents", 5, 3, "#9db4c8"),
        ("RAG grounding layer", 8, 2, "#9db4c8"),
        ("Verification Agent", 9, 3, ACCENT2),
        ("Router classifier", 11, 2, "#9db4c8"),
        ("API + Streamlit UI", 12, 2, "#9db4c8"),
        ("Evaluation harness (Review 2)", 14, 2, ACCENT2),
        ("FinBERT + live-data evaluation", 16, 3, "#cfcfcf"),
        ("Deployment (HF Spaces)", 18, 2, "#cfcfcf"),
        ("Paper + provisional patent draft", 19, 3, "#cfcfcf"),
        ("Final report & defence", 21, 2, "#cfcfcf"),
    ]
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    for i, (name, start, dur, color) in enumerate(tasks):
        y = len(tasks) - i
        ax.barh(y, dur, left=start, height=0.58, color=color,
                edgecolor="white", linewidth=0.8)
        ax.text(start + dur + 0.25, y, f"W{start + 1}–W{start + dur}",
                va="center", fontsize=6.4, color=MUTED)
    ax.set_yticks([len(tasks) - i for i in range(len(tasks))])
    ax.set_yticklabels([t[0] for t in tasks], fontsize=7.4)
    ax.set_xlabel("Project week", fontsize=8)
    ax.set_xlim(0, 26)
    ax.axvline(16, color=BAD, lw=1.2, ls="--")
    ax.text(16.2, len(tasks) + 0.8, "Review 2 (today)", color=BAD, fontsize=7,
            fontweight="bold")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Fig. 1  Project plan", fontsize=9, fontweight="bold", loc="left")
    save(fig, "fig1_gantt.png")


# ------------------------------------------------------------- Fig: ablation
def fig_ablation():
    abl = M["ablation"]
    labels = ["Full\nmechanism", "− numeric\nveto", "− entity\nbinding",
              "Semantic\nonly", "Citation\ntrust only"]
    keys = ["full mechanism", "no numeric veto (additive scoring)",
            "no entity binding", "semantic similarity only",
            "citation trust only (no re-retrieval check)"]
    vals = [abl[k]["catch_rate"] * 100 for k in keys]
    colors = [ACCENT2] + ["#9db4c8"] * 2 + ["#cfcfcf"] * 2

    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    bars = ax.bar(labels, vals, color=colors, edgecolor="white", linewidth=1.0, width=0.62)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 2.2, f"{v:.1f}%", ha="center",
                fontsize=8, fontweight="bold")
    ax.set_ylabel("Hallucination catch rate (%)", fontsize=8)
    ax.set_ylim(0, 108)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Ablation: what each component contributes", fontsize=9,
                 fontweight="bold", loc="left")
    save(fig, "fig5_ablation.png")


# ------------------------------------------------------ Fig: catch by type
def fig_by_type():
    bp = M["hallucination"]["by_perturbation"]
    order = ["numeric_drift", "numeric_swap", "fabricated_detail", "entity_swap"]
    names = ["Numeric drift\n(±8–18%)", "Numeric swap\n(2×, 0.45×)",
             "Fabricated\ndetail", "Entity\nswap"]
    vals = [bp[k]["rate"] * 100 for k in order if k in bp]

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    bars = ax.bar(names, vals, color=[ACCENT] * 3 + [ACCENT2], edgecolor="white",
                  linewidth=1.0, width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.1f}%", ha="center",
                fontsize=8, fontweight="bold")
    ax.axhline(M["hallucination"]["catch_rate_recall"] * 100, color=MUTED, ls="--", lw=1.0)
    ax.text(3.42, M["hallucination"]["catch_rate_recall"] * 100 + 2.5, "overall",
            fontsize=6.6, color=MUTED, ha="right")
    ax.set_ylabel("Catch rate (%)", fontsize=8)
    ax.set_ylim(0, 112)
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_title("Detection by hallucination type", fontsize=9, fontweight="bold",
                 loc="left")
    save(fig, "fig6_by_type.png")


# ------------------------------------------------------ Fig: threshold sweep
def fig_threshold():
    rows = M["threshold_sweep"]
    theta = [r["theta"] for r in rows]
    catch = [r["catch_rate"] * 100 for r in rows]
    retain = [r["genuine_claims_retained"] * 100 for r in rows]

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.plot(theta, catch, marker="o", ms=3.4, color=ACCENT2, lw=1.6,
            label="Hallucinations caught")
    ax.plot(theta, retain, marker="s", ms=3.2, color=OK, lw=1.6,
            label="Genuine claims retained")
    ax.axvspan(0.50, 0.75, color="#f2f6fa", zorder=0)
    ax.axvline(0.60, color=INK, lw=1.0, ls="--")
    ax.text(0.605, 46, "operating point\nθ = 0.60", fontsize=6.8, color=INK)
    ax.text(0.625, 12, "stable plateau", fontsize=6.6, color=MUTED, ha="center")
    ax.set_xlabel("Confidence threshold θ", fontsize=8)
    ax.set_ylabel("%", fontsize=8)
    ax.set_ylim(-4, 108)
    ax.legend(fontsize=7, frameon=False, loc="center left")
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_title("Threshold sensitivity", fontsize=9, fontweight="bold", loc="left")
    save(fig, "fig7_threshold.png")


# ------------------------------------------------------------ Fig: routing
def fig_routing():
    r = M["router"]
    lat = M["cost_latency"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 3.2))

    a1.bar(["Exact\nmatch", "Label\nrecall", "Mean\nJaccard"],
           [r["exact_match_accuracy"] * 100, r["label_recall"] * 100,
            r["mean_jaccard"] * 100],
           color=[ACCENT, ACCENT, "#9db4c8"], width=0.55, edgecolor="white")
    for i, v in enumerate([r["exact_match_accuracy"], r["label_recall"],
                           r["mean_jaccard"]]):
        a1.text(i, v * 100 + 1.6, f"{v * 100:.1f}%", ha="center", fontsize=7.6,
                fontweight="bold")
    a1.set_ylim(0, 112); a1.set_ylabel("%", fontsize=8)
    a1.grid(axis="y", color=GRID, lw=0.6); a1.set_axisbelow(True)
    a1.set_title("Router accuracy (leave-one-out, n=75)", fontsize=8,
                 fontweight="bold", loc="left")

    a2.bar(["No routing\n(all 4 agents)", "With router"],
           [lat["unrouted_mean_total_s"] * 1000, lat["routed_mean_total_s"] * 1000],
           color=["#cfcfcf", ACCENT2], width=0.5, edgecolor="white")
    a2.text(0, lat["unrouted_mean_total_s"] * 1000 + 1.4,
            f"{lat['unrouted_mean_total_s'] * 1000:.0f} ms", ha="center",
            fontsize=7.6, fontweight="bold")
    a2.text(1, lat["routed_mean_total_s"] * 1000 + 1.4,
            f"{lat['routed_mean_total_s'] * 1000:.0f} ms", ha="center",
            fontsize=7.6, fontweight="bold")
    a2.set_ylabel("Mean latency per report (ms)", fontsize=8)
    a2.grid(axis="y", color=GRID, lw=0.6); a2.set_axisbelow(True)
    a2.set_title(f"−{lat['latency_reduction_pct']:.0f}% latency, "
                 f"−{r['agent_invocation_reduction_pct']:.0f}% invocations",
                 fontsize=8, fontweight="bold", loc="left")
    fig.tight_layout()
    save(fig, "fig8_routing.png")


# ------------------------------------------------------- Fig: confidence split
def fig_confidence():
    import csv
    path = os.path.join(ROOT, "results", "verification_detail.csv")
    gen, per = [], []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            (gen if row["gold"] == "supported" else per).append(float(row["confidence"]))

    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    bins = [i * 0.05 for i in range(21)]
    ax.hist(per, bins=bins, color=ACCENT2, alpha=0.85, label="Perturbed claims (gold: unsupported)")
    ax.hist(gen, bins=bins, color=ACCENT, alpha=0.85, label="Genuine claims (gold: supported)")
    ax.axvline(0.60, color=INK, ls="--", lw=1.2)
    ax.text(0.615, ax.get_ylim()[1] * 0.86, "θ = 0.60", fontsize=7.4, fontweight="bold")
    ax.set_xlabel("Assigned confidence", fontsize=8)
    ax.set_ylabel("Claims", fontsize=8)
    ax.legend(fontsize=6.8, frameon=False, loc="upper center")
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_title("Confidence separates supported from unsupported claims",
                 fontsize=9, fontweight="bold", loc="left")
    save(fig, "fig9_confidence.png")


if __name__ == "__main__":
    fig_gantt()
    fig_architecture()
    fig_dfd()
    fig_usecase()
    fig_ablation()
    fig_by_type()
    fig_threshold()
    fig_routing()
    fig_confidence()


# --------------------------------------------------- Fig: live verification demo
def fig_demo():
    """Render actual system output: one genuine claim and two tampered ones."""
    import sys
    sys.path.insert(0, ROOT)
    from agents.validator import VerificationAgent
    from core import Claim
    from pipeline import ResearchPipeline

    pipe = ResearchPipeline(prefer_live=False)
    store = pipe._prepare("TCS.NS")["store"]
    v = VerificationAgent(store)

    probes = [
        ("Operating margin for the year was 24.3%, a movement of -70 bps against "
         "the prior year.", "unmodified agent output"),
        ("Operating margin for the year was 27.9%, a movement of -70 bps against "
         "the prior year.", "one digit changed"),
        ("Return on equity for the year was 51.2%. The board also approved a "
         "share buyback of ₹18,000 cr.", "fabricated sentence appended"),
    ]

    rows = []
    for text, note in probes:
        c = Claim.new("demo", "demo", text)
        c.cited_chunk_ids = [ch.chunk_id for ch, _ in store.query(text, k=3, ticker="TCS.NS")]
        rows.append((text, note, v.verify(c)))

    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    y = 92
    for text, note, verdict in rows:
        passed = verdict.status == "verified"
        col = OK if passed else BAD
        bg = "#f1f7f1" if passed else "#fdf1f0"
        ax.add_patch(FancyBboxPatch((2, y - 25), 96, 27,
                                    boxstyle="round,pad=0.4,rounding_size=1.2",
                                    facecolor=bg, edgecolor=col, linewidth=1.3))
        ax.text(4.5, y - 2.5, ("ACCEPTED" if passed else "WITHHELD"),
                fontsize=8.5, fontweight="bold", color=col)
        ax.text(21, y - 2.5, f"· {note}", fontsize=7, color=MUTED, style="italic")
        ax.text(78, y - 2.5, f"confidence {verdict.confidence:.2f}",
                fontsize=8, fontweight="bold", color=col)

        wrapped = []
        line = ""
        for word in text.split():
            if len(line) + len(word) > 96:
                wrapped.append(line); line = word
            else:
                line = (line + " " + word).strip()
        wrapped.append(line)
        ax.text(4.5, y - 9, "\n".join(wrapped), fontsize=7.6, va="top",
                linespacing=1.5)

        detail = (f"numeric support {verdict.numeric_support:.2f}  ·  "
                  f"semantic {verdict.semantic_support:.2f}  ·  "
                  f"citation {'ok' if verdict.citation_consistent else 'not corroborated'}")
        if verdict.reason:
            detail += f"\n{verdict.reason}"
        ax.text(4.5, y - 16, detail, fontsize=6.6, color=col, va="top",
                style="italic", linespacing=1.5)
        y -= 31

    ax.text(50, 1, "Actual system output — the middle claim differs from the first by "
                   "a single digit.", ha="center", fontsize=7.2, color=MUTED)
    save(fig, "fig10_demo.png")


if __name__ == "__main__":
    fig_demo()
