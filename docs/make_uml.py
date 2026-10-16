"""Class and sequence diagrams for the report."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
INK="#1a1a1a"; ACCENT="#1f4e79"; ACCENT2="#c55a11"; MUTED="#8c8c8c"; OK="#2e7d32"
plt.rcParams.update({"font.family":"DejaVu Sans","figure.dpi":200,"savefig.dpi":200,
                     "savefig.bbox":"tight"})

def save(fig,name):
    p=os.path.join(FIG,name); fig.savefig(p,facecolor="white"); plt.close(fig); print("wrote",p)

# ------------------------------------------------------------------ class diagram
def class_diagram():
    fig, ax = plt.subplots(figsize=(9.6, 6.2))
    ax.set_xlim(0,100); ax.set_ylim(0,100); ax.axis("off")

    def cls(x, y, w, name, attrs, ops, ec=ACCENT, fc="#eef3f8"):
        hh = 4.2
        ah = 3.0 + 2.6*len(attrs)
        oh = 3.0 + 2.6*len(ops)
        h = hh + ah + oh
        ax.add_patch(plt.Rectangle((x, y-h), w, h, facecolor=fc, edgecolor=ec, lw=1.2))
        ax.plot([x, x+w],[y-hh, y-hh], color=ec, lw=1.0)
        ax.plot([x, x+w],[y-hh-ah, y-hh-ah], color=ec, lw=1.0)
        ax.text(x+w/2, y-hh/2-0.3, name, ha="center", va="center", fontsize=7.6,
                fontweight="bold", color=INK)
        for i,a in enumerate(attrs):
            ax.text(x+1.2, y-hh-2.6-2.6*i, a, fontsize=6.1, va="center", color=INK)
        for i,o in enumerate(ops):
            ax.text(x+1.2, y-hh-ah-2.6-2.6*i, o, fontsize=6.1, va="center", color=INK)
        return (x, y, w, h)

    def link(x1,y1,x2,y2,label="",style="-|>",color=MUTED,ls="-",lx=None,ly=None):
        ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle=style,mutation_scale=9,
                                     color=color,lw=1.0,linestyle=ls))
        if label:
            ax.text(lx if lx is not None else (x1+x2)/2, ly if ly is not None else (y1+y2)/2+1.2,
                    label, fontsize=5.8, color=color, ha="center", style="italic")

    cls(1, 99, 21, "Claim",
        ["- claim_id: str","- agent: str","- text: str","- cited_chunk_ids: List[str]",
         "- numbers: List[NumericFact]","- revision: int"],
        ["+ new(agent, section, text)"])
    cls(53, 99, 21, "Chunk",
        ["- chunk_id: str","- text: str","- source: str","- doc_type: str",
         "- ticker: str","- published: str"],
        ["+ make_id(source, text)"])
    cls(27, 99, 22, "Verdict",
        ["- confidence: float","- numeric_support: float","- semantic_support: float",
         "- citation_consistent: bool","- unsupported_numbers: List","- status: str"],
        ["+ to_dict()"])
    cls(79, 99, 20, "Report",
        ["- query: str","- ticker: str","- sections: Dict","- verdicts: Dict",
         "- timings: Dict"],
        ["+ grounding_rate"])

    cls(1, 56, 23, "Agent  «abstract»",
        ["- store: EvidenceStore","- name: str","- section: str"],
        ["+ cite(query, ticker, k)","+ claim(text, cited)","+ run(ticker, context)",
         "+ revise(claim, feedback)"])
    cls(56, 56, 22, "EvidenceStore",
        ["- chunks: Dict[str, Chunk]","- embedder: Embedder","- backend: str"],
        ["+ index(chunks)","+ query(text, k, ticker)","+ get(chunk_id)"])
    cls(29, 56, 22, "VerificationAgent",
        ["- k: int   - tolerance: τ","- threshold: θ   - max_retries: R",
         "- numeric_veto: bool","- entity_check: bool"],
        ["+ verify(claim)","+ verify_with_revision(...)"], ec=ACCENT2, fc="#fdf0e6")
    cls(83, 56, 16, "Router",
        ["- clf: OvR LogReg"],
        ["+ fit(data)","+ route(query, θ)","+ scores(query)"])

    cls(1, 20, 17, "Fundamental\nAgent", [], ["+ run()","+ revise()"])
    cls(21, 20, 15, "Technical\nAgent", [], ["+ run()"])
    cls(39, 20, 15, "Sentiment\nAgent", [], ["+ run()"])
    cls(57, 20, 14, "Risk\nAgent", [], ["+ run()"])
    cls(74, 20, 25, "ReportGenerator",
        [], ["+ render_markdown(report)","+ export_pdf(md, path)"])

    # generalisation — single shared trunk, standard UML style
    TRUNK = 24.0
    for x in (9.5, 28.5, 46.5, 64):
        ax.plot([x, x], [20, TRUNK], color=ACCENT, lw=1.0)
    ax.plot([9.5, 64], [TRUNK, TRUNK], color=ACCENT, lw=1.0)
    link(12, TRUNK, 12, 27.2, style="-|>", color=ACCENT)
    ax.text(13.5, 25.4, "generalisation", fontsize=5.8, color=ACCENT, style="italic")

    link(29, 44, 24, 44, "returns claim for revision", style="-|>", color=ACCENT2,
         ls="--", lx=26.5, ly=45.4)
    link(51, 44, 56, 44, "re-retrieves from", style="-|>", lx=53.5, ly=45.4)
    link(40, 56, 40, 74, "produces", style="-|>", lx=43.5, ly=65)
    link(67, 56, 67, 74, "holds", style="-|>", lx=70, ly=65)
    link(88, 56, 88, 74, "informs", style="-|>", lx=91.5, ly=65)
    link(12, 56, 12, 74, "emits", style="-|>", lx=8.5, ly=65)

    ax.text(50, 2, "Fig. 5  Class diagram (principal classes)", ha="center",
            fontsize=8.4, fontweight="bold")
    save(fig, "fig5_class.png")


# --------------------------------------------------------------- sequence diagram
def sequence_diagram():
    fig, ax = plt.subplots(figsize=(9.6, 6.0))
    ax.set_xlim(0,100); ax.set_ylim(0,100); ax.axis("off")

    lifelines = [("User", 6), ("Router", 21), ("Data Collection", 38),
                 ("Analysis Agent", 56), ("Evidence Store", 74), ("Verification\nAgent", 90)]
    for name, x in lifelines:
        ax.add_patch(plt.Rectangle((x-7, 92), 14, 6, facecolor="#eef3f8",
                                   edgecolor=ACCENT, lw=1.1))
        ax.text(x, 95, name, ha="center", va="center", fontsize=6.8, fontweight="bold")
        ax.plot([x, x], [92, 6], color=MUTED, lw=0.8, ls=(0,(4,3)))

    def msg(x1, x2, y, label, color=MUTED, ls="-", up=1.4):
        ax.add_patch(FancyArrowPatch((x1, y),(x2, y), arrowstyle="-|>", mutation_scale=9,
                                     color=color, lw=1.0, linestyle=ls))
        ax.text((x1+x2)/2, y+up, label, ha="center", fontsize=6.0, color=color,
                style="italic")

    def selfmsg(x, y, label, color=MUTED):
        ax.add_patch(FancyArrowPatch((x, y),(x+7, y), arrowstyle="-", color=color, lw=1.0))
        ax.add_patch(FancyArrowPatch((x+7, y),(x+7, y-3.2), arrowstyle="-", color=color, lw=1.0))
        ax.add_patch(FancyArrowPatch((x+7, y-3.2),(x, y-3.2), arrowstyle="-|>",
                                     mutation_scale=9, color=color, lw=1.0))
        ax.text(x+8.5, y-1.6, label, fontsize=6.0, color=color, style="italic", va="center")

    msg(6, 21, 88, "research question")
    msg(21, 38, 82, "required agents → ingest")
    msg(38, 74, 76, "chunk, embed, index (with provenance)")
    msg(38, 56, 70, "normalised data + computed indicators")
    msg(56, 74, 64, "retrieve evidence")
    msg(74, 56, 58, "top-k chunks", ls="--")
    selfmsg(56, 52, "compute values, form claim")
    msg(56, 90, 45, "claim + asserted citations")
    msg(90, 74, 39, "INDEPENDENT re-retrieval", color=OK)
    msg(74, 90, 33, "top-k chunks (citation ignored)", color=OK, ls="--")
    selfmsg(90, 29, "numeric veto · semantic ·\ncitation · entity binding", color=ACCENT2)

    ax.add_patch(plt.Rectangle((50, 14), 48, 8.5, facecolor="none",
                               edgecolor=ACCENT2, lw=1.0, ls="--"))
    ax.text(51.5, 21, "alt  [confidence < θ  and  retries < R]", fontsize=6.0,
            color=ACCENT2, fontweight="bold")
    msg(90, 56, 17, "revise: reason for failure", color=ACCENT2)

    msg(90, 6, 10, "verified claims only, each with confidence + sources", color=OK)
    ax.text(50, 2.5, "Fig. 6  Sequence diagram — one research query, "
                     "including the bounded revision loop",
            ha="center", fontsize=8.4, fontweight="bold")
    save(fig, "fig6_sequence.png")


if __name__ == "__main__":
    class_diagram()
    sequence_diagram()
