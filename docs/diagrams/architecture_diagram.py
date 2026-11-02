"""Draw docs/architecture_diagram.png, the static copy of the system overview in section 3.

    uv run python docs/diagrams/architecture_diagram.py

Keep it in step with the mermaid diagram in docs/architecture.md.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon  # noqa: E402

INK = "#1F2933"
LINE = "#4B5563"
ORCH = ("#E4E9EF", "#52606D")
AGENT = ("#D9E8F6", "#2F6DA3")
GATE = ("#FBE8B8", "#B7791F")
LLM = ("#D5EEEC", "#2C7A7B")


def box(ax, x, y, w, h, text, colours, size=11.5, bold=False):
    face, edge = colours
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.3,rounding_size=1.6",
            fc=face,
            ec=edge,
            lw=1.8,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=size,
        color=INK,
        weight="bold" if bold else "normal",
        linespacing=1.4,
    )


def arrow(ax, points, label=None, label_at=None, size=10):
    """A polyline with an arrowhead on its last segment."""
    for a, b in zip(points[:-2], points[1:-1], strict=False):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=LINE, lw=1.8, solid_capstyle="round")
    ax.add_patch(
        FancyArrowPatch(
            points[-2],
            points[-1],
            arrowstyle="-|>",
            mutation_scale=18,
            lw=1.8,
            color=LINE,
            shrinkA=0,
            shrinkB=0,
        )
    )
    if label:
        ax.text(
            *label_at,
            label,
            fontsize=size,
            color=INK,
            ha="center",
            va="center",
            linespacing=1.3,
            bbox={"fc": "white", "ec": "none", "pad": 1.5},
        )


TOP = 69  # top edge of the agent cards


def card(ax, x, title, code, llm):
    """An agent: what the code does on top, its one bounded LLM step underneath."""
    w = 23
    box(ax, x, TOP - 20, w, 20, "", AGENT)
    ax.text(x + w / 2, TOP - 4, title, ha="center", va="center", fontsize=10.6, weight="bold")
    ax.text(x + w / 2, TOP - 12.5, code, ha="center", va="center", fontsize=9.2, linespacing=1.4)
    box(ax, x, TOP - 34, w, 12, llm, LLM, size=9.2)
    arrow(ax, [(x + w / 2, TOP - 34.6), (x + w / 2, 27.4)])


def draw(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 160)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.text(
        2, 96.3, "Agentic GAN framework: system overview", fontsize=19, weight="bold", color=INK
    )
    ax.text(
        2,
        92.6,
        "v0.3, 29 October 2026. Code decides every route, verdict and number. Each agent may add "
        "one bounded LLM step.",
        fontsize=11.5,
        color=LINE,
    )

    # orchestration layer
    box(
        ax,
        2,
        78,
        112,
        12,
        "Orchestration layer (code)\n"
        "run config + dataset  >  S0 validate and split  >  stage order, retries, gate routing,\n"
        "rule lock, volume rule.  Sub-agents never call each other.\n"
        "state.json   manifest.json   decisions.jsonl   run folder: report, logs, artifacts",
        ORCH,
        size=10.2,
    )
    box(
        ax,
        118,
        78,
        40,
        12,
        "LLM: writes the run narrative\nfrom the decision log.\nIt cannot change a route.",
        LLM,
        size=10.2,
    )

    # agents, left to right, with the gate routing between quality and training
    xs = [2, 28.5, 55, 92, 135]
    card(
        ax,
        xs[0],
        "Pre-processing\nAgent",
        "profile and clean;\nfit on train rows only;\nraw + baseline files",
        "LLM: picks each\ncolumn's treatment\nfrom the allowed table",
    )
    card(
        ax,
        xs[1],
        "Model Selection\nAgent",
        "rule picks the generator;\ntrain CTGAN; one set\nper fill level",
        "LLM: explains the rule;\non a fail, proposes\nCTGAN settings inside\ngan_bounds",
    )
    card(
        ax,
        xs[2],
        "Quality Evaluation\nAgent",
        "fidelity, privacy and\nutility checks decide\nthe verdict per level",
        "LLM: diagnoses a FAIL\nwith allowed fixes;\nthe verdict stays code",
    )
    card(
        ax,
        xs[3],
        "Model Training\nAgent",
        "4 models on 4 arms (raw,\nbaseline, smote, gan),\nall seeds, frozen cutoffs",
        "LLM: plans arms and\nlevels, writes the\nrun summary",
    )
    card(
        ax,
        xs[4],
        "Evaluation\nAgent",
        "metrics, arm comparison,\nreliable gain, segments,\nSHAP, number check",
        "LLM: writes the\nnarrative; every number\nmust be in the results",
    )
    box(
        ax,
        118.5,
        TOP - 20,
        13,
        20,
        "Volume rule\n(code)\nsmallest fill\nlevel with a\nreliable gain,\nelse none",
        ORCH,
        size=9.2,
    )
    box(ax, 118.5, TOP - 34, 13, 12, "no LLM\nstep", ORCH, size=9.2)

    mid = TOP - 10
    for left, right in ((25.4, 28.1), (51.9, 54.6), (115.4, 118.1), (131.9, 134.6)):
        arrow(ax, [(left, mid), (right, mid)])
    arrow(ax, [(78.4, TOP - 6), (91.6, TOP - 6)])
    arrow(ax, [(78.4, TOP - 14), (91.6, TOP - 14)])
    ax.text(85, TOP - 2.6, "PASS: passing\nfill levels", fontsize=9, ha="center", va="center")
    ax.text(
        85,
        TOP - 19,
        "FAIL, attempts\nexhausted:\nno GAN arm",
        fontsize=9,
        ha="center",
        va="center",
        linespacing=1.25,
    )

    # retry loop above the quality gate
    arrow(ax, [(66.5, TOP + 0.4), (66.5, TOP + 3.8), (40, TOP + 3.8), (40, TOP + 0.6)])
    ax.text(
        53.2,
        TOP + 5.8,
        "FAIL, attempts left: retry with the failed report",
        fontsize=9.6,
        color=INK,
        ha="center",
        va="center",
    )

    # shared LLM runtime
    box(
        ax,
        2,
        12,
        156,
        15,
        "Shared LLM runtime: src/agentic_gan/llm (run_llm_step), used by all six LLM steps\n"
        "Gemini on Vertex AI, temperature 0; the answer must fit the step's Pydantic schema\n"
        "Guardrails:  allowed options and bounds   |   rule-hash lock   |   "
        "one retry, then a deterministic fallback   |   every step logged",
        LLM,
        size=10.4,
    )

    # legend
    for x, colours, label in (
        (2, AGENT, "agent: code does the work"),
        (40, LLM, "bounded LLM step"),
        (70, ORCH, "orchestration, code only"),
    ):
        ax.add_patch(
            FancyBboxPatch(
                (x, 4.2),
                3.6,
                2.4,
                boxstyle="round,pad=0.1,rounding_size=0.6",
                fc=colours[0],
                ec=colours[1],
                lw=1.4,
            )
        )
        ax.text(x + 5.4, 5.4, label, fontsize=10, color=INK, va="center")

    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def diamond(ax, cx, cy, hw, hh, text):
    ax.add_patch(
        Polygon(
            [(cx - hw, cy), (cx, cy + hh), (cx + hw, cy), (cx, cy - hh)],
            closed=True,
            fc=GATE[0],
            ec=GATE[1],
            lw=1.8,
        )
    )
    ax.text(cx, cy, text, ha="center", va="center", fontsize=11, color=INK, linespacing=1.3)


def note(ax, x, y, text, ha="center"):
    ax.text(x, y, text, fontsize=10.5, color=INK, ha=ha, va="center", linespacing=1.3)


def draw_volume_loop(path: Path) -> None:
    """The volume loop of section 4.1, from slide 6 of the 9 October deck."""
    fig, ax = plt.subplots(figsize=(16, 6.2))
    ax.set_xlim(0, 160)
    ax.set_ylim(0, 62)
    ax.axis("off")
    ax.text(
        2,
        58,
        "The volume loop: how much synthetic data to add",
        fontsize=19,
        weight="bold",
        color=INK,
    )

    y = 38
    box(ax, 2, y - 7, 21, 14, "Profile training\ndata, set the\nreal-only baseline", AGENT)
    box(ax, 28, y - 7, 22, 14, "Choose volumes,\ngenerate samples\n(one per fill level)", AGENT)
    diamond(ax, 66, y, 11, 10, "Quality\npasses?")
    box(ax, 82, y - 7, 25, 14, "Test accepted volumes\non real validation\ndata", AGENT)
    diamond(ax, 122, y, 11, 10, "Reliable\ngain?")
    box(ax, 138, y - 7, 20, 14, "Select smallest\nvolume near the\nbest score", AGENT)
    box(ax, 38, 6, 22, 12, "Reject batch,\nrevise generator", ORCH)
    box(ax, 100, 6, 36, 12, "Use real data only,\nadd 0 rows", ORCH)

    arrow(ax, [(23.4, y), (27.4, y)])
    arrow(ax, [(50.6, y), (54.9, y)])
    arrow(ax, [(77.1, y), (81.4, y)], "yes", (79.3, y + 3), size=10.5)
    arrow(ax, [(107.6, y), (110.9, y)])
    arrow(ax, [(133.1, y), (137.4, y)], "yes", (135.3, y + 3), size=10.5)

    # quality fails: revise and try again, or give up when the budget is used
    arrow(ax, [(66, y - 10), (66, 12), (60.6, 12)])
    note(ax, 62.5, 27, "no, within\nbudget", ha="right")
    arrow(ax, [(49, 18.6), (49, y - 7.6)])
    arrow(ax, [(66, 12), (66, 3), (118, 3), (118, 5.4)])
    note(ax, 92, 0.6, "no, budget exhausted", ha="center")

    # no reliable gain
    arrow(ax, [(122, y - 10), (122, 18.6)])
    note(ax, 124, 27, "no", ha="left")

    ax.text(
        2,
        52,
        'Orchestrator logic. The rule for "reliable gain" is in section 4.1 of '
        "docs/architecture.md (proposed, issue #36).",
        fontsize=11.5,
        color=LINE,
    )
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    docs = Path(__file__).resolve().parent.parent
    draw(docs / "architecture_diagram.png")
    draw_volume_loop(docs / "volume_loop_diagram.png")
