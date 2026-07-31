"""Renders architecture.png — the two-loop evaluation architecture.

Diagram-as-code: edit this file and re-run to regenerate.

    python scripts/render_architecture.py

Pillow only (matplotlib's compiled backend is blocked on some managed Windows
hosts). Drawn at 2x and downsampled, since PIL's rasteriser is not antialiased.
"""

import os

from PIL import Image, ImageDraw, ImageFont

# --- canvas ------------------------------------------------------------------
W, H = 1640, 1052
SS = 2  # supersample factor

# --- palette -----------------------------------------------------------------
BG = "#FFFFFF"
INK = "#16202B"
MUTED = "#64748B"
FAINT = "#94A3B8"
LINE = "#CBD5E1"
PANEL = "#F6F8FA"
CARD = "#FFFFFF"

FLOW = "#2563EB"     # loop 1 — measuring the agent
META = "#B45309"     # loop 2 — measuring the instrument
META_BG = "#FEF6EC"
META_INK = "#7C5010"
FROZEN = "#7C3AED"   # hash-frozen artifacts
PASS = "#059669"

FONT_DIRS = ["C:/Windows/Fonts", "/usr/share/fonts/truetype/dejavu", "/Library/Fonts"]
CANDIDATES = {
    "regular": ["segoeui.ttf", "DejaVuSans.ttf", "Arial.ttf", "arial.ttf"],
    "semibold": ["seguisb.ttf", "DejaVuSans-Bold.ttf", "Arial Bold.ttf", "arialbd.ttf"],
    "bold": ["segoeuib.ttf", "DejaVuSans-Bold.ttf", "Arial Bold.ttf", "arialbd.ttf"],
}
_cache = {}


def font(weight, size):
    key = (weight, size)
    if key in _cache:
        return _cache[key]
    for name in CANDIDATES[weight]:
        for d in FONT_DIRS:
            p = os.path.join(d, name)
            if os.path.exists(p):
                _cache[key] = ImageFont.truetype(p, size * SS)
                return _cache[key]
    _cache[key] = ImageFont.load_default()
    return _cache[key]


img = Image.new("RGB", (W * SS, H * SS), BG)
d = ImageDraw.Draw(img)


def s(v):
    return int(round(v * SS))


def rect(x, y, w, h, fill=CARD, outline=LINE, width=1.3, radius=9):
    d.rounded_rectangle(
        [s(x), s(y), s(x + w), s(y + h)],
        radius=s(radius), fill=fill, outline=outline, width=max(1, s(width)),
    )


def text(x, y, txt, weight="regular", size=12, color=INK, anchor="la", spacing=1.25):
    """`spacing` is a line-height multiplier; extra leading only, PIL adds the em."""
    d.multiline_text(
        (s(x), s(y)), txt, font=font(weight, size), fill=color, anchor=anchor,
        spacing=int(size * SS * (spacing - 1.0)), align="center"
        if anchor[0] == "m" else "left",
    )


def line(x1, y1, x2, y2, color=LINE, width=1.3):
    d.line([s(x1), s(y1), s(x2), s(y2)], fill=color, width=max(1, s(width)))


def arrow(x1, y1, x2, y2, color=FLOW, width=2.2, head=8):
    """Horizontal arrow with a solid triangular head."""
    d.line([s(x1), s(y1), s(x2 - head * 0.7), s(y2)], fill=color, width=s(width))
    d.polygon(
        [(s(x2), s(y2)), (s(x2 - head), s(y2 - head * 0.55)),
         (s(x2 - head), s(y2 + head * 0.55))],
        fill=color,
    )


def dashed(x1, y1, x2, y2, color=META, width=2.0, dash=11, gap=7):
    total = abs(x2 - x1)
    step = dash + gap
    n = int(total // step)
    sign = 1 if x2 > x1 else -1
    for i in range(n + 1):
        a = x1 + sign * i * step
        b = a + sign * min(dash, total - i * step)
        if (b - a) * sign <= 0:
            break
        d.line([s(a), s(y1), s(b), s(y2)], fill=color, width=s(width))


def badge(cx, cy, txt, color):
    f = font("bold", 10)
    bb = d.textbbox((0, 0), txt, font=f)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    padx, pady = s(7), s(5)
    d.rounded_rectangle(
        [s(cx) - tw // 2 - padx, s(cy) - th // 2 - pady,
         s(cx) + tw // 2 + padx, s(cy) + th // 2 + pady],
        radius=s(5), fill=color,
    )
    d.text((s(cx), s(cy)), txt, font=f, fill="#FFFFFF", anchor="mm")


def card(x, y, w, h, title, lines=(), accent=LINE, width=1.3,
         title_color=None, title_size=15, body_size=11.5):
    rect(x, y, w, h, outline=accent, width=width)
    if lines:
        text(x + w / 2, y + 21, title, "bold", title_size,
             title_color or INK, anchor="mm")
        for i, ln in enumerate(lines):
            text(x + w / 2, y + 44 + i * 18, ln, "regular", body_size,
                 MUTED, anchor="mm")
    else:
        text(x + w / 2, y + h / 2, title, "bold", title_size,
             title_color or INK, anchor="mm")


def band(x, y, w, h, label, color=INK):
    d.rounded_rectangle([s(x), s(y), s(x + w), s(y + h)], radius=s(11), fill=PANEL)
    if label:
        text(x + w / 2, y - 9, label, "bold", 13, color, anchor="md")


# =============================================================================
# HEADER
# =============================================================================
text(28, 24, "Agent Conformance & Regression Gate", "bold", 31, INK)
text(28, 68,
     "A domain-agnostic pre-production gate for any agent — and a second loop that proves the gate works.",
     "regular", 16, MUTED)
line(28, 106, 1612, 106, LINE, 1.3)

# =============================================================================
# LOOP 1
# =============================================================================
text(28, 126, "LOOP 1", "bold", 17, FLOW)
text(108, 128, "MEASURE THE AGENT    ·    what every eval framework already does",
     "regular", 15, MUTED)

BY, BH = 172, 296          # band top / height
CY = BY + BH / 2

S1, W1 = 24, 268
S2, W2 = 320, 244
S3, W3 = 592, 258
S4, W4 = 878, 258
S5, W5 = 1164, 200
S6, W6 = 1392, 224

# ---- 1. declared inputs -----------------------------------------------------
band(S1, BY, W1, BH, "DECLARED INPUTS")
card(S1 + 12, BY + 14, W1 - 24, 88, "Task Registry",
     ("30 dev   +   10 sealed holdout", "prompt · oracle · ground_truth"),
     accent=FROZEN, width=1.7)
badge(S1 + W1 - 52, BY + 14, "eval_set_sha", FROZEN)

card(S1 + 12, BY + 116, W1 - 24, 88, "Agent Configs",
     ("v0 frozen · improved · regressions", "model · prompt · tools · budget"),
     accent=FROZEN, width=1.7)
badge(S1 + W1 - 58, BY + 116, "agent_cfg_sha", FROZEN)

card(S1 + 12, BY + 218, W1 - 24, 64, "Domain Packs",
     ("travel.yaml · support.yaml · finance.yaml",))

# ---- 2. contract ------------------------------------------------------------
band(S2, BY, W2, BH, "CONTRACT")
card(S2 + 12, BY + 14, W2 - 24, 96, "AgentAdapter",
     ("run()   ·   get_tool_calls()", "— the only two required —",
      "6 more optional, defaulted"),
     accent=FLOW, width=2.2)
card(S2 + 12, BY + 124, W2 - 24, 66, "Target A", ("LangGraph travel agent",))
card(S2 + 12, BY + 202, W2 - 24, 66, "Target B", ("public OSS ReAct agent",))
text(S2 + W2 / 2, BY + 280, "same suite, two agents", "bold", 12, META, anchor="mm")

# ---- 3. execution -----------------------------------------------------------
band(S3, BY, W3, BH, "EXECUTION")
card(S3 + 12, BY + 14, W3 - 24, 116, "Fault Middleware",
     ("decorates any adapter", "seeded RNG  ·  per-task engine",
      "22 fault plugins", "patch targets from the pack"),
     accent=META, width=2.2, title_color=META)
card(S3 + 12, BY + 146, W3 - 24, 136, "Runner",
     ("per-task isolation", "N = 3 seeds  ×  tasks",
      "bounded concurrency", "result cache  ·  --replay"),
     accent=FLOW, width=1.7)

# ---- 4. scoring -------------------------------------------------------------
band(S4, BY, W4, BH, "SCORING")
card(S4 + 12, BY + 14, W4 - 24, 130, "Oracles",
     ("State   — did the world change?", "Trace   — right tools, right order?",
      "Judge   — prose only, ≤20% of tasks"),
     accent=FLOW, width=1.7)
card(S4 + 12, BY + 160, W4 - 24, 122, "Metric Plugins",
     ("accuracy · tool · safety", "performance · cost",
      "domain terms come from the pack,", "never from metric code"))

# ---- 5. aggregate -----------------------------------------------------------
band(S5, BY, W5, BH, "AGGREGATE")
card(S5 + 12, BY + 14, W5 - 24, 130, "pass^k",
     ("per-task pass rate", "across k seeds", "", "not a single bool"),
     accent=FLOW, width=1.7)
card(S5 + 12, BY + 160, W5 - 24, 122, "Wilson bound",
     ("lower confidence", "bound on the mean", "+ flaky quarantine"),
     accent=FLOW, width=1.7)

# ---- 6. gate ----------------------------------------------------------------
band(S6, BY, W6, BH, "GATE")
card(S6 + 12, BY + 14, W6 - 24, 130, "Fast lane",
     ("every PR", "10 golden tasks × 1", "under 3 minutes"),
     accent=PASS, width=1.7)
card(S6 + 12, BY + 160, W6 - 24, 122, "Full lane",
     ("nightly + main", "30 tasks × 3 seeds", "vs baseline artifact"),
     accent=PASS, width=1.7)

# ---- flow arrows ------------------------------------------------------------
for a, b in [(S1 + W1, S2), (S2 + W2, S3), (S3 + W3, S4),
             (S4 + W4, S5), (S5 + W5, S6)]:
    arrow(a + 5, CY, b - 4, CY, FLOW, 2.4, 10)

text(S6 + W6 / 2, BY + BH + 20, "exit 0   /   exit 1", "bold", 14, INK, anchor="mm")

# =============================================================================
# THE HINGE
# =============================================================================
HY = 504
d.rounded_rectangle([s(24), s(HY), s(1612), s(HY + 78)], radius=s(10),
                    fill=META_BG, outline=META, width=s(1.7))
text(48, HY + 16, "A green build proves nothing if the suite cannot go red.",
     "bold", 19, META)
text(48, HY + 47,
     "Found in this repo: 3 of 8 baseline comparisons read a float as a dict, so they were structurally incapable of failing. "
     "From the outside, a working suite and a broken one look identical.",
     "regular", 13, META_INK)

# =============================================================================
# LOOP 2
# =============================================================================
text(28, 606, "LOOP 2", "bold", 17, META)
text(108, 608, "MEASURE THE INSTRUMENT    ·    the part no eval framework ships",
     "regular", 15, MUTED)

MY, MH = 652, 214
band(24, MY, 1588, MH, "")

steps = [
    ("Known-good\nagent v0", "frozen agent_cfg_sha", None),
    ("Inject one\nregression", "from a catalogue of\ndecreasing effect size", META),
    ("Re-run\nLoop 1", "identical pipeline,\nnothing else changed", FLOW),
    ("Did the gate\nturn red?", "expected verdict\nvs actual verdict", None),
    ("Detection rate\nper fault type", "over k seeds", None),
    ("Minimum\nDetectable Effect", "the resolution of\nyour instrument", META),
]

sw, gap, sx = 236, 24, 48
centers = []
for i, (title, sub, accent) in enumerate(steps):
    x = sx + i * (sw + gap)
    centers.append(x + sw / 2)
    rect(x, MY + 34, sw, 118, outline=accent or LINE, width=2.2 if accent else 1.3)
    text(x + sw / 2, MY + 68, title, "bold", 15, accent or INK, anchor="mm")
    text(x + sw / 2, MY + 118, sub, "regular", 11.5, MUTED, anchor="mm")
    if i:
        arrow(x - gap + 3, MY + 93, x - 4, MY + 93, META, 2.2, 9)

# feedback edge
fy = MY + 176
dashed(centers[-1], fy, centers[0], fy, META, 2.0)
d.polygon(
    [(s(centers[0] - 8), s(fy)), (s(centers[0] + 2), s(fy - 5)),
     (s(centers[0] + 2), s(fy + 5))], fill=META)
line(centers[-1], MY + 152, centers[-1], fy, META, 2.0)
line(centers[0], fy, centers[0], MY + 152, META, 2.0)
text(818, fy + 16,
     "recalibrate gate thresholds   ·   quarantine flaky tasks   ·   publish the error bar",
     "bold", 12.5, META, anchor="ma")

# =============================================================================
# FOOTER
# =============================================================================
line(24, 902, 1612, 902, LINE, 1.3)

claims = [
    ("Loop 1 is commodity.",
     "DeepEval, promptfoo, Inspect AI and\nagentevals already run it. This design\nadopts them rather than rebuilding them."),
    ("Loop 2 is the contribution.",
     "Fault injection validates the suite, not\njust the agent. Detection rate and MDE\nship as first-class results."),
    ("The gate is statistical.",
     "Agents are stochastic. Gating on a lower\nconfidence bound — never a raw mean —\nis what stops a merge blocker flaking."),
    ("The contract is the product.",
     "Two required adapter methods, domain\nterms confined to packs. That is what\nlets the suite travel to any agent."),
]
for i, (head, body) in enumerate(claims):
    x = 44 + i * 396
    hl = META if i == 1 else LINE
    d.rounded_rectangle([s(x - 14), s(926), s(x - 10), s(1008)], radius=s(2), fill=hl)
    text(x, 921, head, "bold", 15, META if i == 1 else INK)
    text(x, 950, body, "regular", 12, MUTED, spacing=1.45)

# =============================================================================
out = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "architecture.png")
img.resize((W, H), Image.LANCZOS).save(out, "PNG", optimize=True)
print(f"wrote {out}  ({W}x{H})")
