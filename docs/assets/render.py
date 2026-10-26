"""Render the README images (light and dark variants) and the social preview card.

    uv run --no-project --with fonttools --with uharfbuzz --with resvg-py \
        python docs/assets/render.py

Text is converted to outlines (Bricolage Grotesque, SIL Open Font License) so the images look
the same in every browser. The colored strips are real embeddings: the first dimensions of
Chroma's default embedding model for the sentences in embeddings.json.
"""

from __future__ import annotations

import io
import json
import math
import urllib.request
from pathlib import Path

import uharfbuzz as hb
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont

HERE = Path(__file__).parent
FONT_URL = (
    "https://github.com/google/fonts/raw/main/ofl/bricolagegrotesque/"
    "BricolageGrotesque%5Bopsz%2Cwdth%2Cwght%5D.ttf"
)
FONT_CACHE = Path.home() / ".cache" / "pgchroma-assets" / "BricolageGrotesque.ttf"
EMBEDDINGS = json.loads((HERE / "embeddings.json").read_text())["rows"]

# Benchmark numbers from the README (bench/bench.py, 50k x 384-dim, Chroma 1.5.9 vs pgchroma).
BENCH = [
    ("Ingest", "records per second, higher is better", 12497, 1274, "{:,.0f}/s"),
    ("Filtered query", "p50 latency, lower is better", 11.4, 37.2, "{:.1f} ms"),
    ("get(where=…)", "p50 latency, lower is better", 3.9, 11.3, "{:.1f} ms"),
    ("Query", "p50 latency, lower is better", 7.6, 5.3, "{:.1f} ms"),
]

LIGHT = {
    "ink": "#0f1b2d",
    "ink2": "#4b5b70",
    "muted": "#6f7c8e",
    "hair": "#d3dae3",
    "node": "#f6f8fb",
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "mid": "#ebe9e3",
    "pg": "#eaf2fc",
    "gray": "#8b95a3",
}
DARK = {
    "ink": "#e6edf3",
    "ink2": "#a3afbf",
    "muted": "#8590a0",
    "hair": "#2d3644",
    "node": "#121a25",
    "blue": "#3987e5",
    "orange": "#d95926",
    "mid": "#262c36",
    "pg": "#0f2238",
    "gray": "#647082",
}
CARD = {
    "bg": "#0e1a2b",
    "ink": "#f4f7fb",
    "ink2": "#b4c2d4",
    "muted": "#7f91a8",
    "blue": "#3d8ce8",
    "orange": "#e0612c",
    "mid": "#1f2c40",
}


# ------------------------------------------------------------------ type


class Face:
    """One static instance of the variable font, shaped with HarfBuzz."""

    def __init__(self, data: bytes, key: str, **axes: float):
        static = io.BytesIO()
        instantiateVariableFont(TTFont(io.BytesIO(data)), axes).save(static)
        self.tt = TTFont(io.BytesIO(static.getvalue()))
        self.font = hb.Font(hb.Face(static.getvalue()))
        self.upem = self.tt["head"].unitsPerEm
        self.glyphs = self.tt.getGlyphSet()
        self.order = self.tt.getGlyphOrder()
        self.key = key

    def shape(self, text: str) -> list[tuple[int, int]]:
        buf = hb.Buffer()
        buf.add_str(text)
        buf.guess_segment_properties()
        hb.shape(self.font, buf, {})
        pairs = zip(buf.glyph_infos, buf.glyph_positions, strict=True)
        return [(info.codepoint, pos.x_advance) for info, pos in pairs]

    def width(self, text: str, size: float, tracking: float = 0.0) -> float:
        units = sum(a for _, a in self.shape(text)) + tracking * self.upem * (len(text) - 1)
        return units * size / self.upem

    def outline(self, gid: int) -> str:
        pen = SVGPathPen(self.glyphs, ntos=lambda v: f"{v:.0f}")
        self.glyphs[self.order[gid]].draw(pen)
        return pen.getCommands()


def load_faces() -> dict[str, Face]:
    if not FONT_CACHE.exists():
        FONT_CACHE.parent.mkdir(parents=True, exist_ok=True)
        FONT_CACHE.write_bytes(urllib.request.urlopen(FONT_URL).read())
    data = FONT_CACHE.read_bytes()
    return {
        "display": Face(data, "d", wght=700, opsz=96, wdth=100),
        "text": Face(data, "t", wght=400, opsz=24, wdth=100),
        "label": Face(data, "l", wght=600, opsz=16, wdth=100),
        "small": Face(data, "s", wght=420, opsz=12, wdth=100),
    }


# ----------------------------------------------------------------- color


def _to_oklab(hex_color: str) -> tuple[float, float, float]:
    rgb = [int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    r, g, b = (c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb)
    lms = (
        0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b,
        0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b,
        0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b,
    )
    l_, m_, s_ = (math.copysign(abs(v) ** (1 / 3), v) for v in lms)
    return (
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
    )


def _from_oklab(lab: tuple[float, float, float]) -> str:
    L, a, b = lab
    l_, m_, s_ = (
        (L + 0.3963377774 * a + 0.2158037573 * b) ** 3,
        (L - 0.1055613458 * a - 0.0638541728 * b) ** 3,
        (L - 0.0894841775 * a - 1.2914855480 * b) ** 3,
    )
    rgb = (
        4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_,
        -1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_,
        -0.0041960863 * l_ - 0.7034186147 * m_ + 1.7076147010 * s_,
    )
    out = []
    for c in rgb:
        c = min(1.0, max(0.0, c))
        c = 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055
        out.append(round(c * 255))
    return "#" + "".join(f"{c:02x}" for c in out)


def diverging(value: float, neg: str, mid: str, pos: str) -> str:
    """Two hues and a neutral midpoint, interpolated in OKLab."""
    t = max(-1.0, min(1.0, value))
    end = _to_oklab(pos if t >= 0 else neg)
    m = _to_oklab(mid)
    t = abs(t) ** 0.8
    return _from_oklab(tuple(m[i] + (end[i] - m[i]) * t for i in range(3)))


# ------------------------------------------------------------------- svg


class Svg:
    def __init__(self, width: int, height: int, title: str, background: str | None = None):
        self.width, self.height, self.title, self.background = width, height, title, background
        self.glyphs: dict[str, str] = {}
        self.defs: list[str] = []
        self.body: list[str] = []

    def add(self, element: str) -> None:
        self.body.append(element)

    def text(self, x, y, s, face, size, fill, anchor="start", tracking=0.0) -> float:
        w = face.width(s, size, tracking)
        x -= {"start": 0, "middle": w / 2, "end": w}[anchor]
        uses, pen = [], 0.0
        for gid, advance in face.shape(s):
            key = f"{face.key}{gid}"
            if key not in self.glyphs:
                self.glyphs[key] = face.outline(gid)
            if self.glyphs[key]:
                uses.append(f'<use href="#{key}" x="{pen:.0f}"/>')
            pen += advance + tracking * face.upem
        k = size / face.upem
        self.add(
            f'<g fill="{fill}" transform="translate({x:.2f} {y:.2f}) scale({k:.5f} {-k:.5f})">'
            + "".join(uses)
            + "</g>"
        )
        return w

    def render(self) -> str:
        glyphs = "".join(f'<path id="{k}" d="{d}"/>' for k, d in self.glyphs.items() if d)
        bg = (
            f'<rect width="{self.width}" height="{self.height}" fill="{self.background}"/>'
            if self.background
            else ""
        )
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.width} {self.height}" '
            f'width="{self.width}" height="{self.height}" role="img" aria-labelledby="title">'
            f'<title id="title">{self.title}</title><defs>{glyphs}{"".join(self.defs)}</defs>'
            f"{bg}{''.join(self.body)}</svg>\n"
        )


def strips(
    svg: Svg,
    p: dict,
    x0: float,
    x1: float,
    top: float,
    rows: int,
    row_h: float,
    gap: float,
    cells: int,
    fade_id: str,
) -> None:
    """Rows of embedding values, fading in from the left: vectors as rows of a table."""
    cell_gap = 2.0
    cw = (x1 - x0 - (cells - 1) * cell_gap) / cells
    height = rows * row_h + (rows - 1) * gap
    svg.defs.append(
        f'<linearGradient id="{fade_id}g" x1="0" x2="1" y1="0" y2="0">'
        '<stop offset="0" stop-color="#fff" stop-opacity="0"/>'
        '<stop offset="0.45" stop-color="#fff" stop-opacity="1"/></linearGradient>'
        f'<mask id="{fade_id}"><rect x="{x0}" y="{top}" width="{x1 - x0}" height="{height}" '
        f'fill="url(#{fade_id}g)"/></mask>'
    )
    rects = []
    for r in range(rows):
        values = EMBEDDINGS[r % len(EMBEDDINGS)]
        y = top + r * (row_h + gap)
        for c in range(cells):
            fill = diverging(values[c] / 0.11, p["blue"], p["mid"], p["orange"])
            x = x0 + c * (cw + cell_gap)
            rects.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{cw:.1f}" height="{row_h}" rx="1.5" '
                f'fill="{fill}"/>'
            )
    svg.add(f'<g mask="url(#{fade_id})">{"".join(rects)}</g>')


# ---------------------------------------------------------------- images


def banner(p: dict, f: dict) -> Svg:
    svg = Svg(900, 200, "pgchroma: Chroma's API. Your Postgres.")
    svg.text(-4, 104, "pgchroma", f["display"], 96, p["ink"], tracking=-0.025)
    svg.text(0, 152, "Chroma's API. Your Postgres.", f["text"], 27, p["ink2"])
    strips(svg, p, 500, 900, 22, rows=8, row_h=13, gap=7, cells=44, fade_id="fade")
    return svg


def _node(svg, f, p, x, y, w, h, title, lines, stroke, fill, dashed=False):
    dash = ' stroke-dasharray="5 4"' if dashed else ""
    svg.add(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="1.5"{dash}/>'
    )
    svg.text(x + 18, y + 33, title, f["label"], 17, p["ink"])
    for i, line in enumerate(lines):
        svg.text(x + 18, y + 58 + i * 20, line, f["small"], 13.5, p["ink2"])


def _arrow(svg, f, p, x1, y, x2, label):
    svg.add(
        f'<line x1="{x1}" y1="{y}" x2="{x2 - 2}" y2="{y}" stroke="{p["muted"]}" '
        'stroke-width="1.5" marker-end="url(#head)"/>'
    )
    svg.text((x1 + x2) / 2, y - 9, label, f["small"], 12.5, p["muted"], anchor="middle")


def architecture(p: dict, f: dict) -> Svg:
    svg = Svg(
        900,
        300,
        "pgchroma architecture: your Chroma client talks to pgchroma serve, "
        "which stores collections in Postgres; pgchroma migrate copies Chroma Cloud there",
    )
    svg.defs.append(
        '<marker id="head" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" '
        f'markerHeight="8" orient="auto"><path d="M0,1 L9,5 L0,9 z" fill="{p["muted"]}"/>'
        "</marker>"
    )
    c1, c2, c3 = (0, 225), (345, 565), (668, 900)
    r1, r2 = (10, 112), (178, 112)
    _node(
        svg,
        f,
        p,
        c1[0],
        r1[0],
        c1[1] - c1[0],
        r1[1],
        "Your app",
        ["chromadb for Python or JS,", "LangChain, LlamaIndex"],
        p["hair"],
        p["node"],
    )
    _node(
        svg,
        f,
        p,
        c2[0],
        r1[0],
        c2[1] - c2[0],
        r1[1],
        "pgchroma serve",
        ["Chroma's v2 HTTP API", "stateless, scale it out"],
        p["blue"],
        p["node"],
    )
    _node(
        svg,
        f,
        p,
        c1[0],
        r2[0],
        c1[1] - c1[0],
        r2[1],
        "Chroma Cloud",
        ["or a self-hosted Chroma", "writes stop Nov 22, 2026", "reads stop Dec 22, 2026"],
        p["orange"],
        p["node"],
        dashed=True,
    )
    _node(
        svg,
        f,
        p,
        c2[0],
        r2[0],
        c2[1] - c2[0],
        r2[1],
        "pgchroma migrate",
        ["read-only on the source", "resumable and verified"],
        p["blue"],
        p["node"],
    )
    # Postgres as a cylinder spanning both rows.
    x, w, top, h, ry = c3[0], c3[1] - c3[0], 22, 256, 18
    svg.add(
        f'<path d="M{x},{top + ry} a{w / 2},{ry} 0 0 1 {w},0 v{h - 2 * ry} '
        f'a{w / 2},{ry} 0 0 1 {-w},0 z" fill="{p["pg"]}" stroke="{p["blue"]}" '
        'stroke-width="1.5"/>'
        f'<ellipse cx="{x + w / 2}" cy="{top + ry}" rx="{w / 2}" ry="{ry}" fill="none" '
        f'stroke="{p["blue"]}" stroke-width="1.5"/>'
    )
    svg.text(x + 22, top + 74, "Postgres + pgvector", f["label"], 17, p["ink"])
    for i, line in enumerate(
        ["one table per collection", "HNSW and jsonb indexes", "backups, replicas, SQL joins"]
    ):
        svg.text(x + 22, top + 100 + i * 21, line, f["small"], 13.5, p["ink2"])
    _arrow(svg, f, p, c1[1], r1[0] + 56, c2[0], "Chroma HTTP API")
    _arrow(svg, f, p, c2[1], r1[0] + 56, c3[0], "SQL")
    _arrow(svg, f, p, c1[1], r2[0] + 56, c2[0], "read-only")
    _arrow(svg, f, p, c2[1], r2[0] + 56, c3[0], "SQL")
    return svg


def _bar(x, y, length, h, fill):
    r = min(4.0, length)
    return (
        f'<path d="M{x:.1f},{y:.1f} h{length - r:.1f} a{r},{r} 0 0 1 {r},{r} v{h - 2 * r:.1f} '
        f'a{r},{r} 0 0 1 {-r},{r} h{-(length - r):.1f} z" fill="{fill}"/>'
    )


def benchmark(p: dict, f: dict) -> Svg:
    svg = Svg(
        900,
        316,
        "Benchmark: pgchroma vs Chroma 1.5.9 on 50,000 384-dimensional vectors. "
        "Ingest 12,497 vs 1,274 records/s; filtered query 11.4 vs 37.2 ms; "
        "get with where 3.9 vs 11.3 ms; query 7.6 vs 5.3 ms",
    )
    x = 0.0
    for name, color in (("pgchroma", p["blue"]), ("Chroma 1.5.9", p["gray"])):
        svg.add(f'<rect x="{x}" y="6" width="12" height="12" rx="3" fill="{color}"/>')
        x += 20 + svg.text(x + 20, 17, name, f["small"], 14, p["ink2"]) + 22
    svg.text(
        900,
        17,
        "50,000 × 384-dim vectors, same client and machine",
        f["small"],
        13,
        p["muted"],
        anchor="end",
    )
    panel_w, label_w, value_w, bar_h = 420, 104, 74, 18
    for i, (title, sub, ours, theirs, fmt) in enumerate(BENCH):
        px, py = (i % 2) * 480, 58 + (i // 2) * 134
        svg.text(px, py + 16, title, f["label"], 16, p["ink"])
        svg.text(px, py + 37, sub, f["small"], 13, p["muted"])
        top = py + 54
        scale = (panel_w - label_w - value_w) / max(ours, theirs)
        svg.add(
            f'<line x1="{px + label_w}" y1="{top - 5}" x2="{px + label_w}" '
            f'y2="{top + 2 * bar_h + 15}" stroke="{p["hair"]}" stroke-width="1"/>'
        )
        for j, (name, value, color) in enumerate(
            (("pgchroma", ours, p["blue"]), ("Chroma 1.5.9", theirs, p["gray"]))
        ):
            y = top + j * (bar_h + 10)
            svg.text(px, y + 13.5, name, f["small"], 13.5, p["ink2"])
            length = max(2.0, value * scale)
            svg.add(_bar(px + label_w + 1, y, length, bar_h, color))
            svg.text(
                px + label_w + length + 9, y + 13.5, fmt.format(value), f["small"], 13.5, p["ink"]
            )
    return svg


def social(f: dict) -> Svg:
    p = CARD
    svg = Svg(1280, 640, "pgchroma: Chroma's API. Your Postgres.", background=p["bg"])
    svg.text(84, 262, "pgchroma", f["display"], 150, p["ink"], tracking=-0.025)
    svg.text(92, 338, "Chroma's API. Your Postgres.", f["text"], 44, p["ink2"])
    svg.text(92, 420, "A drop-in Chroma server on Postgres + pgvector,", f["text"], 27, p["muted"])
    svg.text(92, 458, "with a verified migration off Chroma Cloud.", f["text"], 27, p["muted"])
    svg.text(92, 566, "github.com/TheJustas228/pgchroma", f["small"], 24, p["muted"])
    strips(svg, p, 760, 1200, 112, rows=12, row_h=22, gap=12, cells=36, fade_id="fade")
    return svg


def main() -> None:
    f = load_faces()
    for mode, p in (("light", LIGHT), ("dark", DARK)):
        for name, fn in (
            ("banner", banner),
            ("architecture", architecture),
            ("benchmark", benchmark),
        ):
            (HERE / f"{name}-{mode}.svg").write_text(fn(p, f).render())
    card = social(f).render()
    (HERE / "social.svg").write_text(card)
    try:
        import resvg_py
    except ImportError:
        return
    (HERE / "social.png").write_bytes(bytes(resvg_py.svg_to_bytes(svg_string=card)))


if __name__ == "__main__":
    main()
