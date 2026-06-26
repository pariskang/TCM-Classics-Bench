"""Read a single Jicheng book directory into structured records.

A book directory looks like one of:

    book/index.txt                      metadata + (sometimes) the whole text
    book/menu.txt  + book/1.txt 2.txt   metadata in index.txt, text per-chapter

We parse the ``<book>`` metadata, then walk the content tracking the heading
hierarchy (encoded by the number of ``=`` characters) and split it into
passages — the unit the generator reads from.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

try:  # OpenCC is optional; without it raw_text_simp falls back to trad.
    from opencc import OpenCC

    _T2S = OpenCC("t2s")

    def to_simplified(text: str) -> str:
        return _T2S.convert(text)

    HAVE_OPENCC = True
except Exception:  # pragma: no cover - environment dependent
    HAVE_OPENCC = False

    def to_simplified(text: str) -> str:
        return text


from . import taxonomy
from .markup import (
    HEADING_RE,
    Heading,
    extract_formula_blocks,
    strip_inline,
)
from .schema import BookMeta

BOOK_BLOCK_RE = re.compile(r"<book>(.*?)</book>", re.DOTALL)
MENU_BLOCK_RE = re.compile(r"<menu>(.*?)</menu>", re.DOTALL)

# Map Chinese <book> keys to BookMeta fields we care about.
_META_KEYS = {
    "書名": "title",
    "作者": "author",
    "作者描述": "author_note",
    "朝代": "dynasty",
    "年份": "year",
    "分類": "category_raw",
    "品質": "quality",
    "參本": "base_text",
    "版本": "version",
    "底本": "base_text",
}


@dataclass
class Passage:
    heading_path: list[str]
    text_raw: str          # cleaned reading text (繁體)
    formula_blocks: list   # list[markup.FormulaBlock]


def slugify(name: str) -> str:
    """Stable ASCII-safe id; falls back to the original CJK when needed."""
    s = re.sub(r"[\s/]+", "_", name.strip())
    return s


def parse_metadata(index_text: str, book_id: str, source_url: str | None) -> BookMeta:
    raw: dict[str, str] = {}
    m = BOOK_BLOCK_RE.search(index_text)
    if m:
        for line in m.group(1).splitlines():
            line = line.strip()
            if "=" in line:
                k, _, v = line.partition("=")
                raw[k.strip()] = v.strip()

    title = raw.get("書名", book_id)
    level1, level2 = taxonomy.map_category(raw.get("分類"))
    tier = taxonomy.quality_tier(raw.get("品質"))
    extra = {k: v for k, v in raw.items() if k not in _META_KEYS}

    return BookMeta(
        book_id=book_id,
        book_title_trad=title,
        book_title_simp=to_simplified(title),
        author=raw.get("作者"),
        author_note=raw.get("作者描述"),
        dynasty=raw.get("朝代"),
        year=raw.get("年份"),
        category_level_1=level1,
        category_level_2=level2,
        category_raw=raw.get("分類", ""),
        base_text=raw.get("參本") or raw.get("底本") or raw.get("版本"),
        quality_score_source=raw.get("品質"),
        quality_tier=tier.name,
        source_url=source_url,
        extra=extra,
    )


def _read_menu_order(book_dir: Path) -> list[Path]:
    """Return chapter files in menu order, else numeric order, else [index]."""
    menu = book_dir / "menu.txt"
    files: list[Path] = []
    if menu.exists():
        m = MENU_BLOCK_RE.search(menu.read_text(encoding="utf-8", errors="replace"))
        if m:
            for line in m.group(1).splitlines():
                num = line.split("|", 1)[0].strip()
                f = book_dir / f"{num}.txt"
                if f.exists():
                    files.append(f)
    if files:
        return files
    numbered = sorted(
        (p for p in book_dir.glob("*.txt") if p.stem.isdigit()),
        key=lambda p: int(p.stem),
    )
    if numbered:
        return numbered
    return [book_dir / "index.txt"]


def _strip_book_block(text: str) -> str:
    return BOOK_BLOCK_RE.sub("", text)


def _rank_levels(eq_lens: set[int]) -> dict[int, int]:
    """Map each '=' length to a 0-based depth (0 = shallowest)."""
    return {eq: i for i, eq in enumerate(sorted(eq_lens, reverse=True))}


def iter_passages(book_dir: Path) -> list[Passage]:
    """Walk a book's content into passages, tracking heading hierarchy."""
    files = _read_menu_order(book_dir)

    # First pass: collect all heading eq-lengths to rank levels per book.
    all_lines: list[str] = []
    for f in files:
        if not f.exists():
            continue
        content = f.read_text(encoding="utf-8", errors="replace")
        if f.name == "index.txt":
            content = _strip_book_block(content)
        all_lines.extend(content.splitlines())

    eq_lens = {
        len(m.group(1))
        for line in all_lines
        if (m := HEADING_RE.match(line.strip()))
    }
    rank = _rank_levels(eq_lens)

    stack: list[Heading] = []
    passages: list[Passage] = []
    buf: list[str] = []

    def flush() -> None:
        if not buf:
            return
        raw_joined = "\n".join(buf).strip()
        if not raw_joined:
            buf.clear()
            return
        cleaned = strip_inline(raw_joined)
        if cleaned:
            passages.append(
                Passage(
                    heading_path=[h.raw for h in stack],
                    text_raw=cleaned,
                    formula_blocks=extract_formula_blocks(raw_joined),
                )
            )
        buf.clear()

    in_formula = False  # don't split on blank lines inside an <F>...</F> block
    for line in all_lines:
        stripped = line.strip()
        m = HEADING_RE.match(stripped)
        if m and not in_formula:
            flush()
            eq = len(m.group(1))
            depth = rank[eq]
            # Pop headings at the same or deeper level (>= this depth).
            while stack and rank[stack[-1].eq_len] >= depth:
                stack.pop()
            stack.append(Heading(raw=strip_inline(m.group(2)), eq_len=eq, line_no=0))
            continue
        if "<F>" in stripped:
            in_formula = True
        if in_formula:
            buf.append(line)
            if "</F>" in stripped:
                in_formula = False
            continue
        if stripped == "":
            flush()
        else:
            buf.append(line)
    flush()
    return passages
