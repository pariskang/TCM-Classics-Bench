"""Parsing helpers for the lightweight wiki markup used by 中醫笈成.

The Jicheng text files use a small, consistent set of inline / block tokens.
Observed conventions (book-20180111):

Headings
    ``======卷第二======``     a run of ``=`` of length N on its own line.
    The number of ``=`` encodes depth: more ``=`` == shallower (a larger
    structural unit).  Typical mapping inside a single book::

        6 = -> 卷 / juan        (level 1)
        5 = -> 篇 / 病門          (level 2)
        4 = -> 方名 / sub-entry  (level 3)

    Depth is *relative to the book*, so we record the raw ``=`` length and let
    the parser rank them.

Block tokens
    ``<F> ... </F>``   a prescription (方) block.
    ``<l> ... </l>``   an inline small-print annotation, almost always a dose
                       attached to the preceding herb (``桂枝<l>三兩</l>``).
    ``<F>/<z>``        formatting wrappers we strip but flag.
    ``<#/> <&/> <~/>`` soft clause / paragraph breaks.

Reference / note tokens
    ``((  ...  ))``    an editorial note / 校注.
    ``[[book:foo:]]``  a cross-reference to another book.
    ``**bold**``       emphasis, used for formula names inside ``<F>`` blocks.
    ``__ ... __``      a colophon / signature line.

The functions here are deliberately conservative: they never invent text and
they keep the original (繁體) characters untouched.  Only the *markup* is
removed; every character that survives is a character that was in the source.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterator

HEADING_RE = re.compile(r"^(={3,})\s*(.+?)\s*\1\s*$")
# An <F>...</F> prescription block (non-greedy, spans newlines).
FORMULA_BLOCK_RE = re.compile(r"<F>\s*(.*?)\s*</F>", re.DOTALL)
# herb<l>dose</l>  — capture the run of text before the <l> as the herb token.
DOSE_RE = re.compile(r"<l>\s*(.*?)\s*</l>")
EDITORIAL_NOTE_RE = re.compile(r"\(\((.*?)\)\)", re.DOTALL)
CROSSREF_RE = re.compile(r"\[\[(.*?)\]\]")
BOLD_RE = re.compile(r"\*\*(.*?)\*\*")
COLOPHON_RE = re.compile(r"__(.*?)__")
# Soft breaks and stray formatting wrappers that carry no text.
SOFT_BREAK_RE = re.compile(r"<[#&~]/>")
WRAPPER_TAG_RE = re.compile(r"</?[A-Za-z][A-Za-z0-9]*\s*/?>")


@dataclass
class Heading:
    raw: str           # the heading text, markup stripped
    eq_len: int        # number of '=' characters (larger == shallower)
    line_no: int


@dataclass
class FormulaBlock:
    """A prescription extracted from an ``<F>`` block (best-effort)."""

    name: str | None
    ingredients: list[dict] = field(default_factory=list)  # {herb, dose}
    text: str = ""     # the full cleaned block text (preserves 制法/服法)


def strip_inline(text: str, *, keep_dose: bool = True) -> str:
    """Return *text* with inline markup removed but every real glyph kept.

    When *keep_dose* is true the content of ``<l>..</l>`` is kept inline so the
    cleaned reading text still reads ``桂枝三兩`` rather than ``桂枝``.
    """
    text = EDITORIAL_NOTE_RE.sub("", text)
    text = CROSSREF_RE.sub(lambda m: _crossref_label(m.group(1)), text)
    text = BOLD_RE.sub(r"\1", text)
    text = COLOPHON_RE.sub(r"\1", text)
    if keep_dose:
        text = DOSE_RE.sub(r"\1", text)
    else:
        text = DOSE_RE.sub("", text)
    text = SOFT_BREAK_RE.sub("", text)
    text = WRAPPER_TAG_RE.sub("", text)
    # Collapse the runs of full-width spaces the source uses as separators
    # down to a single one, and tidy ordinary whitespace.
    text = re.sub(r"[ \t]+", "", text)
    text = re.sub(r"　{2,}", "　", text)
    return text.strip()


def _crossref_label(inner: str) -> str:
    """Turn ``book:金匱要略_條文版:`` into a readable ``金匱要略_條文版``."""
    parts = [p for p in inner.split(":") if p and p not in ("book", "page")]
    return parts[-1] if parts else ""


def normalize(text: str) -> str:
    """Aggressive normalisation for dedup / retrieval keys.

    Removes *all* punctuation and whitespace, leaving only CJK / alphanumeric
    characters.  Never used as answer text — only as an index key.
    """
    cleaned = strip_inline(text, keep_dose=False)
    return re.sub(r"[^\w㐀-鿿豈-﫿]", "", cleaned)


def iter_headings(lines: list[str]) -> Iterator[Heading]:
    for i, line in enumerate(lines):
        m = HEADING_RE.match(line.strip())
        if m:
            yield Heading(raw=strip_inline(m.group(2)), eq_len=len(m.group(1)), line_no=i)


def parse_formula_block(block_text: str) -> FormulaBlock:
    """Best-effort structured parse of one ``<F>`` block body.

    The name is taken from the leading ``**bold**`` line if present.  Herbs and
    doses are read from ``herb<l>dose</l>`` runs.  The full cleaned text is kept
    so downstream tasks can still see 制法 / 服法 verbatim.
    """
    name = None
    bold = BOLD_RE.search(block_text)
    if bold:
        name = strip_inline(bold.group(1)).rstrip("方：:")

    ingredients: list[dict] = []
    # Find the ingredient line(s): segments shaped ``herb<l>dose</l>``.
    for herb_raw, dose in _iter_herb_dose(block_text):
        herb = strip_inline(herb_raw)
        herb = re.split(r"[　\s，,、]", herb)[-1] if herb else herb
        if not herb:
            continue
        # Doses often carry a 炮製 note: ``三兩，去節`` -> dose + preparation.
        dose_part, prep_part = _split_dose(strip_inline(dose))
        ingredients.append({
            "herb": herb,
            "dose": dose_part or None,
            "preparation": prep_part or None,
        })

    return FormulaBlock(name=name, ingredients=ingredients, text=strip_inline(block_text))


def _split_dose(dose: str) -> tuple[str, str]:
    """Split ``三兩，去節`` into ``("三兩", "去節")``; no comma -> ("dose", "")."""
    parts = re.split(r"[，,]", dose, maxsplit=1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return dose.strip(), ""


_HERB_DOSE_RE = re.compile(r"([㐀-鿿]+)\s*<l>\s*(.*?)\s*</l>")


def _iter_herb_dose(block_text: str) -> Iterator[tuple[str, str]]:
    for m in _HERB_DOSE_RE.finditer(block_text):
        yield m.group(1), m.group(2)


def extract_formula_blocks(raw_text: str) -> list[FormulaBlock]:
    return [parse_formula_block(m.group(1)) for m in FORMULA_BLOCK_RE.finditer(raw_text)]


def remove_punctuation(text: str) -> str:
    """Strip Chinese + western sentence punctuation, for T1 (句読恢復) items.

    Keeps every Han glyph and dose digit; removes only marks a句読 task would
    ask the model to re-insert.
    """
    punct = "，。、；：？！「」『』（）()《》〈〉·…—－,.;:?!　\"'"
    return "".join(ch for ch in text if ch not in punct)
