"""Walk the extracted Jicheng archive into a catalog + corpus JSONL.

Outputs
    catalog.jsonl   one line per book (BookMeta) — lightweight, full coverage.
    corpus.jsonl    one line per passage (CorpusRecord) for the books selected.

Passages shorter than ``min_chars`` (front-matter fragments, stray headers)
are dropped from the corpus but the book still appears in the catalog.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from . import SOURCE_HOMEPAGE, taxonomy
from .markup import normalize
from .parsing import iter_passages, parse_metadata, to_simplified
from .schema import BookMeta, CorpusRecord


def book_dirs(root: Path) -> list[Path]:
    """Every directory under *root* that contains an ``index.txt``."""
    return sorted(p.parent for p in root.glob("*/index.txt"))


def _book_url(book_id: str) -> str:
    return f"{SOURCE_HOMEPAGE}book/{book_id}/"


def load_book_meta(book_dir: Path) -> BookMeta:
    index = book_dir / "index.txt"
    text = index.read_text(encoding="utf-8", errors="replace") if index.exists() else ""
    return parse_metadata(text, book_id=book_dir.name, source_url=_book_url(book_dir.name))


def _passage_id(book_id: str, heading_path: list[str], text: str, idx: int) -> str:
    key = f"{book_id}|{'>'.join(heading_path)}|{idx}|{text[:32]}"
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    return f"{book_id}::{idx:04d}::{h}"


def ingest_book(
    book_dir: Path,
    ingestion_date: str,
    *,
    min_chars: int = 12,
) -> Iterator[CorpusRecord]:
    meta = load_book_meta(book_dir)
    for idx, passage in enumerate(iter_passages(book_dir)):
        text = passage.text_raw
        if len(text) < min_chars:
            continue
        heading_path = passage.heading_path
        chapter = heading_path[0] if heading_path else None
        section = heading_path[1] if len(heading_path) > 1 else None
        formulas = [
            {
                "formula_name": fb.name,
                "ingredients": fb.ingredients,
                "text": fb.text,
            }
            for fb in passage.formula_blocks
        ]
        yield CorpusRecord(
            passage_id=_passage_id(meta.book_id, heading_path, text, idx),
            book_id=meta.book_id,
            book_title_trad=meta.book_title_trad,
            book_title_simp=meta.book_title_simp,
            author=meta.author,
            dynasty=meta.dynasty,
            year=meta.year,
            category_level_1=meta.category_level_1,
            category_level_2=meta.category_level_2,
            base_text=meta.base_text,
            quality_score_source=meta.quality_score_source,
            quality_tier=meta.quality_tier,
            heading_path=heading_path,
            chapter=chapter,
            section_title=section,
            raw_text_trad=text,
            raw_text_simp=to_simplified(text),
            normalized_text=normalize(text),
            formulas=formulas,
            candidate_tasks=taxonomy.tasks_for_category(meta.category_level_1),
            source_url=meta.source_url,
            ingestion_date=ingestion_date,
        )


def write_catalog(root: Path, out: Path) -> int:
    n = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for d in book_dirs(root):
            meta = load_book_meta(d)
            fh.write(json.dumps(meta.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n


def write_corpus(
    root: Path,
    out: Path,
    books: Iterable[str],
    ingestion_date: str,
    *,
    min_chars: int = 12,
) -> int:
    n = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for name in books:
            d = root / name
            if not (d / "index.txt").exists() and not any(d.glob("*.txt")):
                continue
            for rec in ingest_book(d, ingestion_date, min_chars=min_chars):
                fh.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
                n += 1
    return n
