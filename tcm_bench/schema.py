"""Dataclasses for corpus records and benchmark items.

These mirror the JSON-Schema files under ``schemas/`` and the ingest schema in
the protocol.  ``to_dict`` produces the canonical on-disk JSON shape.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from . import (
    CORPUS_VERSION,
    SOURCE_ID,
    SOURCE_LICENSE,
)


@dataclass
class BookMeta:
    """Parsed ``<book>`` metadata for one Jicheng book."""

    book_id: str
    book_title_trad: str
    book_title_simp: str
    author: str | None = None
    author_note: str | None = None
    dynasty: str | None = None
    year: str | None = None
    category_level_1: str = "未分類"
    category_level_2: str = ""
    category_raw: str = ""
    base_text: str | None = None          # 參本 / 版本 / 底本
    quality_score_source: str | None = None
    quality_tier: str = "unscored"
    source_url: str | None = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


@dataclass
class CorpusRecord:
    """One ingested passage, the unit LLM generation reads from."""

    passage_id: str
    book_id: str
    book_title_trad: str
    book_title_simp: str
    author: str | None
    dynasty: str | None
    year: str | None
    category_level_1: str
    category_level_2: str
    base_text: str | None
    quality_score_source: str | None
    quality_tier: str
    heading_path: list[str]
    chapter: str | None
    section_title: str | None
    raw_text_trad: str
    raw_text_simp: str
    normalized_text: str
    formulas: list[dict] = field(default_factory=list)
    candidate_tasks: list[str] = field(default_factory=list)
    punctuation_status: str = "source_punctuated"
    source_id: str = SOURCE_ID
    source_url: str | None = None
    source_license: str = SOURCE_LICENSE
    version_commit: str = CORPUS_VERSION
    ingestion_date: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Evidence:
    book_title_trad: str
    chapter: str | None
    base_text: str | None
    source_id: str = SOURCE_ID
    source: str = "中醫笈成"
    version_commit: str = CORPUS_VERSION
    spans: list[str] = field(default_factory=list)


@dataclass
class BenchItem:
    """A generated benchmark question, ready for validation / review."""

    item_id: str
    task: str                       # e.g. "formula_structure_parsing"
    task_code: str                  # e.g. "T6"
    question: str
    context: str
    answer: object                  # str | dict | list, task-dependent
    evidence: Evidence
    book_id: str
    passage_id: str
    inference_level: str = "direct"   # direct | implicit | external_required
    difficulty: str = "Medium"        # Easy | Medium | Hard | Expert
    options: list = field(default_factory=list)        # MCQ options (T7-T9,T11,T12)
    distractors: list = field(default_factory=list)    # [{option, exclusion_reason}]
    safety_note: str = ""
    quality_warning: str = ""
    generator: str = "deterministic"  # deterministic | <model id>
    review_status: str = "unreviewed"
    source_id: str = SOURCE_ID

    def to_dict(self) -> dict:
        d = asdict(self)
        return d
