"""Unit tests for the TCM-Classics-Bench pipeline.

These use a small synthetic book that mirrors the real Jicheng markup, so the
tests run without the downloaded corpus.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tcm_bench import markup, taxonomy
from tcm_bench.generate import generate_items, generate_t1, generate_t6_from_formulas
from tcm_bench.ingest import ingest_book, load_book_meta
from tcm_bench.validate import validate_item

SYNTHETIC_INDEX = """\
====== 測試方書 ======

<book>
書名=測試方書
作者=張機
朝代=漢
年份=219
分類=金匱
品質=90%
參本=趙開美刊本
</book>

===== 痙濕暍病脈證並治第二 =====

太陽病，其證備，身體強，几几然，脈反沉遲，此為痙，栝蔞桂枝湯主之。

<F>
**栝蔞桂枝湯方：**

栝蔞根<l>二兩</l>　桂枝<l>三兩</l>　附子<l>一枚，炮</l>

右三味，以水九升，煮取三升，分溫三服。
</F>
"""


@pytest.fixture()
def book_dir(tmp_path: Path) -> Path:
    d = tmp_path / "測試方書"
    d.mkdir()
    (d / "index.txt").write_text(SYNTHETIC_INDEX, encoding="utf-8")
    return d


# --- markup -------------------------------------------------------------
def test_strip_inline_keeps_dose_drops_notes():
    assert markup.strip_inline("桂枝<l>三兩</l>((炮製))") == "桂枝三兩"
    assert markup.strip_inline("見[[book:金匱要略方論:]]") == "見金匱要略方論"


def test_remove_punctuation_preserves_glyphs():
    src = "太陽病，其證備。"
    assert markup.remove_punctuation(src) == "太陽病其證備"


def test_formula_block_parse_splits_dose_and_prep():
    fbs = markup.extract_formula_blocks(SYNTHETIC_INDEX)
    assert len(fbs) == 1
    fb = fbs[0]
    assert fb.name == "栝蔞桂枝湯"
    herbs = {i["herb"]: i for i in fb.ingredients}
    assert herbs["附子"]["dose"] == "一枚"
    assert herbs["附子"]["preparation"] == "炮"


# --- taxonomy -----------------------------------------------------------
def test_category_and_quality_mapping():
    assert taxonomy.map_category("金匱") == ("傷寒金匱類", "金匱")
    assert taxonomy.quality_tier("90%").name == "test"
    assert taxonomy.quality_tier("0%").name == "unscored"
    assert taxonomy.quality_tier(None).name == "unscored"
    assert "T1" in taxonomy.tasks_for_category("傷寒金匱類")


# --- parsing / ingest ---------------------------------------------------
def test_metadata_parsed(book_dir: Path):
    meta = load_book_meta(book_dir)
    assert meta.author == "張機"
    assert meta.category_level_1 == "傷寒金匱類"
    assert meta.quality_tier == "test"
    assert meta.book_title_simp  # opencc may or may not change it


def test_ingest_yields_formula_passage(book_dir: Path):
    recs = [r.to_dict() for r in ingest_book(book_dir, "2026-06-26")]
    assert any(r["formulas"] for r in recs)
    formula_rec = next(r for r in recs if r["formulas"])
    assert formula_rec["formulas"][0]["formula_name"] == "栝蔞桂枝湯"


# --- generation + validation -------------------------------------------
def test_t6_is_source_grounded(book_dir: Path):
    recs = [r.to_dict() for r in ingest_book(book_dir, "2026-06-26")]
    formula_rec = next(r for r in recs if r["formulas"])
    items = list(generate_t6_from_formulas(formula_rec))
    assert items
    item = items[0].to_dict()
    res = validate_item(item, formula_rec["raw_text_trad"])
    assert res.ok, res.errors
    # 附子 is a restricted herb -> safety note must be present.
    assert item["safety_note"]


def test_t1_roundtrip_validates(book_dir: Path):
    recs = [r.to_dict() for r in ingest_book(book_dir, "2026-06-26")]
    cond = next(r for r in recs if "太陽病" in r["raw_text_trad"] and not r["formulas"])
    item = generate_t1(cond)
    assert item is not None
    d = item.to_dict()
    res = validate_item(d, cond["raw_text_trad"])
    assert res.ok, res.errors


def test_validation_rejects_invented_herb(book_dir: Path):
    recs = [r.to_dict() for r in ingest_book(book_dir, "2026-06-26")]
    formula_rec = next(r for r in recs if r["formulas"])
    item = next(generate_t6_from_formulas(formula_rec)).to_dict()
    item["answer"]["ingredients"].append(
        {"herb": "人參", "dose": "三兩", "preparation": None, "evidence": "人參三兩"}
    )
    res = validate_item(item, formula_rec["raw_text_trad"])
    assert not res.ok
    assert any("人參" in e for e in res.errors)


def test_external_required_is_rejected(book_dir: Path):
    recs = [r.to_dict() for r in ingest_book(book_dir, "2026-06-26")]
    cond = next(r for r in recs if not r["formulas"])
    item = generate_t1(cond).to_dict()
    item["inference_level"] = "external_required"
    res = validate_item(item, cond["raw_text_trad"])
    assert not res.ok


def test_generate_items_orchestration(book_dir: Path):
    recs = [r.to_dict() for r in ingest_book(book_dir, "2026-06-26")]
    items = list(generate_items(recs, ["T1", "T6"]))
    codes = {it.task_code for it in items}
    assert codes == {"T1", "T6"}
