"""Corpus layering: categories, quality tiers, task taxonomy and pilot lists.

Everything here is *source-grounded*: the category map keys are the literal
``分類`` strings found in Jicheng ``index.txt`` files, and the quality tiers
follow the protocol's reading of the site's ``品質`` field.
"""

from __future__ import annotations

from dataclasses import dataclass

# --------------------------------------------------------------------------
# Category mapping: raw 分類 (level-2-ish) -> a coarse level-1 grouping that
# matches the protocol's corpus layers.
# --------------------------------------------------------------------------
CATEGORY_LEVEL1: dict[str, str] = {
    "內經": "醫經類",
    "經論": "醫經類",
    "難經": "難經類",
    "傷寒": "傷寒金匱類",
    "金匱": "傷寒金匱類",
    "溫病": "溫病類",
    "本草": "本草類",
    "炮製": "本草類",
    "方書": "醫方類",
    "醫方": "醫方類",
    "醫論": "醫論類",
    "醫案": "醫案類",
    "診法": "診法類",
    "診治": "診法類",
    "脈法": "診法類",
    "針灸": "針灸類",
    "經穴": "針灸類",
    "婦科": "臨證各科",
    "兒科": "臨證各科",
    "外科": "臨證各科",
    "傷科": "臨證各科",
    "喉科": "臨證各科",
    "眼科": "臨證各科",
    "齒科": "臨證各科",
    "五官科": "臨證各科",
    "內科": "臨證各科",
    "養生": "養生類",
    "綜合": "綜合類",
    "其他": "其他",
}


def map_category(raw: str | None) -> tuple[str, str]:
    """Return ``(level1, level2)`` for a raw ``分類`` string.

    The raw field is sometimes multi-valued (``醫案 傷寒 金匱``); the first
    token drives the level-1 grouping, the whole raw string is the level-2.
    """
    raw = (raw or "").strip()
    if not raw:
        return ("未分類", "")
    first = raw.split()[0]
    return (CATEGORY_LEVEL1.get(first, "其他"), raw)


# --------------------------------------------------------------------------
# Quality tiers.  NOTE: in book-20180111, ~95% of books carry 品質=0%, which
# is the site's "not proof-tracked" default rather than a true low score.  We
# therefore map 0% / missing to an explicit ``unscored`` tier so that core
# classics (e.g. 素問, shown as 2% on the site) are never silently dropped.
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class QualityTier:
    name: str
    description: str
    eligible_for_test: bool


TIER_TEST = QualityTier("test", "品質≥80%：可進入正式測試集", True)
TIER_CANDIDATE = QualityTier("candidate", "品質50–79%：候選池，需專家複核", False)
TIER_EXPLORE = QualityTier("explore", "品質<50% (非0)：訓練/探索用", False)
TIER_UNSCORED = QualityTier("unscored", "品質0%/未標注：人工抽檢後再決定", False)


def parse_quality(raw: str | None) -> int | None:
    """Parse a ``NN%`` string to an int percentage, or ``None`` if absent."""
    if not raw:
        return None
    raw = raw.strip().rstrip("%").strip()
    try:
        return int(float(raw))
    except ValueError:
        return None


def quality_tier(raw: str | None) -> QualityTier:
    pct = parse_quality(raw)
    if pct is None or pct == 0:
        return TIER_UNSCORED
    if pct >= 80:
        return TIER_TEST
    if pct >= 50:
        return TIER_CANDIDATE
    return TIER_EXPLORE


# --------------------------------------------------------------------------
# Task taxonomy T1–T12.
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Task:
    id: str
    name_zh: str
    name_en: str
    needs_llm: bool          # False => can be generated deterministically
    description: str


TASKS: dict[str, Task] = {
    "T1": Task("T1", "標點/句讀恢復", "punctuation_restoration", False,
               "刪除已標點原文的標點，要求模型復原句讀。"),
    "T2": Task("T2", "文白翻譯", "classical_translation", True,
               "將古文片段翻譯為現代白話，專家校正。"),
    "T3": Task("T3", "術語注釋", "term_annotation", True,
               "抽取術語並要求解釋其古籍語境含義。"),
    "T4": Task("T4", "實體識別", "entity_recognition", True,
               "標注病名、症狀、證候、治法、方劑、中藥、劑量。"),
    "T5": Task("T5", "關係抽取", "relation_extraction", True,
               "抽取 方-藥、病-症、證-治、方-證 等關係。"),
    "T6": Task("T6", "方劑結構解析", "formula_structure_parsing", True,
               "解析方名、組成、劑量、炮製、製法、服法、主治。"),
    "T7": Task("T7", "理論分類", "theory_classification", True,
               "六經、臟腑、經絡、衛氣營血、三焦等理論歸類。"),
    "T8": Task("T8", "方證對應", "syndrome_formula_mapping", True,
               "症狀→病機→治法→方劑 的推理。"),
    "T9": Task("T9", "類方鑑別", "formula_differentiation", True,
               "相似方證之間的鑑別與干擾項設計。"),
    "T10": Task("T10", "證據溯源問答", "evidence_grounded_qa", True,
                "問題→原文證據→出處 的檢索式問答。"),
    "T11": Task("T11", "安全禁忌判斷", "safety_contraindication", True,
                "毒性、禁忌、劑量、炮製的安全邊界判斷。"),
    "T12": Task("T12", "幻覺引用檢測", "hallucinated_citation_detection", True,
                "辨別真引用、錯出處、篡改、偽造。"),
}

# Which tasks a given level-1 category is a natural source for.  T1 (句讀恢復)
# applies to any source-punctuated prose, so it is implied for every category
# (see ``tasks_for_category``) and only the *additional* tasks are listed here.
CATEGORY_TASKS: dict[str, list[str]] = {
    "醫經類": ["T2", "T3", "T7", "T10", "T12"],
    "難經類": ["T2", "T3", "T7", "T10", "T12"],
    "傷寒金匱類": ["T2", "T3", "T4", "T6", "T8", "T9", "T10", "T11", "T12"],
    "溫病類": ["T2", "T3", "T4", "T7", "T8", "T10", "T12"],
    "本草類": ["T3", "T4", "T6", "T11", "T12"],
    "醫方類": ["T4", "T5", "T6", "T8", "T9", "T11", "T12"],
    "醫論類": ["T2", "T3", "T7", "T10", "T12"],
    "醫案類": ["T2", "T4", "T5", "T8", "T10", "T12"],
    "診法類": ["T3", "T4", "T7", "T10", "T12"],
    "針灸類": ["T3", "T4", "T7", "T10", "T12"],
    "臨證各科": ["T4", "T5", "T6", "T8", "T11", "T12"],
    "養生類": ["T2", "T3", "T10", "T12"],
    "綜合類": ["T2", "T3", "T4", "T10", "T12"],
}


def tasks_for_category(level1: str) -> list[str]:
    """Candidate task codes for a level-1 category; T1 is always included."""
    extra = CATEGORY_TASKS.get(level1, ["T2", "T3", "T10", "T12"])
    return ["T1", *extra]


# --------------------------------------------------------------------------
# Pilot corpora (directory names as they appear in the archive).
# --------------------------------------------------------------------------
PILOT_CORPORA: dict[str, list[str]] = {
    "pilot1_classics": ["黃帝內經素問", "靈樞", "難經"],
    "pilot2_zhongjing": [
        "傷寒論_宋本", "金匱要略方論", "金匱要略_條文版",
        "傷寒論類方", "長沙方歌括", "金匱方歌括",
    ],
    "pilot3_materia_formulae": [
        "太平惠民和劑局方", "醫方集解", "神農本草經",
        "本草綱目", "本草備要", "雷公炮炙論", "湯頭歌訣",
    ],
}

ALL_PILOT_BOOKS = [b for books in PILOT_CORPORA.values() for b in books]
