"""Source-grounded validation of generated benchmark items.

Because the full source passage is always available, every generated item is
checked against it: evidence spans must be substrings of the source, formula
names and herbs must occur in the source, doses must not be invented, and the
provenance fields must be intact.  ``external_required`` items are rejected
from the formal test set per the protocol.

``validate_item`` returns a ``ValidationResult``; it never raises on content,
only collects problems, so a batch run can report all failures at once.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .markup import normalize


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _contains(haystack: str, needle: str) -> bool:
    """Substring check that ignores punctuation / spacing differences."""
    if not needle:
        return True
    if needle in haystack:
        return True
    return normalize(needle) in normalize(haystack)


def validate_item(item: dict, source_text: str) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    # 1. Provenance must be intact and point at this source.
    ev = item.get("evidence") or {}
    if not ev.get("book_title_trad"):
        errors.append("evidence.book_title_trad missing")
    if ev.get("source_id") != "jicheng_tcm":
        errors.append("evidence.source_id != jicheng_tcm")

    # 2. Every evidence span must be recoverable from the source.
    for span in ev.get("spans", []):
        if not _contains(source_text, span):
            errors.append(f"evidence span not in source: {span[:24]!r}")

    # 3. external_required items may not enter the formal test set.
    if item.get("inference_level") == "external_required":
        errors.append("inference_level=external_required (excluded from test set)")

    # 4. Task-specific grounding.
    task = item.get("task")
    answer = item.get("answer")
    if task == "formula_structure_parsing" and isinstance(answer, dict):
        errors += _validate_formula(answer, source_text)
    if task == "punctuation_restoration":
        errors += _validate_punctuation(item, source_text)

    # 4b. MCQ items: each distractor needs a source-based exclusion reason;
    # an exclusion that needs outside knowledge downgrades the item.
    if item.get("options"):
        errors += _validate_mcq_shape(item, warnings)

    # 5. Safety: toxic/restricted herbs require a safety note.
    from .generate import SAFETY_HERBS

    text_for_safety = (
        f"{item.get('context','')}{_answer_text(answer)}"
        f"{' '.join(map(str, item.get('options') or []))}"
    )
    if any(h in text_for_safety for h in SAFETY_HERBS) and not item.get("safety_note"):
        warnings.append("toxic/restricted herb present but safety_note empty")

    return ValidationResult(ok=not errors, errors=errors, warnings=warnings)


def _validate_mcq_shape(item: dict, warnings: list[str]) -> list[str]:
    errors: list[str] = []
    options = item.get("options") or []
    if len(options) < 3:
        errors.append(f"MCQ needs >=3 options, got {len(options)}")
    if not item.get("answer"):
        errors.append("MCQ has no answer")
    for d in item.get("distractors", []):
        if not d.get("exclusion_reason"):
            errors.append("distractor without exclusion_reason")
        elif d.get("requires_external"):
            warnings.append("distractor exclusion needs external knowledge -> downgrade to training")
    return errors


def _validate_formula(answer: dict, source_text: str) -> list[str]:
    errors: list[str] = []
    name = answer.get("formula_name")
    if name and not _contains(source_text, name):
        errors.append(f"formula_name not in source: {name!r}")
    for ing in answer.get("ingredients", []):
        herb = ing.get("herb")
        if herb and not _contains(source_text, herb):
            errors.append(f"herb not in source: {herb!r}")
        dose = ing.get("dose")
        if dose and not _contains(source_text, dose):
            errors.append(f"dose not in source: {dose!r} ({herb})")
    return errors


def _validate_punctuation(item: dict, source_text: str) -> list[str]:
    """The answer must equal the source up to punctuation insertion."""
    from .markup import remove_punctuation

    answer = item.get("answer")
    if not isinstance(answer, str):
        return ["punctuation answer must be a string"]
    if remove_punctuation(answer) != remove_punctuation(item.get("context", "")):
        return ["punctuation answer changes characters beyond punctuation"]
    return []


def _answer_text(answer) -> str:
    if isinstance(answer, str):
        return answer
    if isinstance(answer, dict):
        return " ".join(str(v) for v in _flatten(answer))
    if isinstance(answer, list):
        return " ".join(str(v) for v in answer)
    return str(answer)


def _flatten(d):
    for v in d.values():
        if isinstance(v, dict):
            yield from _flatten(v)
        elif isinstance(v, list):
            for x in v:
                yield from (_flatten(x) if isinstance(x, dict) else [x])
        else:
            yield v


# --------------------------------------------------------------------------
# Distractor / MCQ checks (T9 類方鑑別).  Stubs for the human-review stage:
# a correct option must be source-supported and each distractor must carry an
# explicit, source-based exclusion reason, else the item is downgraded.
# --------------------------------------------------------------------------
def validate_mcq(item: dict, source_text: str) -> ValidationResult:
    res = validate_item(item, source_text)
    distractors = item.get("distractors", [])
    for d in distractors:
        if not d.get("exclusion_reason"):
            res.errors.append("distractor without exclusion_reason")
        elif d.get("requires_external"):
            res.warnings.append("distractor exclusion needs external knowledge -> downgrade to training")
    res.ok = not res.errors
    return res
