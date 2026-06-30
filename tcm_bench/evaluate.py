"""Evaluate a model on the benchmark: prompt → answer → score.

The model under test is any :mod:`tcm_bench.llm` client (Anthropic, Azure,
Poe, LiteLLM).  Three prompting modes are supported:

    zero_shot   question only
    few_shot    a few solved examples of the same task family first
    cot         ask the model to reason step by step, then give a final answer

Auto-scored task families (no LLM judge needed):

    MCQ (T7-T9, T11, T12)   accuracy   (predicted option letter == gold)
    NER (T4)                F1         over (text, type) entity spans
    formula (T6)            F1         over herb set
    punctuation (T1)        F1         over 句讀 boundary positions

Open-ended tasks (T2/T3/T5/T10) are marked ``scorable=False`` and excluded
from the headline scores (hook a judge model in if you need them).
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Iterable, Iterator

from .llm import complete_with_retry

PROMPT_MODES = ("zero_shot", "few_shot", "cot")
MCQ_TASKS = {"T7", "T8", "T9", "T11", "T12"}

# Punctuation marks a 句讀 task inserts (kept in sync with markup.remove_punctuation).
_PUNCT = set("，。、；：？！「」『』（）()《》〈〉·…—－,.;:?!\"' 　")

# Lenient entity-type synonyms -> canonical NER type.
_TYPE_SYNONYMS = {
    "中藥": "herb", "中药": "herb", "药": "herb", "藥": "herb", "herb": "herb",
    "劑量": "dose", "剂量": "dose", "用量": "dose", "dose": "dose",
    "方名": "formula", "方劑": "formula", "方剂": "formula", "formula": "formula",
    "炮製": "preparation", "炮制": "preparation", "preparation": "preparation",
}


# --------------------------------------------------------------------------
# Task family helpers
# --------------------------------------------------------------------------
def _is_mcq(item: dict) -> bool:
    return bool(item.get("options"))


def family(item: dict) -> str:
    if _is_mcq(item):
        return "mcq"
    return {"T4": "ner", "T6": "formula", "T1": "punctuation"}.get(item["task_code"], "open")


def scorable(item: dict) -> bool:
    return family(item) != "open"


def _metric_for(item: dict) -> str:
    return {"mcq": "accuracy", "ner": "f1", "formula": "f1", "punctuation": "f1"}.get(
        family(item), "none"
    )


def _extract_json(text: str):
    """Return the last JSON object/array embedded in *text*, or None."""
    for pat in (r"\{.*\}", r"\[.*\]"):
        m = re.search(pat, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                continue
    return None


# --------------------------------------------------------------------------
# Prompt building
# --------------------------------------------------------------------------
_SYSTEM = (
    "你是一名精通中醫古籍的專家，正在參加古籍理解測評。"
    "請嚴格依照要求作答，只依據題目所給的原文，不要編造原文未出現的內容。"
)
_COT_HINT = "請先簡要逐步推理，然後在最後一行按指定格式給出最終答案。"


def _answer_format(item: dict) -> str:
    fam = family(item)
    if fam == "mcq":
        return "只能選一個選項。最後一行輸出：`答案：X`（X 為選項字母）。"
    if fam == "ner":
        return ('輸出 JSON：{"entities":[{"text":"原文片段","type":"formula|herb|dose|preparation"}]}。'
                "只列出原文中出現的實體。")
    if fam == "formula":
        return '輸出 JSON：{"formula_name":"","ingredients":[{"herb":"","dose":""}]}。'
    if fam == "punctuation":
        return "輸出加好標點後的完整原文，最後一行以 `標點：` 起頭給出結果。"
    return "請直接作答。"


def _render_question(item: dict) -> str:
    parts = [item.get("question", "").strip()]
    if item.get("context"):
        parts.append(f"原文：{item['context']}")
    if item.get("options"):
        parts.append("選項：\n" + "\n".join(item["options"]))
    return "\n".join(p for p in parts if p)


def gold_render(item: dict) -> str:
    """The target answer string, used to render few-shot examples."""
    fam = family(item)
    if fam == "mcq":
        return f"答案：{_gold_mcq_letter(item)}"
    if fam == "ner":
        return json.dumps({"entities": item["answer"].get("entities", [])}, ensure_ascii=False)
    if fam == "formula":
        ans = item.get("answer") or {}
        ings = [{"herb": i.get("herb"), "dose": i.get("dose")} for i in ans.get("ingredients", [])]
        return json.dumps({"formula_name": ans.get("formula_name"), "ingredients": ings},
                          ensure_ascii=False)
    if fam == "punctuation":
        return f"標點：{item.get('answer','')}"
    return str(item.get("answer", ""))


def build_eval_prompt(item: dict, mode: str = "zero_shot", shots: list | None = None) -> tuple[str, str]:
    """Return ``(system, user)`` for *item* under the given prompting *mode*."""
    shots = shots or []
    blocks: list[str] = []
    if mode == "few_shot" and shots:
        for i, s in enumerate(shots, 1):
            blocks.append(f"【示例{i}】\n{_render_question(s)}\n{gold_render(s)}")
        blocks.append("【正式題目】")
    blocks.append(_render_question(item))
    blocks.append(_answer_format(item))
    if mode == "cot":
        blocks.append(_COT_HINT)
    return _SYSTEM, "\n\n".join(blocks)


# --------------------------------------------------------------------------
# Parsing predictions
# --------------------------------------------------------------------------
def _gold_mcq_letter(item: dict) -> str:
    a = str(item.get("answer", "")).strip()
    m = re.match(r"\s*([A-Da-d])", a)
    return (m.group(1) if m else a[:1]).upper()


def _parse_mcq(raw: str) -> str:
    m = re.search(r"(?:答案|answer)\s*[:：]?\s*([A-Da-d])", raw, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    letters = re.findall(r"(?<![A-Za-z])([A-Da-d])(?![A-Za-z])", raw)
    return letters[-1].upper() if letters else ""


def _parse_entities(raw: str) -> set[tuple[str, str]]:
    data = _extract_json(raw)
    ents = data.get("entities") if isinstance(data, dict) else data if isinstance(data, list) else []
    out: set[tuple[str, str]] = set()
    for e in ents or []:
        if isinstance(e, dict) and e.get("text"):
            typ = _TYPE_SYNONYMS.get(str(e.get("type", "")).strip(), str(e.get("type", "")).strip().lower())
            out.add((str(e["text"]).strip(), typ))
    return out


def _parse_herbs(raw: str) -> set[str]:
    data = _extract_json(raw)
    ings = []
    if isinstance(data, dict):
        ings = data.get("ingredients") or []
    return {str(i.get("herb", "")).strip() for i in ings if isinstance(i, dict) and i.get("herb")}


def _boundaries(text: str) -> set[int]:
    """Positions (count of non-punct chars seen) where a punctuation follows."""
    b, n = set(), 0
    for ch in text:
        if ch in _PUNCT:
            b.add(n)
        else:
            n += 1
    return b


def _parse_punct(raw: str) -> str:
    m = re.search(r"標點[:：]\s*(.+)\Z", raw, re.DOTALL)
    return (m.group(1) if m else raw).strip()


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def _set_f1(pred: set, gold: set) -> tuple[float, float, float]:
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def score_prediction(item: dict, raw: str) -> dict:
    fam = family(item)
    if fam == "mcq":
        pred = _parse_mcq(raw)
        gold = _gold_mcq_letter(item)
        ok = pred == gold and bool(gold)
        return {"metric": "accuracy", "score": 1.0 if ok else 0.0, "correct": ok,
                "scorable": True, "detail": {"pred": pred, "gold": gold}}
    if fam == "ner":
        pred = _parse_entities(raw)
        gold = {(e["text"].strip(), _TYPE_SYNONYMS.get(e["type"], e["type"]))
                for e in item["answer"].get("entities", [])}
        p, r, f = _set_f1(pred, gold)
        return {"metric": "f1", "score": round(f, 4), "correct": None, "scorable": True,
                "detail": {"precision": round(p, 4), "recall": round(r, 4),
                           "n_pred": len(pred), "n_gold": len(gold)}}
    if fam == "formula":
        pred = _parse_herbs(raw)
        gold = {str(i.get("herb", "")).strip()
                for i in (item.get("answer") or {}).get("ingredients", []) if i.get("herb")}
        p, r, f = _set_f1(pred, gold)
        return {"metric": "f1", "score": round(f, 4), "correct": None, "scorable": True,
                "detail": {"precision": round(p, 4), "recall": round(r, 4),
                           "n_pred": len(pred), "n_gold": len(gold)}}
    if fam == "punctuation":
        pred = _boundaries(_parse_punct(raw))
        gold = _boundaries(str(item.get("answer", "")))
        p, r, f = _set_f1(pred, gold)
        return {"metric": "f1", "score": round(f, 4), "correct": None, "scorable": True,
                "detail": {"precision": round(p, 4), "recall": round(r, 4)}}
    return {"metric": "none", "score": None, "correct": None, "scorable": False, "detail": {}}


# --------------------------------------------------------------------------
# Running an evaluation
# --------------------------------------------------------------------------
def _short(text: str, n: int = 300) -> str:
    text = text.replace("\n", " ")
    return text if len(text) <= n else text[:n] + "…"


def evaluate_item(
    item: dict, client, *, mode: str = "zero_shot", shots: list | None = None,
    max_retries: int = 4, model_label: str | None = None,
) -> dict:
    system, user = build_eval_prompt(item, mode, shots)
    rec = {
        "item_id": item["item_id"], "task_code": item["task_code"],
        "family": family(item), "prompt_mode": mode,
        "model": model_label or getattr(client, "model", "model"),
    }
    try:
        raw = complete_with_retry(client, system, user, max_tokens=1500,
                                  temperature=0.0, max_retries=max_retries)
    except Exception as e:  # noqa: BLE001
        rec.update({"metric": _metric_for(item), "score": 0.0 if scorable(item) else None,
                    "correct": False if scorable(item) else None, "scorable": scorable(item),
                    "pred": "", "error": f"{type(e).__name__}: {e}"})
        return rec
    rec.update(score_prediction(item, raw))
    rec["pred"] = _short(raw)
    rec["error"] = None
    return rec


def _shots_by_family(pool: Iterable[dict], n_shots: int) -> dict[str, list]:
    by_fam: dict[str, list] = defaultdict(list)
    for it in pool:
        if scorable(it) and len(by_fam[family(it)]) < n_shots + 5:
            by_fam[family(it)].append(it)
    return by_fam


def evaluate_dataset(
    items: Iterable[dict], client, *, mode: str = "zero_shot",
    shots_pool: Iterable[dict] | None = None, n_shots: int = 3,
    max_workers: int = 8, stats: dict | None = None, model_label: str | None = None,
) -> Iterator[dict]:
    """Yield an eval record per item, concurrently and as each completes."""
    from .concurrency import bounded_imap

    by_fam = _shots_by_family(shots_pool or [], n_shots) if mode == "few_shot" else {}

    def run(item: dict) -> dict:
        shots = [s for s in by_fam.get(family(item), []) if s["item_id"] != item["item_id"]][:n_shots]
        return evaluate_item(item, client, mode=mode, shots=shots, model_label=model_label)

    for rec in bounded_imap(run, items, max_workers=max_workers, stats=stats):
        if rec is not None:
            yield rec


def aggregate(records: Iterable[dict]) -> dict:
    """Per-task and overall scores from eval records."""
    records = list(records)
    by_task: dict[str, list] = defaultdict(list)
    for r in records:
        if r.get("scorable"):
            by_task[r["task_code"]].append(r)
    per_task = {}
    for tc, rs in sorted(by_task.items()):
        per_task[tc] = {
            "metric": rs[0]["metric"],
            "score": round(sum(x["score"] for x in rs) / len(rs), 4),
            "n": len(rs),
        }
    task_scores = [v["score"] for v in per_task.values()]
    scored = sum(len(v) for v in by_task.values())
    return {
        "n_total": len(records),
        "n_scored": scored,
        "n_unscored": len(records) - scored,
        "n_errors": sum(1 for r in records if r.get("error")),
        "per_task": per_task,
        "overall_macro": round(sum(task_scores) / len(task_scores), 4) if task_scores else None,
    }
