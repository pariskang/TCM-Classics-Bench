"""LLM prompt templates for source-grounded question generation.

Every template hard-codes the source-grounding constraints from the protocol:
the model may only use the supplied 原文 + metadata, must not import modern
textbook facts, must not invent herbs/doses/diseases, and must tag the
required inference level.  Templates return ``str`` ready for the LLM client.
"""

from __future__ import annotations

import json

# Shared system prompt — sets the grounding contract for every task.
SYSTEM = (
    "你是一名中醫古籍 benchmark 構建專家。你只能基於中醫笈成（Jicheng-TCM）"
    "提供的古籍原文與元數據生成測評題。\n"
    "硬性約束：\n"
    "1. 只能基於給定原文、書名、篇章、底本、朝代、作者與品質字段判斷。\n"
    "2. 不得引入現代教材知識作為事實依據。\n"
    "3. 不得補充原文中沒有出現的方劑、藥物、劑量、病名、症狀。\n"
    "4. 凡需推理，必須標註 inference_level = direct / implicit / external_required。\n"
    "5. external_required 的任務不得進入正式 benchmark。\n"
    "6. 涉及犀角、麝香、朱砂、雄黃、烏頭、附子等安全或現代禁限用藥材時，"
    "必須在 safety_note 中說明「僅作古籍理解測評，不構成用藥建議」。\n"
    "7. 所有答案必須可由 evidence_span 回指原文；輸出必須是合法 JSON。"
)


def _meta_block(rec: dict) -> str:
    """Compact JSON of the fields a generator is allowed to see."""
    keep = {
        "source": "中醫笈成",
        "book_title_trad": rec.get("book_title_trad"),
        "author": rec.get("author"),
        "dynasty": rec.get("dynasty"),
        "year": rec.get("year"),
        "base_text": rec.get("base_text"),
        "quality_score_source": rec.get("quality_score_source"),
        "chapter": rec.get("chapter"),
        "section_title": rec.get("section_title"),
        "raw_text_trad": rec.get("raw_text_trad"),
    }
    return json.dumps(keep, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------
# Task router — decide which tasks a passage can support.
# --------------------------------------------------------------------------
TASK_ROUTER_TEMPLATE = """請判斷以下中醫笈成古籍片段適合生成哪些測評任務。

可選任務：
T1 標點恢復  T2 文白翻譯  T3 術語注釋  T4 實體識別  T5 關係抽取
T6 方劑結構解析  T7 理論分類  T8 方證對應  T9 類方鑑別
T10 證據溯源問答  T11 安全禁忌判斷  T12 幻覺引用檢測

輸入：
{meta}

請輸出 JSON：
{{
  "suitable_tasks": [],
  "unsuitable_tasks": [],
  "reason": "",
  "detected_entities": {{
    "disease": [], "symptom": [], "syndrome": [], "pathogenesis": [],
    "treatment": [], "formula": [], "herb": [], "dose": [],
    "preparation": [], "administration": [], "contraindication": []
  }},
  "evidence_spans": [],
  "difficulty": "Easy/Medium/Hard/Expert",
  "quality_warning": ""
}}"""


# --------------------------------------------------------------------------
# T6 — formula structure parsing.
# --------------------------------------------------------------------------
FORMULA_PARSE_TEMPLATE = """請基於以下中醫笈成古籍片段，生成「方劑結構解析」測評題。

約束：
1. 方名、主治、組成、劑量、炮製、製法、服法必須來自原文。
2. 原文沒有出現的字段填 null。
3. 不得把現代劑量換算加入標準答案。
4. 不得刪除「童子小便」「生薑自然汁」「研飛」「酒煮」「醋淬」等古籍製法信息。
5. 涉及安全/現代禁限用藥材時，必須在 safety_note 說明僅作古籍理解測評。
6. 輸出必須包含 evidence_span。

輸入：
{{"book": "{book}", "chapter": "{chapter}", "raw_text_trad": "{raw}"}}

輸出 JSON：
{{
  "task": "formula_structure_parsing",
  "question": "請從以下古籍方劑條文中解析方名、主治、組成、劑量、炮製、製法與服法。",
  "context": "",
  "answer": {{
    "formula_name": "",
    "indication": "",
    "ingredients": [{{"herb": "", "dose": "", "preparation": "", "evidence": ""}}],
    "manufacturing_method": "",
    "administration": "",
    "contraindication": null
  }},
  "evidence": [],
  "safety_note": "",
  "difficulty": ""
}}"""


# --------------------------------------------------------------------------
# T2 — classical -> vernacular translation.
# --------------------------------------------------------------------------
TRANSLATION_TEMPLATE = """請基於以下中醫笈成古籍片段，生成「文白翻譯」測評題。
標準答案為現代白話翻譯，須忠於原文、不增不減、不引入現代醫學解釋。

輸入：
{meta}

輸出 JSON：
{{
  "task": "classical_translation",
  "question": "請將下列古文片段翻譯為現代白話。",
  "context": "<原文>",
  "answer": "<白話翻譯>",
  "evidence": ["<原文 span>"],
  "inference_level": "direct",
  "difficulty": ""
}}"""


def task_router_prompt(rec: dict) -> str:
    return TASK_ROUTER_TEMPLATE.format(meta=_meta_block(rec))


def formula_parse_prompt(rec: dict) -> str:
    return FORMULA_PARSE_TEMPLATE.format(
        book=rec.get("book_title_trad", ""),
        chapter=rec.get("chapter") or "",
        raw=rec.get("raw_text_trad", ""),
    )


def translation_prompt(rec: dict) -> str:
    return TRANSLATION_TEMPLATE.format(meta=_meta_block(rec))


PROMPT_BUILDERS = {
    "router": task_router_prompt,
    "T6": formula_parse_prompt,
    "T2": translation_prompt,
}
