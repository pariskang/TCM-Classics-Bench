"""Question generation: deterministic generators + a pluggable LLM client.

Two deterministic generators need no API key and are *source-grounded by
construction*:

    T1  punctuation restoration  — strip punctuation from already-punctuated
        source text; the answer is the original.
    T6  formula structure parsing — read the ``<F>`` blocks already extracted
        during ingestion; every field is copied verbatim from the source.

The LLM-backed path (``LLMGenerator``) drives the prompt templates for tasks
that genuinely need a model — over any provider in :mod:`tcm_bench.llm`
(Anthropic, Azure, Poe, LiteLLM) — and parses the JSON reply into a BenchItem.
Generated items are *candidates*: they must still pass ``validate`` and human
review before entering the benchmark.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Iterator

from . import prompts
from .markup import remove_punctuation
from .schema import BenchItem, Evidence

# Herbs that are toxic / restricted under modern regulation and therefore need
# a safety note when they show up in a generated item.
SAFETY_HERBS = (
    "犀角", "麝香", "朱砂", "雄黃", "烏頭", "附子", "砒霜", "水銀",
    "輕粉", "斑蝥", "馬錢子", "硃砂", "鉛", "丹砂", "巴豆", "大戟",
)
SAFETY_NOTE = "僅作古籍理解測評，不構成用藥建議。"


def _item_id(book_id: str, task_code: str, passage_id: str, salt: str = "") -> str:
    h = hashlib.sha1(f"{passage_id}|{task_code}|{salt}".encode()).hexdigest()[:10]
    return f"{book_id}::{task_code}::{h}"


def _evidence(rec: dict, spans: list[str]) -> Evidence:
    # Prefer the most specific heading (篇/方名) for provenance; fall back to
    # the chapter, then nothing.
    path = rec.get("heading_path") or []
    chapter = " · ".join(path) if path else rec.get("chapter")
    return Evidence(
        book_title_trad=rec["book_title_trad"],
        chapter=chapter,
        base_text=rec.get("base_text"),
        spans=spans,
    )


def _safety_note_for(text: str) -> str:
    return SAFETY_NOTE if any(h in text for h in SAFETY_HERBS) else ""


# --------------------------------------------------------------------------
# T1 — punctuation restoration (deterministic).
# --------------------------------------------------------------------------
def generate_t1(rec: dict, *, min_chars: int = 20, max_chars: int = 220) -> BenchItem | None:
    text = rec["raw_text_trad"]
    if not (min_chars <= len(text) <= max_chars):
        return None
    stripped = remove_punctuation(text)
    if stripped == text:  # nothing to restore
        return None
    return BenchItem(
        item_id=_item_id(rec["book_id"], "T1", rec["passage_id"]),
        task="punctuation_restoration",
        task_code="T1",
        question="請為下列無標點古文片段恢復句讀（標點）。",
        context=stripped,
        answer=text,
        evidence=_evidence(rec, [text]),
        book_id=rec["book_id"],
        passage_id=rec["passage_id"],
        inference_level="direct",
        difficulty="Medium",
        generator="deterministic",
    )


# --------------------------------------------------------------------------
# T6 — formula structure parsing (deterministic, from <F> blocks).
# --------------------------------------------------------------------------
def generate_t6_from_formulas(rec: dict) -> Iterator[BenchItem]:
    for i, fb in enumerate(rec.get("formulas", [])):
        ingredients = fb.get("ingredients") or []
        if not fb.get("formula_name") or not ingredients:
            continue
        answer = {
            "formula_name": fb["formula_name"],
            "indication": None,
            "ingredients": [
                {
                    "herb": ing["herb"],
                    "dose": ing.get("dose") or None,
                    "preparation": ing.get("preparation") or None,
                    "evidence": f"{ing['herb']}{ing.get('dose') or ''}{ing.get('preparation') or ''}",
                }
                for ing in ingredients
            ],
            "manufacturing_method": None,
            "administration": None,
            "contraindication": None,
        }
        spans = [ing["herb"] for ing in ingredients]
        yield BenchItem(
            item_id=_item_id(rec["book_id"], "T6", rec["passage_id"], salt=str(i)),
            task="formula_structure_parsing",
            task_code="T6",
            question="請從以下古籍方劑條文中解析方名、組成與劑量。",
            context=fb.get("text") or rec["raw_text_trad"],
            answer=answer,
            evidence=_evidence(rec, spans),
            book_id=rec["book_id"],
            passage_id=rec["passage_id"],
            inference_level="direct",
            difficulty="Medium",
            safety_note=_safety_note_for(fb.get("text", "")),
            generator="deterministic",
        )


# --------------------------------------------------------------------------
# LLM-backed generation.
# --------------------------------------------------------------------------
class LLMGenerator:
    """Drive the prompt templates with any :class:`tcm_bench.llm.LLMClient`.

    Works with the Anthropic, Azure, Poe and LiteLLM clients in
    :mod:`tcm_bench.llm` — anything exposing ``complete(system, prompt)``.
    """

    def __init__(
        self,
        client,
        *,
        difficulty: str = "Hard",
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ):
        self.client = client
        self.model = getattr(client, "model", "llm")
        self.difficulty = difficulty
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(self, task_code: str, rec: dict) -> BenchItem | None:
        prompt = prompts.build_prompt(task_code, rec, self.difficulty)
        if prompt is None:
            return None
        raw = self.client.complete(
            prompts.SYSTEM, prompt, max_tokens=self.max_tokens, temperature=self.temperature
        )
        data = _extract_json(raw)
        if data is None:
            return None
        return _bench_item_from_llm(task_code, rec, data, self.model, self.difficulty)


def AnthropicGenerator(model: str = "claude-opus-4-8", api_key: str | None = None) -> LLMGenerator:
    """Backwards-compatible helper: an :class:`LLMGenerator` on Anthropic."""
    from .llm import AnthropicClient

    return LLMGenerator(AnthropicClient(model=model, api_key=api_key))


def _extract_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _answer_text(answer) -> str:
    """Flatten an answer (str | dict | list) to a string for keyword scans."""
    if isinstance(answer, str):
        return answer
    if isinstance(answer, dict):
        return " ".join(_answer_text(v) for v in answer.values())
    if isinstance(answer, list):
        return " ".join(_answer_text(v) for v in answer)
    return str(answer)


def _default_task_name(task_code: str) -> str:
    from .taxonomy import TASKS

    task = TASKS.get(task_code)
    return task.name_en if task else task_code


def _bench_item_from_llm(
    task_code: str, rec: dict, data: dict, model: str, difficulty: str = "Hard"
) -> BenchItem:
    answer = data.get("answer", data)
    spans = data.get("evidence") or data.get("evidence_spans") or []
    options = data.get("options") or []
    safety = data.get("safety_note", "")
    # Auto-fill the safety note if a restricted herb slipped through unflagged.
    blob = f"{data.get('context','')}{_answer_text(answer)}{' '.join(map(str, options))}"
    if not safety and any(h in blob for h in SAFETY_HERBS):
        safety = SAFETY_NOTE
    return BenchItem(
        item_id=_item_id(rec["book_id"], task_code, rec["passage_id"], salt=model),
        task=data.get("task") or _default_task_name(task_code),
        task_code=task_code,
        question=data.get("question", ""),
        context=data.get("context") or rec["raw_text_trad"],
        answer=answer,
        evidence=_evidence(rec, spans if isinstance(spans, list) else [str(spans)]),
        book_id=rec["book_id"],
        passage_id=rec["passage_id"],
        inference_level=data.get("inference_level", "direct"),
        difficulty=data.get("difficulty") or difficulty,
        options=options if isinstance(options, list) else [],
        distractors=data.get("distractors") or [],
        safety_note=safety,
        quality_warning=data.get("quality_warning", ""),
        generator=model,
    )


# --------------------------------------------------------------------------
# Orchestration.
# --------------------------------------------------------------------------
DETERMINISTIC = {"T1", "T6"}


def generate_items(
    records: Iterable[dict],
    tasks: Iterable[str],
    *,
    llm: "LLMGenerator | None" = None,
) -> Iterator[BenchItem]:
    """Yield candidate items for *tasks* over *records*.

    Deterministic tasks (T1, T6) always run.  Other tasks run only when an
    *llm* generator is supplied; otherwise they are skipped (no fabrication).
    Consumers may ``break`` early — this is a generator, so generation (and
    any API calls) stops as soon as you stop pulling from it.
    """
    task_set = set(tasks)
    for rec in records:
        candidate = set(rec.get("candidate_tasks", [])) & task_set
        if "T1" in candidate:
            item = generate_t1(rec)
            if item:
                yield item
        if "T6" in candidate:
            yield from generate_t6_from_formulas(rec)
        if llm is not None:
            for tc in sorted(candidate - DETERMINISTIC):
                item = llm.generate(tc, rec)
                if item:
                    yield item


def generate_items_concurrent(
    records: Iterable[dict],
    tasks: Iterable[str],
    *,
    llm: "LLMGenerator | None" = None,
    max_workers: int = 8,
    skip: "set[tuple[str, str]] | None" = None,
) -> Iterator[BenchItem]:
    """Like :func:`generate_items`, but runs the LLM tasks concurrently.

    Deterministic items (T1, T6) are produced inline and yielded first — they
    are CPU-cheap.  LLM tasks (one API call each) are fanned out across a
    thread pool and yielded **as they complete**, so a progress bar advances
    smoothly and a streaming writer can persist each item immediately.

    *skip* is a set of ``(passage_id, task_code)`` already done — used for
    **resume**: those jobs are not re-submitted, so re-running with a larger
    target continues from where a previous run stopped instead of restarting.

    Submission is bounded to ``~2 * max_workers`` jobs in flight (a sliding
    window), so a consumer that ``break``s after reaching its target wastes at
    most that many extra API calls — not the whole job list.

    Order is not preserved for the LLM items.  With ``llm=None`` this is just
    the deterministic stream (``max_workers`` ignored).
    """
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

    task_set = set(tasks)
    skip = skip or set()

    # Build deterministic items (yielded first) and the LLM job list, both
    # honouring the resume *skip* set.
    det_items: list[BenchItem] = []
    jobs: list[tuple[str, dict]] = []
    for rec in records:
        candidate = set(rec.get("candidate_tasks", [])) & task_set
        pid = rec["passage_id"]
        if "T1" in candidate and (pid, "T1") not in skip:
            item = generate_t1(rec)
            if item:
                det_items.append(item)
        if "T6" in candidate and (pid, "T6") not in skip:
            det_items.extend(generate_t6_from_formulas(rec))
        if llm is not None:
            for tc in sorted(candidate - DETERMINISTIC):
                if (pid, tc) not in skip:
                    jobs.append((tc, rec))

    yield from det_items
    if llm is None or not jobs:
        return

    window = max(1, max_workers) * 2
    idx = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        inflight: dict = {}
        while idx < len(jobs) and len(inflight) < window:
            tc, rec = jobs[idx]
            idx += 1
            inflight[pool.submit(llm.generate, tc, rec)] = True

        while inflight:
            finished, _ = wait(list(inflight), return_when=FIRST_COMPLETED)
            for fut in finished:
                del inflight[fut]
                try:
                    item = fut.result()
                except Exception:  # a single failed call must not kill the batch
                    item = None
                if idx < len(jobs):  # refill the window
                    tc, rec = jobs[idx]
                    idx += 1
                    inflight[pool.submit(llm.generate, tc, rec)] = True
                if item is not None:
                    yield item


def balanced_take(items: Iterable[dict], n: int) -> list[dict]:
    """Take up to *n* items, spread round-robin across (task_code, book_id).

    Deterministic: preserves input order within each bucket and interleaves
    buckets, so no single book or task dominates a small sample.
    """
    from collections import defaultdict, deque

    buckets: dict[tuple, deque] = defaultdict(deque)
    order: list[tuple] = []
    for it in items:
        key = (it.get("task_code"), it.get("book_id"))
        if key not in buckets:
            order.append(key)
        buckets[key].append(it)

    out: list[dict] = []
    while len(out) < n and any(buckets[k] for k in order):
        for key in order:
            if buckets[key]:
                out.append(buckets[key].popleft())
                if len(out) >= n:
                    break
    return out
