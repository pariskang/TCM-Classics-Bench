#!/usr/bin/env python3
"""Build the committed dataset artifacts from an extracted Jicheng archive.

The full ingested corpus is large (tens of MB, dominated by 本草綱目), so the
repository ships *reproducible* artifacts rather than the whole corpus:

    data/catalog.jsonl            every book's metadata (full coverage)
    data/corpus/<pilot>.jsonl     a capped, per-book-balanced passage sample
    data/bench/<pilot>.jsonl      validated deterministic candidate items
    data/STATS.json               counts used by the README / paper

Run ``scripts/download_corpus.sh`` first, then::

    python scripts/build_release.py --root corpus_src/book

Anyone can regenerate the *full* corpus with ``python -m tcm_bench ingest``.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from tcm_bench import CORPUS_VERSION, ingest, taxonomy
from tcm_bench.generate import generate_items
from tcm_bench.validate import validate_item

ROOT_OUT = Path("data")
INGEST_DATE = "2026-06-26"


def _write_jsonl(path: Path, rows: list[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def _balanced_sample(records: list[dict], per_book: int) -> list[dict]:
    """Up to *per_book* passages from each book, preferring formula passages."""
    by_book: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_book[r["book_id"]].append(r)
    out: list[dict] = []
    for book, recs in by_book.items():
        recs.sort(key=lambda r: (0 if r.get("formulas") else 1))
        out.extend(recs[:per_book])
    return out


def build(root: Path, per_book: int, bench_cap: int) -> None:
    # 1. Full catalog.
    n_books = ingest.write_catalog(root, ROOT_OUT / "catalog.jsonl")
    print(f"catalog: {n_books} books")

    stats: dict = {"corpus_version": CORPUS_VERSION, "books": n_books, "pilots": {}}

    for pilot, books in taxonomy.PILOT_CORPORA.items():
        # 2. Full ingest (in memory), then a balanced committed sample.
        full: list[dict] = []
        for name in books:
            d = root / name
            if not (d / "index.txt").exists():
                continue
            full.extend(r.to_dict() for r in ingest.ingest_book(d, INGEST_DATE))
        sample = _balanced_sample(full, per_book)

        # 3. Deterministic generation over the FULL corpus, validate, cap.
        src_by_id = {r["passage_id"]: r["raw_text_trad"] for r in full}
        items = [it.to_dict() for it in generate_items(full, ["T1", "T4", "T6"])]
        valid, failed = [], 0
        for it in items:
            res = validate_item(it, src_by_id.get(it["passage_id"], ""))
            if res.ok:
                valid.append(it)
            else:
                failed += 1

        capped = _cap_items(valid, bench_cap)
        _write_jsonl(ROOT_OUT / "bench" / f"{pilot}.jsonl", capped)

        # Dedicated NER (T4) test subset, capped per book.
        ner_items = _cap_items([it for it in valid if it["task_code"] == "T4"], bench_cap)
        if ner_items:
            _write_jsonl(ROOT_OUT / "bench_ner" / f"{pilot}.jsonl", ner_items)

        # Make the committed corpus a superset of every passage the committed
        # items cite, so the dataset can be re-validated from committed files
        # alone (no full re-ingest needed).
        by_id = {r["passage_id"]: r for r in full}
        sampled_ids = {r["passage_id"] for r in sample}
        for it in capped:
            pid = it["passage_id"]
            if pid not in sampled_ids and pid in by_id:
                sample.append(by_id[pid])
                sampled_ids.add(pid)
        _write_jsonl(ROOT_OUT / "corpus" / f"{pilot}.jsonl", sample)

        task_counts = Counter(it["task_code"] for it in items)
        stats["pilots"][pilot] = {
            "books": books,
            "passages_full": len(full),
            "passages_sampled": len(sample),
            "formula_passages": sum(1 for r in full if r.get("formulas")),
            "items_generated": len(items),
            "items_valid": len(valid),
            "items_failed_validation": failed,
            "items_committed": len(capped),
            "ner_committed": len(ner_items),
            "by_task": dict(task_counts),
        }
        print(
            f"{pilot}: {len(full)} passages, {len(items)} items "
            f"({len(valid)} valid, {failed} failed) -> {len(capped)} committed"
        )

    (ROOT_OUT / "STATS.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("wrote data/STATS.json")


def _cap_items(items: list[dict], cap: int) -> list[dict]:
    """Cap per (task, book), balanced, so no single book dominates."""
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for it in items:
        buckets[(it["task_code"], it["book_id"])].append(it)
    out: list[dict] = []
    for bucket in buckets.values():
        out.extend(bucket[:cap])
    out.sort(key=lambda it: (it["task_code"], it["book_id"], it["item_id"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="extracted archive dir (…/book)")
    ap.add_argument("--per-book", type=int, default=30, help="passage sample cap per book")
    ap.add_argument("--bench-cap", type=int, default=25, help="item cap per task per book")
    args = ap.parse_args()
    build(Path(args.root), args.per_book, args.bench_cap)


if __name__ == "__main__":
    main()
