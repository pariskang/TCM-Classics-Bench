"""Command-line entry points for the TCM-Classics-Bench pipeline.

    python -m tcm_bench catalog  --root corpus_src/book --out data/catalog.jsonl
    python -m tcm_bench ingest   --root corpus_src/book --out data/pilot \
                                 --pilot pilot1_classics
    python -m tcm_bench generate --corpus data/pilot/pilot1_classics.jsonl \
                                 --out data/samples/items.jsonl --tasks T1 T6
    python -m tcm_bench validate --items data/samples/items.jsonl \
                                 --corpus data/pilot/pilot1_classics.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from . import ingest, taxonomy
from .generate import LLMGenerator, balanced_take, generate_items
from .llm import PROVIDERS, make_client
from .validate import validate_item


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _write_jsonl(path: Path, rows) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def cmd_catalog(args) -> None:
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = ingest.write_catalog(Path(args.root), out)
    print(f"catalog: {n} books -> {out}")


def cmd_ingest(args) -> None:
    root = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pilots = args.pilot or list(taxonomy.PILOT_CORPORA)
    for pilot in pilots:
        books = taxonomy.PILOT_CORPORA[pilot]
        out = out_dir / f"{pilot}.jsonl"
        n = ingest.write_corpus(root, out, books, args.date, min_chars=args.min_chars)
        print(f"ingest {pilot}: {n} passages -> {out}")


def _make_llm(args):
    return LLMGenerator(make_client(args.provider, args.model))


def cmd_generate(args) -> None:
    records = _read_jsonl(Path(args.corpus))
    llm = _make_llm(args) if args.llm else None
    items = (item.to_dict() for item in generate_items(records, args.tasks, llm=llm))
    n = _write_jsonl(Path(args.out), items)
    print(f"generate: {n} candidate items -> {args.out}")


def _ingest_pilots(root: Path, pilots, date: str, min_chars: int) -> list[dict]:
    records: list[dict] = []
    for pilot in pilots:
        for name in taxonomy.PILOT_CORPORA[pilot]:
            d = root / name
            if (d / "index.txt").exists():
                records.extend(
                    r.to_dict() for r in ingest.ingest_book(d, date, min_chars=min_chars)
                )
    return records


def cmd_simple(args) -> None:
    """简易模式: one command -> a balanced, validated N-item test set.

    Defaults to the deterministic generators (T1 + T6), so it needs no API key
    and produces ~5k questions in seconds.  Add --llm for model-backed tasks.
    """
    root = Path(args.root)
    pilots = args.pilot or list(taxonomy.PILOT_CORPORA)
    records = _ingest_pilots(root, pilots, args.date, args.min_chars)
    random.Random(args.seed).shuffle(records)
    src = {r["passage_id"]: r["raw_text_trad"] for r in records}

    llm = _make_llm(args) if args.llm else None
    valid: list[dict] = []
    failed = 0
    for item in generate_items(records, args.tasks, llm=llm):
        d = item.to_dict()
        if validate_item(d, src.get(d["passage_id"], "")).ok:
            valid.append(d)
        else:
            failed += 1
        # With an LLM each item is an API call — stop as soon as we have N.
        if llm is not None and len(valid) >= args.n:
            break

    items = valid[: args.n] if llm is not None else balanced_take(valid, args.n)
    out = Path(args.out)
    n = _write_jsonl(out, items)
    by_task = dict(Counter(i["task_code"] for i in items))
    print(
        f"simple: {n} items -> {out}  (requested {args.n}, "
        f"{failed} failed validation)\n  by task: {by_task}\n"
        f"  books covered: {len({i['book_id'] for i in items})}"
    )


def cmd_validate(args) -> None:
    items = _read_jsonl(Path(args.items))
    corpus = {r["passage_id"]: r for r in _read_jsonl(Path(args.corpus))}
    ok = bad = 0
    for item in items:
        src = corpus.get(item["passage_id"], {}).get("raw_text_trad", "")
        res = validate_item(item, src)
        if res.ok:
            ok += 1
        else:
            bad += 1
            print(f"FAIL {item['item_id']}: {'; '.join(res.errors)}")
    print(f"validate: {ok} ok, {bad} failed")


def _add_llm_args(c: argparse.ArgumentParser) -> None:
    c.add_argument("--llm", action="store_true", help="enable LLM-backed tasks (T2-T12)")
    c.add_argument(
        "--provider", default="anthropic", choices=sorted(PROVIDERS),
        help="LLM provider: anthropic | azure | poe | litellm",
    )
    c.add_argument(
        "--model", default=None,
        help="model / deployment id (provider default used if omitted)",
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tcm_bench", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("catalog", help="write per-book metadata catalog")
    c.add_argument("--root", required=True)
    c.add_argument("--out", default="data/catalog.jsonl")
    c.set_defaults(func=cmd_catalog)

    c = sub.add_parser("ingest", help="ingest pilot corpora to JSONL")
    c.add_argument("--root", required=True)
    c.add_argument("--out", default="data/pilot")
    c.add_argument("--pilot", nargs="*", choices=list(taxonomy.PILOT_CORPORA))
    c.add_argument("--date", default="2026-06-26")
    c.add_argument("--min-chars", type=int, default=12)
    c.set_defaults(func=cmd_ingest)

    c = sub.add_parser("generate", help="generate candidate bench items")
    c.add_argument("--corpus", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--tasks", nargs="+", default=["T1", "T6"])
    _add_llm_args(c)
    c.set_defaults(func=cmd_generate)

    c = sub.add_parser("simple", help="简易模式: turnkey N-item test set (default 5000)")
    c.add_argument("--root", required=True, help="extracted archive dir (…/book)")
    c.add_argument("--out", default="data/simple_5k.jsonl")
    c.add_argument("--n", type=int, default=5000, help="number of questions (default 5000)")
    c.add_argument("--pilot", nargs="*", choices=list(taxonomy.PILOT_CORPORA))
    c.add_argument("--tasks", nargs="+", default=["T1", "T6"])
    c.add_argument("--seed", type=int, default=20260626)
    c.add_argument("--date", default="2026-06-26")
    c.add_argument("--min-chars", type=int, default=20)
    _add_llm_args(c)
    c.set_defaults(func=cmd_simple)

    c = sub.add_parser("validate", help="source-grounded validation")
    c.add_argument("--items", required=True)
    c.add_argument("--corpus", required=True)
    c.set_defaults(func=cmd_validate)
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
