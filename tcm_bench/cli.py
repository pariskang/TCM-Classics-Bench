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
from pathlib import Path

from . import ingest, taxonomy
from .generate import AnthropicGenerator, generate_items
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


def cmd_generate(args) -> None:
    records = _read_jsonl(Path(args.corpus))
    llm = AnthropicGenerator(model=args.model) if args.llm else None
    items = (item.to_dict() for item in generate_items(records, args.tasks, llm=llm))
    n = _write_jsonl(Path(args.out), items)
    print(f"generate: {n} candidate items -> {args.out}")


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
    c.add_argument("--llm", action="store_true", help="enable LLM-backed tasks")
    c.add_argument("--model", default="claude-opus-4-8")
    c.set_defaults(func=cmd_generate)

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
