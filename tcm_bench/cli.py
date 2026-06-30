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
import sys
from collections import Counter
from pathlib import Path

from . import ingest, taxonomy
from .evaluate import PROMPT_MODES
from .generate import (
    DETERMINISTIC,
    LLMGenerator,
    balanced_take,
    generate_items,
    generate_items_concurrent,
)
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
    return LLMGenerator(
        make_client(args.provider, args.model),
        difficulty=getattr(args, "difficulty", "Hard"),
    )


def _resolve_llm(args):
    """Build the LLM generator only if --llm AND at least one LLM task.

    Warns (and returns None) on the common '--llm but only T1/T6' mistake, so
    no client/SDK is needed for a deterministic run.
    """
    if not args.llm:
        return None
    if not (set(args.tasks) - DETERMINISTIC):
        print(
            "WARNING: --llm was set but --tasks contains only deterministic "
            f"tasks {sorted(DETERMINISTIC)}; no LLM call will be made. "
            "Add LLM tasks, e.g. --tasks T2 T3 T8 T9 T11.",
            file=sys.stderr,
        )
        return None
    return _make_llm(args)


def _gen_stream(records, tasks, *, llm, workers, skip=None, stats=None):
    """Pick the serial or concurrent generator and yield item dicts."""
    if llm is not None and workers > 1:
        gen = generate_items_concurrent(
            records, tasks, llm=llm, max_workers=workers, skip=skip, stats=stats
        )
    else:
        gen = generate_items(records, tasks, llm=llm)
    for item in gen:
        yield item.to_dict()


def _stream_write(path: Path, item_dicts, *, limit=None, progress=False) -> list[dict]:
    """Append each item to *path* as it is produced (real-time persistence).

    Returns the list of written items.  Shows a tqdm bar when *progress* and
    tqdm is installed; otherwise prints a periodic counter.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    bar = None
    if progress:
        try:
            from tqdm.auto import tqdm

            bar = tqdm(total=limit, unit="item", desc="generate")
        except ImportError:
            bar = None
    written: list[dict] = []
    with path.open("w", encoding="utf-8") as fh:
        for d in item_dicts:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
            fh.flush()
            written.append(d)
            if bar is not None:
                bar.update(1)
            elif progress and len(written) % 100 == 0:
                print(f"  ... {len(written)} items", flush=True)
            if limit is not None and len(written) >= limit:
                break
    if bar is not None:
        bar.close()
    return written


def cmd_generate(args) -> None:
    records = _read_jsonl(Path(args.corpus))
    llm = _resolve_llm(args)
    stream = _gen_stream(records, args.tasks, llm=llm, workers=args.workers)
    written = _stream_write(Path(args.out), stream, progress=args.progress)
    print(f"generate: {len(written)} candidate items -> {args.out}")


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

    llm = _resolve_llm(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    failed = 0

    if llm is None:
        # Deterministic (fast): generate the whole pool, then balance to N so
        # books/tasks are evenly represented.
        valid = []
        for d in _gen_stream(records, args.tasks, llm=None, workers=1):
            if validate_item(d, src.get(d["passage_id"], "")).ok:
                valid.append(d)
            else:
                failed += 1
        items = balanced_take(valid, args.n)
        _write_jsonl(out, items)
    else:
        # LLM path: concurrent + stream each validated item to disk in real
        # time + progress bar; stop once we have N (each item is an API call).
        bar = None
        if args.progress:
            try:
                from tqdm.auto import tqdm

                bar = tqdm(total=args.n, unit="item", desc="simple")
            except ImportError:
                bar = None
        items = []
        llm_stats: dict = {}
        with out.open("w", encoding="utf-8") as fh:
            for d in _gen_stream(
                records, args.tasks, llm=llm, workers=args.workers, stats=llm_stats
            ):
                if validate_item(d, src.get(d["passage_id"], "")).ok:
                    items.append(d)
                    fh.write(json.dumps(d, ensure_ascii=False) + "\n")
                    fh.flush()
                    if bar is not None:
                        bar.update(1)
                else:
                    failed += 1
                if len(items) >= args.n:
                    break
        if bar is not None:
            bar.close()
        if llm_stats:
            print(
                f"  llm: jobs={llm_stats.get('jobs_total')} "
                f"produced={llm_stats.get('yielded')} "
                f"errored={llm_stats.get('errored')} "
                f"parse_failed={llm_stats.get('parse_failed')}",
                file=sys.stderr,
            )

    by_task = dict(Counter(i["task_code"] for i in items))
    print(
        f"simple: {len(items)} items -> {out}  (requested {args.n}, "
        f"{failed} failed validation)\n  by task: {by_task}\n"
        f"  books covered: {len({i['book_id'] for i in items})}"
    )


def cmd_evaluate(args) -> None:
    """Score a model on a benchmark JSONL, with resume + real-time writes."""
    import time

    from . import evaluate as ev

    items = _read_jsonl(Path(args.items))
    if args.tasks:
        items = [it for it in items if it["task_code"] in set(args.tasks)]
    if args.limit:
        items = items[: args.limit]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.resume and out.exists():
        for r in _read_jsonl(out):
            done.add((r["item_id"], r["prompt_mode"]))
    todo = [it for it in items if (it["item_id"], args.mode) not in done]
    print(f"evaluate: {len(items)} items, {len(todo)} to do "
          f"(resumed {len(done)}), mode={args.mode}, model={args.model or args.provider}")

    client = make_client(args.provider, args.model)
    stats: dict = {}
    bar = None
    if args.progress:
        try:
            from tqdm.auto import tqdm

            bar = tqdm(total=len(todo), unit="item", desc=f"eval/{args.mode}")
        except ImportError:
            bar = None

    records = list(_read_jsonl(out)) if (args.resume and out.exists()) else []
    start = time.time()
    n = 0
    with out.open("a" if args.resume else "w", encoding="utf-8") as fh:
        for rec in ev.evaluate_dataset(
            todo, client, mode=args.mode, shots_pool=items, n_shots=args.shots,
            max_workers=args.workers, stats=stats,
            model_label=args.model or args.provider,
        ):
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            records.append(rec)
            n += 1
            if bar is not None:
                bar.update(1)
            elif args.progress and n % 20 == 0:
                rate = n / max(time.time() - start, 1e-6)
                eta = (len(todo) - n) / rate if rate else 0
                print(f"  {n}/{len(todo)}  {rate:.1f}/s  ETA {eta/60:.1f}min", flush=True)
    if bar is not None:
        bar.close()

    summary = ev.aggregate(records)
    print("\n=== scores ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.scores_out:
        Path(args.scores_out).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"scores -> {args.scores_out}")


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
    c.add_argument(
        "--workers", type=int, default=8,
        help="concurrent LLM calls (used only with --llm; default 8)",
    )
    c.add_argument(
        "--difficulty", default="Hard", choices=["Medium", "Hard", "Expert"],
        help="LLM item difficulty (default Hard)",
    )
    c.add_argument(
        "--progress", action="store_true", help="show a tqdm progress bar",
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

    c = sub.add_parser("evaluate", help="score a model on a benchmark JSONL")
    c.add_argument("--items", required=True, help="benchmark items JSONL")
    c.add_argument("--out", required=True, help="per-item eval records JSONL")
    c.add_argument("--scores-out", default=None, help="write aggregate scores JSON here")
    c.add_argument("--mode", default="zero_shot", choices=list(PROMPT_MODES))
    c.add_argument("--shots", type=int, default=3, help="few-shot examples (mode=few_shot)")
    c.add_argument("--tasks", nargs="*", help="restrict to these task codes")
    c.add_argument("--limit", type=int, default=None)
    c.add_argument("--provider", default="anthropic", choices=sorted(PROVIDERS))
    c.add_argument("--model", default=None)
    c.add_argument("--workers", type=int, default=8)
    c.add_argument("--resume", action="store_true", help="continue a prior run")
    c.add_argument("--progress", action="store_true", help="progress bar + ETA")
    c.set_defaults(func=cmd_evaluate)
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
