# Data licensing & provenance

## Source corpus — 中醫笈成 / Jicheng-TCM

The classical texts that this benchmark is built from come from
[中醫笈成 (jicheng.tw)](https://jicheng.tw/tcm/), a free library that
collects, collates, and punctuates Chinese-medicine classics.

Per the Jicheng copyright statement:

- Editorial work (selection, arrangement, **punctuation**, annotation) on
  **public-domain** texts is released under **CC0**.
- Text **authored by the site itself** is released under **CC BY 4.0**.
- Individual pages may carry their own copyright notice, and embedded images
  or multimedia may have different licensing. The site notes that license
  labelling is not guaranteed to be error-free.

**You must verify licensing per page / per book before redistribution.** The
`source_url`, `base_text`, and `version_commit` fields in every record exist
so each item can be traced back to its exact source for that verification.

This builder targets the archive snapshot
[`book-20180111.7z`](https://jicheng.tw/files/jcw/book-20180111.7z), recorded
as `version_commit: "book-20180111"` on every record.

## This repository

- **Code** (`tcm_bench/`, `scripts/`, `tests/`) — released under the MIT
  License (see `LICENSE`).
- **Generated benchmark items** (`data/bench/`) — derived works that quote
  short public-domain spans plus machine-generated questions/answers. They
  inherit the CC0 status of the underlying public-domain editorial text;
  treat the question scaffolding as CC0 as well. Re-verify any item whose
  evidence may touch site-authored (CC BY 4.0) or third-party content.
- **Catalog & corpus samples** (`data/catalog.jsonl`, `data/corpus/`) —
  metadata and short public-domain excerpts, provided for reproducibility.

## Safety notice

Items that mention toxic or modern-restricted materia medica (e.g. 烏頭, 附子,
朱砂, 雄黃, 犀角, 麝香) carry a `safety_note`: these items exist **only for
evaluating classical-text understanding and are not medical advice.**
