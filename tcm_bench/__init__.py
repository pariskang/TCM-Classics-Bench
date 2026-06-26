"""TCM-Classics-Bench: a source-grounded benchmark builder for classical
Chinese-medicine texts, grounded on the 中醫笈成 / Jicheng-TCM corpus.

The package is organised as a small pipeline:

    markup    -> clean / segment the Jicheng wiki markup
    parsing   -> read a book directory into structured records
    taxonomy  -> categories, quality tiers, task (T1-T12) definitions
    schema    -> dataclasses + JSON-Schema for corpus records & bench items
    ingest    -> walk the extracted archive -> catalog + corpus JSONL
    prompts   -> LLM prompt templates used for question generation
    generate  -> orchestrate generation (deterministic + LLM-backed)
    validate  -> source-grounded validation of generated items
"""

__version__ = "0.1.0"

SOURCE_ID = "jicheng_tcm"
# The archive distributed by jicheng.tw that this builder targets.
CORPUS_VERSION = "book-20180111"
SOURCE_HOMEPAGE = "https://jicheng.tw/tcm/"
SOURCE_DOWNLOAD = "https://jicheng.tw/tcm/download.html"
SOURCE_ARCHIVE = "https://jicheng.tw/files/jcw/book-20180111.7z"
# Public-domain text editing released CC0; site-authored text CC BY 4.0.
# Per-page / embedded media may differ and must be verified individually.
SOURCE_LICENSE = (
    "CC0 for editorial work on public-domain text; "
    "CC BY 4.0 for site-authored text; verify per page"
)
