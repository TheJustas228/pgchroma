# Contributing

Bug reports, compatibility gaps and pull requests are welcome. The most useful report is a few
lines of `chromadb` client code where pgchroma answers differently from Chroma.

## Setup

```bash
uv sync
uv run pytest            # uses $DATABASE_URL, or starts an embedded Postgres (pgserver)
uv run ruff check . && uv run ruff format --check .
```

`tests/test_compat.py` runs every scenario against a real Chroma server (started from the
`chromadb` wheel) and against pgchroma, and compares the results. To pin down a behavior, add a
scenario there: it fails until pgchroma matches Chroma. The differential tests need Python
3.11+; `tests/test_js_client.py` needs Node.js.

Features single-node Chroma cannot run (the Search API, sparse vectors) are tested against the
behavior in Chroma's source and docs instead; say which in the test.

## Pull requests

- One change per pull request, with a test that fails without it.
- Match the surrounding code; keep the README's compatibility table honest.
- README images are generated: edit `docs/assets/render.py`, then run
  `uv run --no-project --with fonttools --with uharfbuzz --with resvg-py python docs/assets/render.py`.
