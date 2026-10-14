# Deployment guide

Nothing here has been deployed yet. This is the prepared, tested-where-possible setup.

## What ships for hosting

| Piece | Purpose |
|---|---|
| `data/demo/voc_demo.db` (676 KB) | 1,000 reviews with sentiment, aspects (sentence+clause), 9 topics and a cached Gemini strategy, so the hosted app opens on a finished analysis in seconds. Seeded into `data/voc.db` on first start (`db.ensure_demo_db`). Rebuild with `scripts/build_demo_db.py`; a test fails if it goes out of date. See `data/demo/README.md` for the data licence note. |
| Lazy model loading | `torch`/`transformers` are imported only when something needs scoring (analysing an upload, rebuilding aspects), and topic discovery loads BERTopic only on click. Viewing the demo imports neither (tested). |
| `requirements.txt` | Fully pinned; torch from the PyTorch **CPU** index (the default Linux PyPI wheel pulls ~2 GB of CUDA libraries). Verified that `torch-2.13.0+cpu`, hdbscan, numba, llvmlite, pyarrow, scikit-learn, pandas, spaCy, tokenizers and safetensors all have Linux x86-64 cp313 wheels. `requirements-dev.txt` adds pytest. |
| `.streamlit/config.toml` | headless, no file watcher, 20 MB upload limit. |
| Secrets | `GEMINI_API_KEY` (and optionally `GEMINI_MODEL`, `GEMINI_FALLBACK_MODEL`, `MAX_ANALYZE_ROWS`) are read from `st.secrets`, falling back to environment variables / `.env`. Template: `.streamlit/secrets.toml.example`. |
| `MAX_ANALYZE_ROWS` | Optional cap on rows per upload (e.g. `2000`) so a visitor can't push a small host out of memory. |
| `Dockerfile` | python:3.13-slim, non-root user, CPU torch, models baked into the image, honours `$PORT` (default 7860). **Not build-tested** (Docker Desktop wasn't running when this was prepared). |
| `.github/workflows/ci.yml` | Installs `requirements-dev.txt`, runs `pytest -m "not slow"` on every push/PR. |

## Which host?

**Short answer: Streamlit Community Cloud for the free public demo; a Docker host with ≥ 4 GB RAM (Hugging Face Spaces, or any small VM) if you want visitors to run the full pipeline.**

What the app needs, roughly:

| Activity | Memory (estimate, not measured on a host) |
|---|---|
| Viewing the demo (no torch import) | a few hundred MB |
| Analysing an upload (spaCy + RoBERTa-base on CPU) | ~1.5–2 GB |
| Topic discovery (BERTopic + MiniLM + UMAP + HDBSCAN on top) | ~2–3 GB |

**Streamlit Community Cloud** — free, deploys straight from the GitHub repo, native `st.secrets`. Its documented resource range is **690 MB minimum to 2.7 GB maximum per app** (docs figure dated February 2024; Streamlit says limits can change without notice, so confirm on the current docs). Viewing the demo fits comfortably; analysing uploads or running topic discovery may hit the ceiling, and a hosted app that goes over its limit is throttled or shut down. Set `MAX_ANALYZE_ROWS` (e.g. 1000–2000) and treat it as a demo, not a service. It also does not use the Dockerfile.

**Hugging Face Spaces (Docker SDK)** — the Space documentation lists CPU Basic as **2 vCPU / 16 GB RAM**, which fits every code path, and it runs the `Dockerfile` as-is (port 7860, uid 1000). One caution: the documentation and the community disagree on whether free CPU Spaces are still available for Docker apps (some sources say a paid plan is required for compute Spaces), so check the Space-creation page before counting on it being free. It needs a YAML header at the top of the README (`sdk: docker`, `app_port: 7860`); I did not add that to the main README because it would show as a table on GitHub. Put it in the Space's own README.

**Recommendation:** start on Streamlit Community Cloud (free, the demo database makes it instant, `MAX_ANALYZE_ROWS=1500`), and keep the Dockerfile as the path to Hugging Face Spaces or a small VM if you outgrow it. Do not try to run the full 3,000-review pipeline on Community Cloud.

## Steps (when you're ready)

**Streamlit Community Cloud**
1. Push the repo; on share.streamlit.io choose the repo, branch `main`, main file `dashboard/app.py`, Python 3.13.
2. Under *Advanced settings → Secrets* paste the contents of `.streamlit/secrets.toml.example` with your real key and `MAX_ANALYZE_ROWS = 1500`.
3. First boot installs the pinned requirements (several minutes); after that the app opens on the demo dataset.

**Docker / Hugging Face Spaces**
```bash
docker build -t voc-intel .
docker run -p 7860:7860 -e GEMINI_API_KEY=your_key voc-intel   # then open http://localhost:7860
```
For a Space: create it with SDK *Docker*, add the YAML header to the Space README, push this repo's files, and add `GEMINI_API_KEY` under *Settings → Secrets*.

## Known gaps

- The Dockerfile has not been built. Wheels exist for every pinned package, but build it once locally before relying on it.
- Memory figures above are estimates; none were measured on a hosted instance.
- The demo uses a licensed third-party dataset (see `data/demo/README.md`); confirm the terms allow public redistribution before making the repo or the app public.
- Hosted filesystems are ephemeral: uploads and strategies made by visitors vanish on restart; the demo database is re-seeded each time.
