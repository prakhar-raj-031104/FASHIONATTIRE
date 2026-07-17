# Multimodal Fashion & Context Retrieval

An intelligent image search engine that retrieves fashion images from natural-language
descriptions — understanding **what** someone wears, **where** they are, and the **vibe**
of the outfit. Built to beat a vanilla CLIP baseline on the fashion-specific failure
modes the assignment calls out (compositionality and fine-grained attributes).

```
"A red tie and a white shirt in a formal setting."   → the right images, not the color-swapped trap
"Casual weekend outfit for a city walk."             → inferred (hoodie/jeans/street) with no garment words in the query
```

---

## TL;DR — why this is better than "CLIP + FAISS"

Vanilla CLIP embeds the whole image into **one** vector, so it behaves like a bag of
concepts: it cannot tell *"red shirt, blue pants"* from *"blue shirt, red pants"*, and it
under-weights fine-grained fashion attributes. This system adds **structure** on top of a
fashion-domain encoder:

| Signal | Model | Failure mode it fixes | Eval query |
|---|---|---|---|
| **Global similarity** | FashionCLIP | generic semantics / scene | all |
| **Caption similarity** | BLIP → MPNet | style/vibe *inference* (no garment words) | "casual weekend city walk" |
| **Attribute match** | FashionCLIP zero-shot tagger | explicit multi-attribute grounding | "yellow raincoat", "business office" |
| **Region binding** | SegFormer + FashionCLIP | **compositional color↔garment binding** | "red tie **and** white shirt" |

The four signals are min-max normalized and fused with configurable weights; inactive
signals (e.g. no color-garment binding in the query) are dropped and their weight
redistributed. See [`docs/WRITEUP.md`](docs/WRITEUP.md) for the full rationale, trade-offs,
and future work — that document is the source for the submission PDF.

---

## Architecture

```
                         INDEXING (Part A)                         RETRIEVAL (Part B)
   raw image                                            query text
      │                                                     │
      ├─ FashionCLIP  ─────────► global.faiss   ◄──ANN───   ├─ FashionCLIP (text)  → q_clip
      │                                                     ├─ MPNet (text)        → q_sent
      ├─ BLIP caption ─► MPNet ─► caption.faiss  ◄──ANN───  │
      │                                                     ▼
      ├─ CLIP zero-shot tagger ─► attributes ──► SQLite   candidate pool (union of ANN hits)
      │                                                     │
      └─ SegFormer regions ─► crops ─► FashionCLIP          ▼
                          ├─ region.faiss                HybridScorer
                          └─ colors/type ──► SQLite   (global + caption + attribute + region)
                                                            │
                                                            ▼   (optional cross-encoder rerank)
                                                        Top-K images
```

**Two-stage retrieval** (ANN recall → feature-rich rerank) is the same pattern used at
web scale, so the ranking logic is identical whether the corpus is 1k or 1M images — only
the FAISS index type changes (flat → IVF/HNSW).

---

## Project layout — *logic is separated from data*

```
configs/            # ALL tunable knobs (models, paths, weights). No hardcoding in code.
  models.yaml         swap any checkpoint in one line
  paths.yaml          every filesystem location
  retrieval.yaml      scoring weights + search params
src/
  utils/            config loader (+device resolution), logger, typed schema (data contracts)
  attributes/       controlled vocab, CLIP zero-shot tagger, rule-based query parser
  models/           thin wrappers: FashionCLIP, BLIP, MPNet, SegFormer, cross-encoder
  database/         SQLite metadata store + FAISS vector store (3 indexes)
  indexing/         dataset loader (dataset-agnostic) + indexer orchestration
  pipelines/        model factories (single config→object mapping)
  retrieval/        HybridScorer (4 signals + fusion) + Retriever (2-stage)
  evaluation/       the 5 official queries + contact-sheet + precision@k
tests/              pure-logic unit tests (query parsing + compositional binding)
scripts/            optional hybrid dataset builder (Fashionpedia + COCO)
main.py             CLI: index / query / evaluate
docs/WRITEUP.md     the submission write-up (approaches, chosen arch, future work)
```

Vectors live in FAISS; **metadata lives in SQLite** — never in filenames. Every pipeline
stage passes typed `ImageRecord` / `RegionRecord` / `QuerySpec` objects.

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# GPU strongly recommended. The code auto-detects CUDA; on CPU it forces float32 and the
# BLIP/SegFormer stages get slow (turn them off in configs/models.yaml if CPU-only).
```

> **GPU note:** if `nvidia-smi` fails ("couldn't communicate with the NVIDIA driver"),
> the driver isn't loaded — fix that first (reboot / reinstall driver), otherwise indexing
> falls back to CPU.

### 1. Get a dataset (≥ 500–1000 images) into `data/raw/`

Any folder of images works — the loader is dataset-agnostic. For the best match to the
evaluation axes, build a **hybrid** (fine-grained garments + diverse scenes):

```bash
pip install datasets
python scripts/prepare_dataset.py --fashionpedia 700 --coco-person 300
```

…or just drop your own images into `data/raw/` (subfolders are fine).

### 2. Build the index (Part A)

```bash
python main.py index
# → indexes/global.faiss, caption.faiss, region.faiss, metadata.sqlite
```

### 3. Query (Part B)

```bash
python main.py query "a red tie and a white shirt in a formal setting" -k 10
python main.py query "someone wearing a blue shirt sitting on a park bench"
```

Each result prints its fused score **and the per-signal breakdown**, so you can see *why*
it ranked where it did.

### 3b. (Optional) Web portal — search visually + inspect scoring

```bash
pip install flask          # already in requirements.txt
python webapp/app.py       # then open http://localhost:5000
```

- **Search** page: describe a scene, get matching images. Each search shows **how the query
  was parsed** (attributes + color→garment bindings) and every result's **4-signal score
  breakdown**, so the retrieval logic is transparent.
- **Evaluation** page: runs the 5 official queries live with an explanation of exactly how
  scoring works — a self-serve way to verify system behaviour.

The portal is a thin Flask layer over the same `Retriever` the CLI uses — the browser shows
exactly what the ML system returns.

### 4. Evaluate the 5 official queries

```bash
python main.py evaluate            # writes outputs/results/eval.json + eval.html
# open outputs/results/eval.html for a visual contact sheet
# optional precision@k: pass --labels labels.json  ({ "<query>": [relevant_image_ids] })
```

---

## Design choices worth defending (short version)

- **FashionCLIP over generic CLIP/OpenCLIP** — domain encoder trained on ~800k fashion
  pairs; better garment/color/style grounding for essentially free.
- **CLIP zero-shot tagging over an LLM attribute extractor** — grounds attributes in
  *pixels*, not in a caption an LLM might hallucinate; deterministic, reproducible, cheap.
- **SegFormer region decomposition** — the one component that genuinely fixes
  compositional binding; dataset-agnostic (no reliance on Fashionpedia masks).
- **BLIP-large over BLIP-2** — captions are generated once, offline; BLIP-2's marginal
  quality isn't worth ~7× the memory.
- **Cross-encoder rerank left OFF by default** — over captions it mostly re-scores signal
  we already have; wired in so its effect can be measured honestly, not assumed.

Full reasoning, alternatives, and their trade-offs are in [`docs/WRITEUP.md`](docs/WRITEUP.md).

## Tests

```bash
pip install pytest && pytest -q      # or run the two files directly
```

`tests/test_scorer.py` is the headline check: two images identical to whole-image CLIP
(red-tie/white-shirt vs the swap) are correctly disambiguated by the region-binding
signal.
