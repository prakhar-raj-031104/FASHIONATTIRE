# Multimodal Fashion & Context Retrieval

Natural-language image search over **3,200 fashion images** that understands **what** someone
wears, **where** they are, and the **vibe** of the outfit.

```
"A red tie and a white shirt in a formal setting."   → binds each colour to the right garment
"Casual weekend outfit for a city walk."             → infers hoodie/jeans/street with no garment words
```

**Measured headline:** on a colour-swap test, vanilla CLIP returns **75.0%** identical results
for a query and its colour-swapped twin; this system returns **52.5%** — a **30% relative gain**
in compositional separation, winning on **8/12** pairs.

📄 Full design write-up: [`docs/Glance_ML_Assignment_Writeup.pdf`](docs/Glance_ML_Assignment_Writeup.pdf) · source: [`docs/WRITEUP.md`](docs/WRITEUP.md)

---

## 1. Why not just CLIP + FAISS?

CLIP is trained contrastively over a **single pooled embedding per image**. Pooling is order-
and position-agnostic: the vector records *"there is red, white, a tie, a shirt"* but **not
which attribute binds to which garment**. So `"red tie + white shirt"` and `"white tie + red
shirt"` land at nearly the same point — literally the same bag of concepts.

This is **architectural, not capacity** — a bigger encoder cannot recover information destroyed
at pooling. **The fix must add structure below the image level**, which is what per-garment
region decomposition does here.

---

## 2. The flow

### Part A — Indexing (`src/indexing/`)

```
                          ┌─ FashionCLIP ──────────────► global.faiss      (3,200 vectors)
                          │
                          ├─ BLIP caption ─► MPNet ────► caption.faiss     (3,200 vectors)
   raw image ─────────────┤
                          ├─ CLIP zero-shot tagger ────► SQLite.attributes (environment /
                          │                                                 style / garment / colour)
                          │
                          └─ SegFormer garment regions ─► crop each ─► FashionCLIP ─► region.faiss
                                                                    └─ colour + type ─► SQLite.regions
                                                                                        (9,851 regions)
```

### Part B — Retrieval (`src/retrieval/`)

```
  query text
      │
      ├─ QueryParser ──► QuerySpec { attributes, (colour,garment) bindings }
      ├─ FashionCLIP ──► q_clip ─┐
      └─ MPNet ────────► q_sent ─┤
                                 ▼
              STAGE 1 · RECALL (ANN, sub-linear)
              union of top-400 from global.faiss + caption.faiss
                                 │
                                 ▼
              STAGE 2 · RERANK (only the candidate pool)
              ┌────────────────────────────────────────────┐
              │ 0.40 · global_clip      semantics + scene   │
              │ 0.20 · caption_sim      style / vibe        │
              │ 0.20 · attribute_match  multi-attribute     │
              │ 0.20 · region_binding   colour↔garment      │
              └────────────────────────────────────────────┘
                                 │
                                 ▼
                            Top-k images
```

Each signal is **min-max normalised across the candidate pool** before weighting (CLIP cosines
sit in ~0.15–0.35 while attribute/region scores span [0,1], so a raw sum would drown CLIP).
Signals that don't apply to a query are **dropped and their weight redistributed**.

| Signal | Failure mode it fixes | Query it rescues |
|---|---|---|
| `global_clip` | overall semantics, scene | all |
| `caption_sim` | style inference with **no garment words** | 4 |
| `attribute_match` | grounded multi-attribute filtering | 1, 2 |
| `region_binding` | **compositional colour↔garment binding** | 5 |

---

## 3. Repository layout

```
configs/            every tunable knob — no constant is hardcoded in logic
  models.yaml         model checkpoints, device, dtype, batch sizes
  paths.yaml          all filesystem locations
  retrieval.yaml      scoring weights, candidate pool, binding aggregation
src/
  utils/            config + device resolution, logger, typed schema
  models/           the ONLY code importing HuggingFace: FashionCLIP · BLIP · MPNet ·
                    SegFormer · CrossEncoder
  attributes/       controlled vocabulary · zero-shot tagger · query parser
  database/         SQLite metadata store + FAISS vector store (3 indexes)
  indexing/         ▶ PART A: dataset loader (dataset-agnostic) + indexer
  retrieval/        ▶ PART B: HybridScorer (4 signals) + Retriever (2-stage)
  pipelines/        model factories (single config → object mapping)
  evaluation/       official queries · contact sheet · compositionality swap test
tests/              pure-logic unit tests (query parsing + compositional binding)
webapp/             optional Flask demo portal
scripts/            dataset builder · PDF generator
main.py             CLI: index / query / evaluate / compositional
```

**Logic is separated from data:** vectors live in FAISS, metadata in SQLite — *never in
filenames*. Swapping FashionCLIP → SigLIP is a one-line YAML change. The loader assumes only
*"a folder of images"*, so Fashionpedia / DeepFashion / COCO / phone photos all work unchanged.

---

## 4. Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Put images in `data/raw/` (any subfolder structure; symlinks are followed):

```bash
ln -s /path/to/your/images data/raw/mydata        # or:
python scripts/prepare_dataset.py --fashionpedia 700 --coco-person 300
```

GPU is auto-detected (`device: auto`); on CPU it falls back to float32. Defaults are tuned for
a 4 GB card — on 8 GB+ raise the batch sizes and switch to `blip-image-captioning-large`.

---

## 5. Usage

```bash
python main.py index                                       # Part A — build the index
python main.py query "someone in a blue shirt on a park bench" -k 10   # Part B — search
python main.py evaluate                                    # 5 official queries → eval.html
python main.py compositional                               # colour-swap test vs vanilla CLIP
python webapp/app.py                                       # visual portal @ localhost:5000
pytest -q                                                  # unit tests
```

Every result prints its **per-signal breakdown**, so any ranking decision is auditable:

```
 1. [0.942] .../0d2c93...jpg
     caption: a little girl in a yellow raincoat and red tights
     signals: global_clip=1.00 caption_sim=1.00 attribute_match=1.00 region_binding=0.77
```

### Web portal

`python webapp/app.py` → **Search** shows how your query was parsed (attributes + bindings) plus
each result's signal bars; **Evaluation** explains the scoring and runs all 5 official queries live.

---

## 6. Results & metrics

### 6.1 The five official evaluation queries

| # | Query | Score | Top result | |
|---|---|---|---|---|
| 1 | *A person in a bright yellow raincoat.* | **0.942** | "a little girl in a **yellow raincoat** and red tights" | ⭐ |
| 2 | *Professional business attire inside a modern office.* | **0.916** | "a woman in a black suit and **white shirt**" | ✅ |
| 3 | *Someone wearing a blue shirt sitting on a park bench.* | **0.906** | "a young girl sitting on a **bench in the park**" | ✅ |
| 4 | *Casual weekend outfit for a city walk.* | **0.896** | "on the **sidewalk** in a white shirt and **jeans**" | ⭐ |
| 5 | *A red tie and a white shirt in a formal setting.* | **0.702** | "a man in a vest and red pants" | ⚠️ corpus-limited |

Reproduce: `python main.py evaluate` → `outputs/results/eval.json` + `eval.html`.

### 6.2 Compositionality — colour-swap test (objective, label-free)

For a query and its colour-swapped twin, a bag-of-words model returns the **same** images
(overlap ≈ 1.0); a compositional one returns **different** ones. Both systems run on the **same
index** — only the ranking differs. 12 pairs × k=10 = **120 slots**.

| Colour-swapped pair | vanilla CLIP | this system | |
|---|---|---|---|
| green top / yellow skirt | 1.00 | **0.10** | ✅ |
| purple top / black trousers | 0.70 | **0.20** | ✅ |
| yellow top / blue skirt | 0.80 | **0.30** | ✅ |
| green jacket / white shirt | 0.80 | **0.40** | ✅ |
| brown coat / black trousers | 0.90 | **0.50** | ✅ |
| red skirt / white blouse | 0.70 | **0.60** | ✅ |
| blue shirt / red trousers | 0.90 | **0.70** | ✅ |
| black dress / white jacket | 0.90 | **0.80** | ✅ |
| black jacket / white trousers | 0.70 | 0.70 | = |
| pink blouse / black skirt | 0.70 | 0.70 | = |
| white blouse / blue jeans | 0.40 | 0.50 | ✗ |
| red tie / white shirt | 0.50 | 0.80 | ✗ |
| **MEAN overlap@10** | **0.750** | **0.525** | **−30.0%** |

**8 wins · 2 ties · 2 losses.** All 8 wins are pairs where CLIP is most blind (0.70–1.00); both
losses are pairs CLIP *already* separated (0.40, 0.50) — the mechanism helps most exactly where
the baseline fails hardest. Reproduce: `python main.py compositional`.

### 6.3 Latency

| | |
|---|---|
| Mean over the 5 official queries | **318 ms** |
| Warm query | **~150 ms** |
| Indexing throughput | 3,200 images in **~11 min** (~0.2 s/image, 4 GB RTX 2050) |

### 6.4 Index statistics

| Artefact | Size | Contents |
|---|---|---|
| `global.faiss` | 6.6 MB | 3,200 × 512-d image vectors |
| `caption.faiss` | 9.9 MB | 3,200 × 768-d caption vectors |
| `region.faiss` | 20.3 MB | 9,851 × 512-d garment-region vectors |
| `metadata.sqlite` | 17.4 MB | attributes, captions, region boxes/colours/types |

### 6.5 Corpus composition (zero-shot tagged)

**Environments** — the assignment's required axes are all present:

| indoor studio | urban street | home interior | park | office | shopping mall |
|---|---|---|---|---|---|
| 1,177 | 765 | 391 | 292 | 198 | 142 |

**Garment regions** (top types of 9,851):

| shoe | dress | bag | blouse | skirt | leggings | trousers | coat | jeans | blazer |
|---|---|---|---|---|---|---|---|---|---|
| 2,808 | 1,411 | 664 | 613 | 594 | 540 | 384 | 335 | 260 | 221 |

### 6.6 Tests

`6/6 passing` — query parsing for all 5 official queries, plus the headline compositional test:
two images identical to whole-image CLIP (red-tie/white-shirt vs the swap) are correctly
disambiguated by the region-binding signal.

---

## 7. Design decisions

| Decision | Rationale |
|---|---|
| **FashionCLIP** over generic CLIP | fine-tuned on ~800k fashion pairs — better garment/colour/fabric grounding for ~zero cost |
| **CLIP zero-shot tagging** over an LLM attribute extractor | grounds attributes in *pixels*, not in a caption an LLM might hallucinate; deterministic and reproducible |
| **SegFormer region decomposition** | the one component that genuinely fixes colour↔garment binding; dataset-agnostic |
| **BLIP-base** over BLIP-2 | captions are generated once, offline; BLIP-2's marginal gain isn't worth ~7× the memory |
| **Rule-based query parser** over an LLM | the schema is small and fixed; instant, reproducible, unit-testable — and a pluggable upgrade point |
| **FAISS + SQLite** | the brief says favour ML logic over storage engineering — so effort went into ranking |
| **Cross-encoder rerank OFF by default** | it mostly re-scores signal we already have; wired in so its uplift can be *measured*, not assumed |
| **Binding aggregation `0.5·mean + 0.5·min`** | a multi-binding query is a conjunction — half-satisfaction must not score like full, but plain `min` is brittle to one missed segmentation |

---

## 8. Scalability to 1M images

The retrieval **logic is size-invariant**; only the index type changes.

- **Recall:** `IndexFlatIP` → `IndexIVFFlat`/HNSW is a one-line change in
  `VectorStore.build(index_type="ivf")` — ids and the `search()` contract are identical.
- **Rerank:** cost is **independent of corpus size** — only the fixed 400-candidate pool is scored.
- **Memory:** 1M × 512-d fp16 ≈ 1 GB global (regions ~2.3×); IVF-PQ compresses 8–32×.
- **Metadata:** structured attributes enable **SQL pre-filtering** (`environment=office AND
  EXISTS a tie region`) to shrink the ANN space *before* search.
- **Indexing:** embarrassingly parallel — shard across workers and merge FAISS shards.

---

## 9. Known limitations

| Limitation | Cause | Fix |
|---|---|---|
| Query 5 weak | only **44 ties** in 3,200 images — Fashionpedia is womenswear/runway-heavy | **dataset composition**, not a ranker change |
| Office scenes thin | ~198 office-like images | blend in a scene-rich dataset (loader is dataset-agnostic) |
| Multi-person binding | SegFormer parses clothing, not *people* | person detector; bind within one individual |
| Uncalibrated confidences | softmax is relative, not probabilistic | per-axis Platt/temperature calibration |
| Hand-set weights | no labelled set within the time budget | learn the fusion (LambdaMART) once labels exist |

Full analysis, approaches considered, and future work (locations/weather + precision) are in the
[write-up PDF](docs/Glance_ML_Assignment_Writeup.pdf).

---

## 10. Stack

FashionCLIP (`patrickjohncyh/fashion-clip`) · BLIP (`Salesforce/blip-image-captioning-base`) ·
MPNet (`all-mpnet-base-v2`) · SegFormer (`mattmdjaga/segformer_b2_clothes`) · FAISS · SQLite ·
Flask. Hardware: single 4 GB NVIDIA RTX 2050.
