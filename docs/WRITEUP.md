# Multimodal Fashion & Context Retrieval

**Glance ML Internship Assignment** · **Codebase:** <GITHUB_REPO_URL>

A natural-language image search engine over **3,200 fashion images** that reasons jointly about **what** a person wears, **where** they are, and the outfit's **vibe**.

**Core thesis:** a single global CLIP embedding is *structurally* incapable of compositional retrieval, so the fix must add **structure**, not a bigger encoder.

| Corpus | Latency | Key measured result | Training |
|---|---|---|---|
| 3,200 images · 9,851 garment regions · FAISS ×3 + SQLite | ~150 ms warm | Colour-swap overlap@10: vanilla CLIP **0.750** → this system **0.525** (**−30%**, wins 8/12 pairs) | **None** — fully zero-shot |

---

## 1. Why this is hard

| # | Query | Capability | CLIP's problem |
|---|---|---|---|
| 1 | bright yellow raincoat | colour + garment | sparse fine-grained fashion vocabulary |
| 2 | business attire in a modern office | style + scene | needs scene grounding |
| 3 | blue shirt on a park bench | colour + garment + scene | multi-constraint conjunction |
| 4 | casual weekend outfit for a city walk | **style inference** | **no garment words at all** |
| 5 | a red tie **and** a white shirt | **compositional binding** | **canonical CLIP failure** |

**The actual failure mechanism.** CLIP is trained contrastively over a **single pooled embedding per image**. Pooling is order- and position-agnostic: the vector encodes *"there is red, white, a tie, a shirt"* but **not which attribute binds to which object**. So `"red tie + white shirt"` and `"white tie + red shirt"` map to nearly the same point — literally the same bag of concepts. This is **architectural, not capacity**: scaling or fine-tuning the encoder cannot recover information destroyed at pooling. **Therefore the fix must reintroduce structure below the image level.**

---

## 2. Approaches and trade-offs *(Deliverable 1)*

| Approach | Strengths | Weaknesses | Best when |
|---|---|---|---|
| **A. Vanilla CLIP + FAISS** | trivial, strong zero-shot | **bag-of-words**: no binding; weak fashion terms | quick baseline, single-concept queries |
| **B. FashionCLIP + FAISS** | large domain gain on garment/colour/fabric for ~zero cost | same architectural binding limit | any fashion task — the *floor*, not the solution |
| **C. Caption → LLM attributes** | interpretable; good scene/style | errors compound (hallucinated caption → wrong attribute); non-deterministic; extra GPU | open-vocabulary attributes with reliable captions |
| **D. CLIP zero-shot tagging** | grounded in *pixels*; deterministic; cheap; editable vocabulary | image-level only → still no binding | structured, filterable attributes without training |
| **E. Region decomposition** | genuinely **solves binding**; dataset-agnostic | segmentation cost; fails on tiny/occluded items | whenever compositionality is graded |
| **F. Detector + scene graph + VQA, or Pinecone/microservices** | maximal expressiveness | large complexity, more failure surface, **no measured gain at this scale** | large production systems |

**Decision: B + D + E**, plus a caption signal from C (BLIP only, **no LLM extraction** — grounding attributes in pixels beats grounding them in possibly-hallucinated text). **F rejected** as over-engineering: the brief says favour ML logic over indexing engineering, so I used FAISS + SQLite and spent the effort on ranking.

---

## 3. Chosen architecture *(Deliverable 2)*

**Part A — Indexer (`src/indexing/`).** Per image: (1) **FashionCLIP** global embedding → `global.faiss`; (2) **BLIP caption → MPNet** sentence embedding → `caption.faiss`; (3) **CLIP zero-shot attributes** (environment/style/garment/colour, each value embedded through several **prompt templates and averaged**) → SQLite; (4) **SegFormer garment regions**, each cropped, embedded, and tagged with **its own colour and type** → `region.faiss` + SQLite. Artefact (4) is the crux: it preserves *which colour sits on which garment*.

**Part B — Retriever (`src/retrieval/`).** *Stage 1 (recall):* query embedded by FashionCLIP + MPNet; union of top-N from the global and caption indexes forms a 400-candidate pool. *Stage 2 (rerank):* only that pool is scored with expensive signals. This ANN→rerank split is why the logic is unchanged at 1M images (§6).

```
score = 0.40·global_clip + 0.20·caption_sim + 0.20·attribute_match + 0.20·region_binding
```

| Signal | Fixes | Rescues |
|---|---|---|
| `global_clip` | overall semantics + scene | all |
| `caption_sim` | **style inference** — caption *"a man in a hoodie walking down a street"* matches *"casual weekend outfit for a city walk"* with **zero shared garment words** | **4** |
| `attribute_match` | grounded multi-attribute filtering | **1, 2** |
| `region_binding` | **compositional colour↔garment binding** | **5** |

**Query parsing.** A deterministic rule + synonym + adjacency parser yields a `QuerySpec`: structured attributes plus `(colour, garment)` **bindings** (`"a red tie and a white shirt"` → `[(red,tie), (white,shirt)]`). Rules over an LLM because the schema is small and fixed, it is instant, reproducible and unit-testable — while remaining a drop-in point for an LLM parser (§7.2).

**Binding score.** A binding scores high only if **one single region** matches *both* garment type *and* colour. Multiple bindings combine as `0.5·mean + 0.5·min`: the query is a **conjunction**, so half-satisfaction must not score like full (plain `mean` is too lenient), but plain `min` is brittle — one missed segmentation would zero a perfect image.

**Two subtleties, both found by measurement:**
1. **Per-signal normalisation is mandatory.** CLIP cosines occupy ~0.15–0.35 while attribute/region scores span [0,1]; a raw weighted sum would drown the CLIP signal. Each signal is min-max normalised **across the candidate pool** before weighting; signals that don't apply are dropped and their weight **redistributed**.
2. **Bound attributes must be excluded from `attribute_match`.** That signal is *order-agnostic*: both swap variants reduce to `{red, white}`, so counting them again pulls swapped queries back **together**, diluting the order-sensitive binding. Excluding them improved measured separation **15.8% → 23.7%**.

---

## 4. Measured results

### 4.1 Compositionality — objective and label-free

For a query **Q** and its **colour-swapped twin Q′**, a bag-of-words retriever cannot tell them apart → `top-k` sets nearly **identical**; a compositional one returns **different** images. So **overlap@k is an inverse proxy for compositional sensitivity, needing no human labels.** Both systems run on the **same index**; only ranking differs. **12 pairs × k=10 = 120 slots**, sized so one image (0.008) cannot dominate the mean.

| Pair | CLIP | Ours | | Pair | CLIP | Ours |
|---|---|---|---|---|---|---|
| green top / yellow skirt | 1.00 | **0.10** ✅ | | blue shirt / red trousers | 0.90 | **0.70** ✅ |
| purple top / black trousers | 0.70 | **0.20** ✅ | | black dress / white jacket | 0.90 | **0.80** ✅ |
| yellow top / blue skirt | 0.80 | **0.30** ✅ | | black jacket / white trousers | 0.70 | 0.70 = |
| green jacket / white shirt | 0.80 | **0.40** ✅ | | pink blouse / black skirt | 0.70 | 0.70 = |
| brown coat / black trousers | 0.90 | **0.50** ✅ | | white blouse / blue jeans | 0.40 | 0.50 ✗ |
| red skirt / white blouse | 0.70 | **0.60** ✅ | | red tie / white shirt | 0.50 | 0.80 ✗ |
| **MEAN** | **0.750** | **0.525** | | **relative reduction** | | **−30.0%** |

**8/12 wins, 2 ties, 2 losses.** Clearest case: *green top / yellow skirt*, where CLIP returns an **identical** top-10 for both phrasings (1.00) — the textbook failure — while this system separates them almost completely (0.10).

**Where gains concentrate.** All 8 wins are pairs where CLIP is most blind (0.70–1.00); both losses are pairs CLIP *already* separated (0.40, 0.50). **The mechanism contributes most precisely where the baseline fails hardest** — the desired behaviour for a corrective signal, and it suggests making the binding weight *adaptive* to how confidently the dense signals already discriminate.

### 4.2 Honest negative result, diagnosed

*Red tie / white shirt* — query 5 — **did not improve**. Traced in data: the index held only **18 tie regions / 3,200 images**, because two filters removed small garments (0.5% area threshold; a 6-region cap applied to an *area-sorted* list, so a tie — the smallest item — was truncated). I lowered the threshold to 0.15% and raised the cap: regions **7,321 → 9,851**, ties **18 → 44** — and query 5 *still* didn't improve.

**The remaining ceiling is the corpus, not the algorithm.** Fashionpedia is womenswear/runway-heavy: 44 ties (~1.4%), few red. The mechanism demonstrably works where garments exist (1.00 → 0.10 above). The fix is **dataset composition** (blend in menswear/formal imagery), not a ranker change. *Two plausible assumptions — "regions exist" and "more regions will fix it" — were both false; only measurement revealed it.*

### 4.3 Official queries · corpus

| # | Score | Top result | |
|---|---|---|---|
| 1 | 0.942 | *"a little girl in a **yellow raincoat** and red tights"* | ⭐ exact |
| 2 | 0.916 | *"a woman in a black suit and **white shirt**"* | ✅ attire right; office *scene* is the weak axis |
| 3 | 0.906 | *"a young girl sitting on a **bench in the park**"* | ✅ scene + pose |
| 4 | 0.896 | *"on the **sidewalk** in a white shirt and **jeans**"* | ⭐ inferred from **no garment words** |
| 5 | 0.702 | *"a man in a vest and red pants"* | ⚠️ corpus-limited (§4.2) |

Latency **318 ms** mean (~150 ms warm). Zero-shot environment tagging confirms the required axes: urban street **765**, home interior **391**, park **292**, office **198**; remainder catalogue/runway "studio" (1,177) — the Fashionpedia bias, and why query 2's scene component is weakest.

---

## 5. Shortcomings and fixes

| Shortcoming | Cause | Fix |
|---|---|---|
| Small garments under-detected | area threshold + region cap; only 44 ties/3,200 | lower threshold (done); accessory-specialised detector; weight by *saliency* not area |
| Multi-person images bind wrongly | SegFormer parses clothing, not *people* — a red tie on A and white shirt on B satisfies both | person detector; require bindings to resolve **within one individual** |
| Uncalibrated attribute confidences | softmax gives *relative*, not probabilistic, scores; scales differ per axis | per-axis Platt/temperature calibration |
| Scene axis corpus-limited | Fashionpedia is catalogue-heavy (~6% office) | blend a scene-rich people dataset — loader is dataset-agnostic by design |
| Parser misses negation/comparatives | "no tie", "more formal than" out of grammar | small instruction-tuned LLM emitting the **same** `QuerySpec` JSON — interface exists |
| Hand-set weights | no labelled set in the time budget | learn the fusion (§7.2) |

---

## 6. Modular code · Scalability · Zero-shot *(Criteria 2–4)*

**Modular — logic separated from data.** `configs/*.yaml` holds every knob (no constant in code); `src/models/` wrappers are the **only** code importing HuggingFace, so FashionCLIP → SigLIP is a one-line YAML change; **vectors in FAISS, metadata in SQLite — never in filenames**; typed dataclasses (`ImageRecord`/`RegionRecord`/`QuerySpec`) pass between independently testable stages; the loader assumes only *"a folder of images"*, so Fashionpedia/DeepFashion/COCO/phone photos all work unchanged. *Filename-independence is structural: the dataset's filenames are MD5 hashes with zero semantic content, and `image_path` never enters any scoring computation.*

**Scalability to 1M+.** The retrieval **logic is size-invariant**; only the index type changes. *Recall:* `IndexFlatIP` → `IndexIVFFlat`/HNSW is a **one-line** change in `VectorStore.build(index_type="ivf")` — ids and the `search()` contract are identical, so nothing else moves; O(N) → ~O(√N)/O(log N). *Rerank:* cost is **independent of corpus size** — only the fixed 400-candidate pool is scored — the key scalability property. *Memory:* 1M × 512-d fp16 ≈ **1 GB** global (regions ~2.3×); IVF-PQ compresses 8–32×. *Metadata:* SQLite is fine into low millions, and structured attributes enable **SQL pre-filtering** (`environment=office AND EXISTS a tie region`) to shrink the ANN space *before* search. *Indexing:* embarrassingly parallel — shard and merge FAISS shards. Measured **3,200 images in ~11 min** on one 4 GB laptop GPU (~0.2 s/image).

**Zero-shot.** **Nothing is trained on this corpus or these queries** — FashionCLIP, BLIP, MPNet, SegFormer are all off-the-shelf, and attributes come from **zero-shot CLIP classification** against an editable vocabulary. A new colour, garment, city or weather condition is added by editing `vocab.py` with **no retraining**. Query 4 is the demonstration: *"casual weekend outfit for a city walk"* has **no garment label**, yet returns sidewalk/jeans imagery because caption and attribute signals infer the vibe rather than matching a keyword.

---

## 7. Future work *(Deliverable 4)*

### 7.1 Adding locations (cities, places) and weather

Environment is already a **first-class structured axis**, so this is mostly *data* plus new scoring terms — no architectural rewrite.

1. **Vocabulary extension (no code-path change):** add `weather = [sunny, rainy, snowy, overcast…]` and richer `place` values to `vocab.py` with prompt templates. FashionCLIP tags them **zero-shot** immediately; the parser gains synonyms. This alone makes *"a raincoat on a rainy street"* prefer genuinely wet scenes.
2. **Specialised classifiers for precision:** **Places365** for scene/venue and a sky/weather classifier, stored as two more attribute axes and two more weighted terms.
3. **Geo metadata + reverse geocoding:** resolve GPS EXIF to city/landmark. Enables a powerful hybrid — a *hard SQL geo filter* ("in Paris") composed with *soft visual ranking*, exactly the pre-filter pattern from §6.
4. **Landmark recognition** ("in front of the Eiffel Tower") via embedding lookup against a landmark gallery — another region-like signal.
5. **Garment↔weather consistency prior:** reward agreement (raincoat ⟷ rain, parka ⟷ snow), capturing context a query *implies* but doesn't state — enabling *"dressed for bad weather"*.

Each is "another axis + another weighted signal", slotting into the existing fusion with no change to the retrieval flow.

### 7.2 Improving precision

1. **Build a labelled validation set** — highest value. Without labels, tuning is guesswork; with a few hundred `(query, relevant_id)` pairs we get **precision@k / nDCG** and every change becomes measurable. The harness already accepts `--labels`.
2. **Learn the fusion:** logistic regression or **LambdaMART** over the four signals replaces hand-tuned constants and naturally learns *query-dependent* weighting.
3. **Contrastive fine-tuning on hard negatives:** LoRA-adapt FashionCLIP using **colour-swapped pairs** as explicit negatives, pushing binding into the *encoder* — attacking the root cause in §1.
4. **Stronger colour grounding:** ensemble the CLIP colour head with a classical **CIELAB** dominant-colour estimate per region (robust to lighting) — should lift queries 1, 3, 5.
5. **Per-person binding** (§5) — the biggest correctness fix for crowded scenes.
6. **Cross-encoder reranking** — implemented behind a flag, to be enabled only once its uplift is *measured*.
7. **Query expansion** for style inference: expand *"weekend casual"* into likely garments via a learned style→garment prior.

---

## Appendix — Reproducing

```bash
pip install -r requirements.txt
python main.py index                                    # Part A — build FAISS + SQLite
python main.py query "a red tie and a white shirt" -k 10  # Part B — top-k retrieval
python main.py evaluate                                 # 5 official queries → contact sheet
python main.py compositional                            # the colour-swap test of §4.1
python webapp/app.py                                    # optional visual portal
pytest -q                                               # unit tests
```

**Stack:** FashionCLIP (`patrickjohncyh/fashion-clip`) · BLIP (`blip-image-captioning-base`) · MPNet (`all-mpnet-base-v2`) · SegFormer (`mattmdjaga/segformer_b2_clothes`) · FAISS · SQLite. Hardware: one 4 GB NVIDIA RTX 2050.
