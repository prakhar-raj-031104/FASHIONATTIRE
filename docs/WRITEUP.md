# Multimodal Fashion & Context Retrieval — Design Write-up

> This is the source document for the submission PDF. It covers: (1) the space of possible
> approaches and their trade-offs, (2) the chosen architecture and *why* it handles fashion
> queries, (3) future work — locations/weather and precision, and (4) scalability &
> zero-shot behaviour.

---

## 1. Problem framing

The task is not "retrieve images" — it is **compositional, attribute-grounded retrieval**
over three axes the query can mix freely:

- **Garment** (formal / casual / outerwear, plus fine type: tie, raincoat, blazer…)
- **Environment** (office, street, park, home…)
- **Color / vibe** (a wide palette; "professional", "casual weekend")

The evaluation queries are deliberately chosen to break a naïve system:

| # | Query | Stresses |
|---|---|---|
| 1 | *bright yellow raincoat* | color + garment |
| 2 | *business attire inside a modern office* | style + scene |
| 3 | *blue shirt sitting on a park bench* | color + garment + scene |
| 4 | *casual weekend outfit for a city walk* | **style inference** — no garment words |
| 5 | *red tie **and** white shirt, formal setting* | **compositional binding** |

Two of these (4, 5) are exactly where a vanilla CLIP index fails.

---

## 2. Approaches considered (and their trade-offs)

### A. Vanilla CLIP / OpenCLIP + FAISS (the baseline everyone builds)
Embed each image with CLIP, embed the query with CLIP's text tower, cosine + top-k.

- ✅ Trivial, strong zero-shot, one model.
- ❌ **Bag-of-words compositionality**: CLIP pools the whole image into one vector, so
  *"red tie + white shirt"* and *"white tie + red shirt"* map to nearly the same point.
- ❌ Weak on fine-grained fashion attributes (generic training distribution).
- ❌ No scene/style *inference* beyond what one vector encodes.
- **Verdict:** necessary baseline, insufficient alone — the assignment says so explicitly.

### B. Fashion-domain encoder (FashionCLIP) + FASS
Swap generic CLIP for **FashionCLIP** (CLIP fine-tuned on ~800k fashion image-text pairs).

- ✅ Big, cheap quality win on garments/fabrics/colors/styles.
- ❌ Same architectural bag-of-words limitation — binding still unsolved.
- **Verdict:** adopt the encoder, but it is the *floor*, not the solution.

### C. Caption → structured attributes → hybrid scoring
Caption each image (BLIP), extract structured attributes, add an attribute-match term.

- ✅ Interpretable; handles multi-attribute and scene/style well; good for query 4.
- ⚠️ If attributes are extracted from the *caption* by an LLM, errors compound
  (caption hallucination → wrong attribute) and it is non-deterministic.
- ⚠️ Captioner choice matters: BLIP-2 (~15GB) vs BLIP-large (~2GB).
- **Verdict:** adopt the *idea*, but ground attributes in pixels (see D) rather than in a
  hallucination-prone caption, and use BLIP-large.

### D. CLIP zero-shot tagging for structured attributes (no LLM)
Use FashionCLIP itself as a zero-shot classifier against controlled vocabularies
("a photo of a person in {office/park/…}", "a {red/blue/…} garment"), with prompt
ensembling.

- ✅ Grounded in the image, deterministic, reproducible, no extra GPU-hungry LLM.
- ✅ Vocabulary maps 1:1 onto the evaluation axes; extending to new cities/weather is a
  data edit, not a model change.
- ❌ Still image-level, so color↔garment binding remains unsolved by this term alone.
- **Verdict:** adopt — this is our structured-attribute source.

### E. Region / part decomposition (the compositionality fix)
Segment garments (SegFormer clothes-parser), crop each, embed + color-tag each region
independently; at query time decompose into `(color, garment)` bindings and require each
to be satisfied by *some* region.

- ✅ Genuinely defeats the bag-of-words failure — binding is enforced **spatially**.
- ✅ Dataset-agnostic (no reliance on Fashionpedia masks; works on any image).
- ⚠️ Extra model + per-image segmentation cost at index time (offline, one-off).
- **Verdict:** adopt — this is the headline ML contribution and the answer to query 5.

### F. Heavy over-engineering (rejected)
VQA-per-attribute, scene-graph parsers, an object detector *and* a segmenter, a local LLM
for query parsing, Pinecone/Kafka/K8s.

- ❌ More failure surface, more latency, no retrieval-quality gain at this scale.
- **Verdict:** rejected. Every component must earn its place by improving ranking.

---

## 3. Chosen architecture

A **domain encoder + structure** design: FashionCLIP as the shared space, enriched by
three orthogonal signals, fused by a normalized weighted sum.

### Indexing (Part A) — per image
1. **FashionCLIP** whole-image embedding → `global.faiss`.
2. **BLIP-large** caption → **MPNet** sentence embedding → `caption.faiss`.
3. **FashionCLIP zero-shot tagger** → structured attributes (environment/style/garment/
   color, each a value→confidence map) → SQLite.
4. **SegFormer** garment regions → crop each → FashionCLIP embedding + CLIP color/type
   tag per region → `region.faiss` + SQLite.

### Retrieval (Part B) — per query
1. **Parse** the query (rule + synonym + adjacency) into a `QuerySpec`:
   `attributes` (axis→values) and `bindings` (`(color, garment)` pairs).
2. **Recall (ANN):** union of top-N from `global.faiss` (query CLIP text) and
   `caption.faiss` (query sentence) → candidate pool.
3. **Precision (rerank):** score every candidate with four signals, min-max normalize
   each across the pool, weight-fuse (inactive signals dropped, weights renormalized):

```
final = w1·global_clip  +  w2·caption_sim  +  w3·attribute_match  +  w4·region_binding
        (defaults: 0.40        0.20              0.20                  0.20)
```

4. *(Optional)* cross-encoder rerank over captions as a tie-breaker (off by default).

### Why each signal exists (mapped to the eval)
- **global_clip** — semantic + scene floor; carries every query.
- **caption_sim** — answers query 4: the caption "a man in a hoodie walking down a street"
  matches "casual weekend outfit for a city walk" even with **zero** shared garment words.
  This is where *style inference* lives.
- **attribute_match** — answers queries 1 & 2: explicit, grounded color/garment/scene/style
  confidences the query can filter on ("yellow"+"raincoat", "business"+"office").
- **region_binding** — answers query 5: "red" must land on the *tie* region **and** "white"
  on the *shirt* region; the color-swapped image scores low. This is verified by a unit
  test (`tests/test_scorer.py`) on two images that whole-image CLIP cannot separate.

### Why normalized fusion (an easy thing to get wrong)
FashionCLIP cosines live in a narrow band (~0.15–0.35) while attribute/region scores span
[0,1]. A raw weighted sum would silently drown the CLIP signal (or vice-versa). We min-max
normalize **each signal across the candidate pool** before weighting, so every term
competes on equal footing. Weights are tunable on a small labelled validation set (the
evaluation harness computes precision@k when a labels file is supplied).

---

## 4. Future work

### 4a. Adding **locations** (cities, places) and **weather**
The design already treats "environment" as a structured axis, so extending it is mostly a
*data* change plus one new signal — not an architecture rewrite:

1. **Vocabulary extension (zero-code-path):** add `weather = [sunny, rainy, snowy,
   overcast, …]` and richer `place` values to `src/attributes/vocab.py` + prompt
   templates. FashionCLIP zero-shot-tags them immediately; the query parser gains synonym
   entries. This alone lets "someone in a raincoat on a rainy street" prefer wet scenes.
2. **Dedicated place/weather models** for precision: a scene classifier (e.g. Places365)
   and a sky/weather classifier give calibrated tags instead of CLIP zero-shot; stored as
   two more attribute axes and two more scoring terms.
3. **Geo/EXIF + reverse-geocoding:** if images carry GPS EXIF, resolve to city/landmark
   and store as metadata — enabling literal "in Paris" filters via SQL pre-filtering,
   combined with the visual signal (a strong hybrid: structured geo filter → visual rerank).
4. **Cross-modal weather consistency:** a light regularizer that boosts images whose
   *garment* and *weather* agree (raincoat ⟷ rain, coat ⟷ snow), catching context the raw
   query implies but doesn't state.

Because environment is already a first-class axis with its own scoring term, each of these
plugs into the existing fusion with a new weight — no change to the retrieval flow.

### 4b. Improving **precision**
- **Fine-tune / adapt FashionCLIP** on the target catalog (LoRA on image-text pairs, or
  contrastive fine-tuning on hard negatives — especially color-swapped pairs to sharpen
  binding at the *encoder* level, not just the scorer).
- **Learned fusion instead of hand weights:** collect a few hundred (query, relevant)
  labels and fit a small logistic/LambdaMART reranker over the four signals — turns
  hand-tuned weights into a learned ranking model.
- **Hard-negative mining** for the region-binding term (mine same-garments/different-color
  pairs) to calibrate color confidences.
- **Better color grounding:** complement CLIP color tags with a classical dominant-color
  estimate in CIELAB per region (robust to lighting), ensembled with the CLIP color head.
- **Query understanding upgrade:** swap the rule parser for a small instruction-tuned LLM
  that emits the same `QuerySpec` JSON (handles negation "no tie", quantities, comparatives)
  — the schema and downstream code are unchanged (it is already a pluggable component).
- **Calibrated attribute thresholds** per axis (Platt scaling) so "soft" attribute scores
  are comparable across axes.
- **Enable + tune the cross-encoder** and measure precision@k uplift honestly via the eval
  harness before committing it.

---

## 5. Scalability to 1M+ images

- **Recall stage:** swap `IndexFlatIP` → `IndexIVFFlat`/HNSW — a **one-line** change in
  `VectorStore.build` (`index_type="ivf"`); ids and the `search()` contract are unchanged,
  so nothing else moves. Sublinear ANN keeps latency ~ms.
- **Rerank stage:** only the small candidate pool (default 200) is scored with the
  expensive signals — cost is independent of corpus size.
- **Metadata:** SQLite is fine to low-millions; the clean seam (`MetadataDB`) allows a swap
  to Postgres/DuckDB. **SQL attribute pre-filtering** ("environment=office AND has a tie
  region") can shrink the ANN search space before it even runs.
- **Indexing** is embarrassingly parallel (shard images across GPUs/workers, merge FAISS
  shards). Embeddings are computed once, offline.
- **Memory:** 1M × 512-d fp16 ≈ 1 GB for the global index; region index scales with avg
  regions/image (~2–4×). Product Quantization (IVFPQ) compresses this 8–32× if needed.

## 6. Zero-shot capability

Nothing in the system is trained on the corpus or on the query labels:
- FashionCLIP, BLIP, MPNet, SegFormer are all used **off-the-shelf**.
- Attributes come from **zero-shot** CLIP tagging against an editable vocabulary.
- New concepts (a new color, garment, city, weather) are added by editing the vocabulary
  — **no retraining**. This directly satisfies the "handles descriptions not seen as an
  explicit training label" criterion.

---

## Appendix — honest limitations
- SegFormer parses one clothed person well; crowded multi-person scenes get coarser
  regions (mitigation: add a person detector and bind per-person — future work).
- Rule-based parsing covers the evaluation grammar and common synonyms but not arbitrary
  negation/comparatives (mitigation: the pluggable LLM parser in §4b).
- Attribute confidences from CLIP zero-shot are *relative*, not calibrated probabilities
  (mitigation: per-axis calibration in §4b). Fusion normalization makes this robust in
  practice, and every result ships with its per-signal breakdown for transparency.
