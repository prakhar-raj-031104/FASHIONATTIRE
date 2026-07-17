"""Indexing pipeline orchestration (Part A of the assignment).

For each image, in memory-friendly chunks:
  1. FashionCLIP whole-image embedding           -> global.faiss
  2. BLIP caption -> sentence embedding           -> caption.faiss
  3. CLIP zero-shot structured attributes         -> SQLite (images.attributes)
  4. SegFormer garment regions -> per-region crop -> FashionCLIP embedding + CLIP
     color/type tagging                            -> region.faiss + SQLite (regions)

Region crops are batched across the whole chunk before hitting FashionCLIP, so GPU
utilisation stays high even though segmentation is per-image.
"""
from __future__ import annotations

from typing import List

import numpy as np

from ..attributes.vocab import REGION_TYPE_CANDIDATES
from ..database import FaissIndex, MetadataDB, VectorStore
from ..pipelines import (
    build_captioner,
    build_clip,
    build_segmenter,
    build_sentence_encoder,
    build_tagger,
)
from ..utils.config import Config
from ..utils.logger import get_logger
from ..utils.schema import ImageRecord, RegionRecord
from .dataset_loader import discover_images, load_image


class Indexer:
    def __init__(self, cfg: Config, chunk_size: int = 16):
        self.cfg = cfg
        self.chunk_size = chunk_size
        self.log = get_logger("indexer", cfg.path("outputs", "logs"))

        self.log.info("Loading models on device=%s ...", cfg.device)
        self.clip = build_clip(cfg)
        self.captioner = build_captioner(cfg)
        self.segmenter = build_segmenter(cfg)
        self.sentence = build_sentence_encoder(cfg)
        self.tagger = build_tagger(self.clip)
        self.log.info(
            "Models ready (captioner=%s, segmenter=%s).",
            self.captioner is not None, self.segmenter is not None,
        )

    # ------------------------------------------------------------------ #
    def _process_regions(self, image_id: int, pil, seg_regions,
                         region_id_start: int) -> List[RegionRecord]:
        """Crop, embed, and tag every garment region of one image."""
        records: List[RegionRecord] = []
        if not seg_regions:
            return records
        crops = [pil.crop(tuple(r.bbox)) for r in seg_regions]
        crop_embs = self.clip.encode_image(crops)  # (R, D), L2-normed

        for i, (seg, emb) in enumerate(zip(seg_regions, crop_embs)):
            type_scores = self.tagger.type_region(seg.seg_label, emb) \
                if seg.seg_label in REGION_TYPE_CANDIDATES else {}
            colors = self.tagger.color_region(emb)
            gtype = max(type_scores, key=type_scores.get) if type_scores else seg.default_type
            records.append(RegionRecord(
                region_id=region_id_start + i,
                image_id=image_id,
                garment_label=seg.seg_label,
                garment_type=gtype,
                bbox=seg.bbox,
                area_frac=seg.area_frac,
                colors=colors,
                type_scores=type_scores,
                embedding=emb.astype(np.float32),
            ))
        return records

    # ------------------------------------------------------------------ #
    def run(self, image_dir=None) -> None:
        cfg = self.cfg
        cfg.ensure_dirs()
        image_dir = image_dir or cfg.path("data", "raw_images")
        items = discover_images(image_dir)
        if not items:
            raise RuntimeError(f"No images found under {image_dir}")
        self.log.info("Discovered %d images under %s", len(items), image_dir)

        db = MetadataDB(cfg.path("index", "metadata_db"))
        global_index = VectorStore.build(self.clip.embed_dim)
        caption_index = VectorStore.build(self.sentence.embed_dim) if self.captioner else None
        region_index = VectorStore.build(self.clip.embed_dim) if self.segmenter else None

        region_counter = 0
        try:
            from tqdm import tqdm
            chunks = range(0, len(items), self.chunk_size)
            for start in tqdm(chunks, desc="Indexing", unit="chunk"):
                batch = items[start:start + self.chunk_size]
                ids = [iid for iid, _ in batch]
                pils = [load_image(p) for _, p in batch]

                clip_embs = self.clip.encode_image(pils)
                captions = self.captioner.caption(pils) if self.captioner else [""] * len(pils)
                cap_embs = (self.sentence.encode(captions)
                            if self.captioner else np.zeros((len(pils), 0)))

                # image-level records + FAISS payloads
                region_ids_all: List[int] = []
                region_vecs_all: List[np.ndarray] = []
                for j, (iid, path) in enumerate(batch):
                    rec = ImageRecord(
                        image_id=iid, image_path=path, caption=captions[j],
                        clip_embedding=clip_embs[j],
                        caption_embedding=cap_embs[j] if self.captioner else None,
                        attributes=self.tagger.tag_image(clip_embs[j]),
                    )
                    if self.segmenter:
                        seg = self.segmenter.segment(pils[j])
                        rec.regions = self._process_regions(iid, pils[j], seg, region_counter)
                        region_counter += len(rec.regions)
                        for r in rec.regions:
                            region_ids_all.append(r.region_id)
                            region_vecs_all.append(r.embedding)
                    db.insert_image(rec)

                # add vectors to FAISS
                global_index.add(np.array(ids, dtype=np.int64), clip_embs)
                if caption_index is not None:
                    caption_index.add(np.array(ids, dtype=np.int64), cap_embs)
                if region_index is not None and region_vecs_all:
                    region_index.add(np.array(region_ids_all, dtype=np.int64),
                                     np.stack(region_vecs_all, axis=0))
                db.commit()

            self.log.info("Indexed %d images, %d regions.", db.count_images(), region_counter)

            # persist
            global_index.save(cfg.path("index", "faiss_global"))
            if caption_index is not None:
                caption_index.save(cfg.path("index", "faiss_caption"))
            if region_index is not None:
                region_index.save(cfg.path("index", "faiss_region"))
            self.log.info("Saved FAISS indexes + metadata DB under %s",
                          cfg.path("index", "dir"))
        finally:
            db.close()
