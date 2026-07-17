"""SQLite metadata store — the structured, queryable side of the index.

Separation of concerns (an explicit evaluation criterion): vectors live in FAISS,
*metadata* lives here. FAISS answers "which vectors are nearest"; SQLite answers "what is
image 42, what are its attributes, and which regions belong to it". Keeping them apart is
also what makes the 1M-scale story clean — you pre-filter candidates with SQL attributes
before/after the ANN search.

SQLite is chosen over Postgres/etc. on purpose: zero-ops, single-file, perfectly
sufficient for <=~1M rows here. The access is wrapped so it could be swapped later.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from ..utils.schema import ImageRecord, RegionRecord


class MetadataDB:
    def __init__(self, db_path: Path | str):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False so the connection can be shared across threads (the web
        # demo serves requests on worker threads). Reads are the only concurrent access;
        # SQLite serializes them internally. The indexer still uses it single-threaded.
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self.conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS images (
                image_id      INTEGER PRIMARY KEY,
                image_path    TEXT NOT NULL,
                caption       TEXT,
                attributes    TEXT   -- JSON: {axis: {value: conf}}
            );
            CREATE TABLE IF NOT EXISTS regions (
                region_id     INTEGER PRIMARY KEY,
                image_id      INTEGER NOT NULL,
                garment_label TEXT,
                garment_type  TEXT,
                bbox          TEXT,   -- JSON [x0,y0,x1,y1]
                area_frac     REAL,
                colors        TEXT,   -- JSON {color: conf}
                type_scores   TEXT,   -- JSON {type: conf}
                FOREIGN KEY(image_id) REFERENCES images(image_id)
            );
            CREATE INDEX IF NOT EXISTS idx_regions_image ON regions(image_id);
            CREATE INDEX IF NOT EXISTS idx_regions_type  ON regions(garment_type);
            """
        )
        self.conn.commit()

    # --- writes --------------------------------------------------------- #
    def insert_image(self, rec: ImageRecord) -> None:
        cur = self.conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO images(image_id, image_path, caption, attributes) "
            "VALUES (?, ?, ?, ?)",
            (rec.image_id, rec.image_path, rec.caption, json.dumps(rec.attributes)),
        )
        for r in rec.regions:
            cur.execute(
                "INSERT OR REPLACE INTO regions(region_id, image_id, garment_label, "
                "garment_type, bbox, area_frac, colors, type_scores) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (r.region_id, r.image_id, r.garment_label, r.garment_type,
                 json.dumps(r.bbox), r.area_frac, json.dumps(r.colors),
                 json.dumps(r.type_scores)),
            )

    def commit(self) -> None:
        self.conn.commit()

    # --- reads ---------------------------------------------------------- #
    def count_images(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]

    def _row_to_region(self, row: sqlite3.Row) -> RegionRecord:
        return RegionRecord(
            region_id=row["region_id"],
            image_id=row["image_id"],
            garment_label=row["garment_label"],
            garment_type=row["garment_type"],
            bbox=json.loads(row["bbox"]) if row["bbox"] else [],
            area_frac=row["area_frac"],
            colors=json.loads(row["colors"]) if row["colors"] else {},
            type_scores=json.loads(row["type_scores"]) if row["type_scores"] else {},
        )

    def get_image(self, image_id: int, with_regions: bool = True) -> Optional[ImageRecord]:
        row = self.conn.execute(
            "SELECT * FROM images WHERE image_id=?", (image_id,)
        ).fetchone()
        if row is None:
            return None
        rec = ImageRecord(
            image_id=row["image_id"],
            image_path=row["image_path"],
            caption=row["caption"] or "",
            attributes=json.loads(row["attributes"]) if row["attributes"] else {},
        )
        if with_regions:
            rec.regions = self.get_regions(image_id)
        return rec

    def get_images(self, image_ids: List[int]) -> Dict[int, ImageRecord]:
        """Batch fetch (one query for images, one for regions) — avoids N+1."""
        if not image_ids:
            return {}
        qs = ",".join("?" * len(image_ids))
        recs: Dict[int, ImageRecord] = {}
        for row in self.conn.execute(
            f"SELECT * FROM images WHERE image_id IN ({qs})", image_ids
        ):
            recs[row["image_id"]] = ImageRecord(
                image_id=row["image_id"], image_path=row["image_path"],
                caption=row["caption"] or "",
                attributes=json.loads(row["attributes"]) if row["attributes"] else {},
            )
        for row in self.conn.execute(
            f"SELECT * FROM regions WHERE image_id IN ({qs})", image_ids
        ):
            r = self._row_to_region(row)
            if r.image_id in recs:
                recs[r.image_id].regions.append(r)
        return recs

    def get_regions(self, image_id: int) -> List[RegionRecord]:
        rows = self.conn.execute(
            "SELECT * FROM regions WHERE image_id=?", (image_id,)
        ).fetchall()
        return [self._row_to_region(r) for r in rows]

    def region_to_image(self, region_ids: List[int]) -> Dict[int, int]:
        if not region_ids:
            return {}
        qs = ",".join("?" * len(region_ids))
        return {
            row["region_id"]: row["image_id"]
            for row in self.conn.execute(
                f"SELECT region_id, image_id FROM regions WHERE region_id IN ({qs})",
                region_ids,
            )
        }

    def iter_images(self) -> Iterator[ImageRecord]:
        for row in self.conn.execute("SELECT image_id FROM images ORDER BY image_id"):
            rec = self.get_image(row["image_id"])
            if rec:
                yield rec

    def close(self) -> None:
        self.conn.close()
