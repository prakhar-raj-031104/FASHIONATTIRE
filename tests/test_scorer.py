"""The headline test: region-binding defeats CLIP's bag-of-words failure.

We build two images that whole-image CLIP CANNOT tell apart (identical global vectors,
identical image-level color attributes): one is a RED tie + WHITE shirt, the other the
swap. Only the region-binding signal — which checks that 'red' sits on the *tie* region
and 'white' on the *shirt* region — should rank the correct image first.
"""
import numpy as np

from src.retrieval.scorer import HybridScorer
from src.utils.schema import GarmentBinding, ImageRecord, QuerySpec, RegionRecord

CFG = {
    "weights": {"global_clip": 0.4, "caption_sim": 0.2,
                "attribute_match": 0.2, "region_binding": 0.2},
    "attribute_match": {"mode": "soft"},
    "region_binding": {"color_weight": 0.5, "garment_weight": 0.5},
}


def _img(image_id, tie_color, shirt_color):
    regions = [
        RegionRecord(region_id=image_id * 10, image_id=image_id, garment_label="scarf",
                     garment_type="tie", bbox=[0, 0, 10, 10], area_frac=0.02,
                     colors={tie_color: 0.9}, type_scores={"tie": 0.8}),
        RegionRecord(region_id=image_id * 10 + 1, image_id=image_id,
                     garment_label="upper-clothes", garment_type="shirt",
                     bbox=[0, 0, 20, 20], area_frac=0.2,
                     colors={shirt_color: 0.9}, type_scores={"shirt": 0.8}),
    ]
    # Identical image-level color attributes -> attribute_match cannot disambiguate.
    attrs = {"colors": {"red": 0.5, "white": 0.5}, "style": {"formal": 0.8}}
    return ImageRecord(image_id=image_id, image_path=f"/img/{image_id}.jpg",
                       caption="a person in formal wear", attributes=attrs, regions=regions)


def test_region_binding_breaks_the_tie():
    correct = _img(1, tie_color="red", shirt_color="white")   # matches the query
    swapped = _img(2, tie_color="white", shirt_color="red")   # the CLIP trap

    query = QuerySpec(
        raw="a red tie and a white shirt in a formal setting",
        attributes={"colors": ["red", "white"], "accessories": ["tie"],
                    "upper_garment": ["shirt"], "style": ["formal"]},
        bindings=[GarmentBinding("red", "tie"), GarmentBinding("white", "shirt")],
    )

    # Identical global vectors: whole-image CLIP is blind to the difference.
    q_clip = np.ones(4, dtype=np.float32) / 2
    global_vecs = np.stack([q_clip, q_clip])

    scored = HybridScorer(CFG).score(
        query, q_clip, None, [correct, swapped], global_vecs, None)

    assert scored[0].image_id == 1, "region binding should rank the correct image first"
    assert scored[0].signals["region_binding"] > scored[1].signals["region_binding"]
