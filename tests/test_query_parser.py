"""Unit tests for the query parser — pure logic, no models required.

These lock down the behaviour the whole ranking depends on: that a query is decomposed
into the right structured attributes and, crucially, the right color<->garment BINDINGS.
"""
from src.attributes import QueryParser


def _bindings(spec):
    return {(b.color, b.garment_type) for b in spec.bindings if b.is_bound()}


def test_yellow_raincoat():
    spec = QueryParser().parse("A person in a bright yellow raincoat.")
    assert ("yellow", "raincoat") in _bindings(spec)
    assert "yellow" in spec.attributes.get("colors", [])


def test_business_office_scene():
    spec = QueryParser().parse("Professional business attire inside a modern office.")
    assert "office" in spec.attributes.get("environment", [])
    assert "business" in spec.attributes.get("style", [])


def test_blue_shirt_park_bench():
    spec = QueryParser().parse("Someone wearing a blue shirt sitting on a park bench.")
    assert ("blue", "shirt") in _bindings(spec)
    assert "park" in spec.attributes.get("environment", [])


def test_casual_city_walk_infers_style_and_scene():
    spec = QueryParser().parse("Casual weekend outfit for a city walk.")
    assert "casual" in spec.attributes.get("style", [])
    assert "urban street" in spec.attributes.get("environment", [])
    # No explicit garment binding — region_binding must therefore be inactive.
    assert not spec.has_bindings()


def test_compositional_two_bindings():
    spec = QueryParser().parse("A red tie and a white shirt in a formal setting.")
    b = _bindings(spec)
    assert ("red", "tie") in b
    assert ("white", "shirt") in b
    assert "formal" in spec.attributes.get("style", [])
