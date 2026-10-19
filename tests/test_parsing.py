"""Garment boxes from clothes parsing (vision/parsing.py): the parsed label map, not the LLM, decides where a
piece is; the LLM's loose box only picks which person's piece it is. No model needed: label maps are synthetic."""
import numpy as np

from ensemble.vision.parsing import CLASSES, box_for, regions

H, W = 400, 300


def labels(*parts):
    """A label map with rectangles of parsing classes: parts = (class name, x0, y0, x1, y1)."""
    a = np.zeros((H, W), np.uint8)
    for name, x0, y0, x1, y1 in parts:
        a[y0:y1, x0:x1] = CLASSES[name]
    return a


def close(b, want, tol=4):
    return b is not None and all(abs(x - y) <= tol for x, y in zip(b, want))


def test_regions_are_connected_components():
    m = np.zeros((10, 10), bool); m[1:3, 1:3] = True; m[6:9, 5:8] = True
    assert sorted(r[:4] for r in regions(m)) == [(1, 1, 3, 3), (5, 6, 8, 9)]


def test_a_loose_llm_box_snaps_to_the_parsed_garment():
    lab = labels(("Upper-clothes", 100, 80, 200, 220), ("Pants", 110, 220, 190, 380))
    assert close(box_for(lab, "top", [60, 40, 160, 150]), [100, 80, 200, 220])
    assert close(box_for(lab, "bottom", None), [110, 220, 190, 380])


def test_shoes_take_both_shoes():
    lab = labels(("Left-shoe", 110, 380, 140, 395), ("Right-shoe", 160, 380, 190, 396))
    assert close(box_for(lab, "shoes", None), [110, 380, 190, 396])


def test_the_llm_box_picks_the_right_person_in_a_group_photo():
    lab = labels(("Upper-clothes", 10, 50, 90, 200), ("Upper-clothes", 200, 60, 290, 210))
    assert close(box_for(lab, "top", [190, 40, 280, 180]), [200, 60, 290, 210])


def test_a_dress_falls_back_to_upper_clothes_and_skirt_when_no_dress_pixels():
    lab = labels(("Upper-clothes", 100, 80, 200, 200), ("Skirt", 100, 200, 200, 300))
    assert close(box_for(lab, "dress", None), [100, 80, 200, 300])


def test_no_parsing_class_keeps_the_llm_box():
    lab = labels(("Upper-clothes", 100, 80, 200, 220))
    assert box_for(lab, "jewellery", [120, 90, 130, 100]) is None      # no class for jewellery
    assert box_for(lab, "bag", [10, 10, 50, 50]) is None                # no bag pixels
