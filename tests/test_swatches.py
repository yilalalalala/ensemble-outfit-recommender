"""Colour swatches read from product photos (ensemble.serving.swatches): the colour comes from a clean patch of the
garment's main colour (not the centre, where buttons and lapels sit), plain pieces snap to the standard colour card
inside the catalogue colour's family, and patterned pieces are shown as a crop of their fabric."""
import io

import numpy as np
from PIL import Image

from ensemble.serving.swatches import CARD, FAMILIES, analyse, family_of, snap, swatch, to_lab

W, H = 600, 900
BACKDROP = (236, 235, 233)


def shot(colour, garment=(150, 150, 450, 800), buttons=False, stripes=None, legs=False):
    """A 2:3 product shot on the studio backdrop: one garment, optionally with dark buttons down the centre,
    horizontal stripes of a second colour, or two trouser legs with backdrop between them."""
    a = np.zeros((H, W, 3), np.uint8); a[:] = BACKDROP
    x0, y0, x1, y1 = garment
    a[y0:y1, x0:x1] = colour
    if stripes is not None:
        rows = np.arange(y0, y1)[(np.arange(y1 - y0) // 14) % 2 == 0]
        a[rows, x0:x1] = stripes
    if buttons:   # a placket of big dark buttons right at the centre of the photo
        for cy in range(380, 620, 60):
            a[cy - 18:cy + 18, W // 2 - 18:W // 2 + 18] = (20, 20, 20)
    if legs:
        a[int(y0 + (y1 - y0) * .35):y1, (x0 + x1) // 2 - 30:(x0 + x1) // 2 + 30] = BACKDROP
    return Image.fromarray(a)


def test_plain_colour_comes_from_a_clean_patch_not_the_buttons_at_the_centre():
    r = analyse(shot((232, 220, 196), buttons=True), "Solid", "Light Beige")
    assert r["kind"] == "plain" and r["card"] == "Light Beige" and r["hex"] == CARD["Light Beige"]
    x0, y0, x1, y1 = r["box"]
    centre_x = W // 2 / W
    assert x1 < centre_x - 18 / W or x0 > centre_x + 18 / W or y1 < 362 / H or y0 > 638 / H   # off the button column


def test_patch_avoids_the_gap_between_trouser_legs():
    r = analyse(shot((31, 42, 68), garment=(150, 100, 450, 850), legs=True), "Solid", "Dark Blue")
    x0, y0, x1, y1 = (v * s for v, s in zip(r["box"], (W, H, W, H)))
    gap_top = 100 + 750 * .35
    assert not (y1 > gap_top and x0 < 330 and x1 > 270)
    assert r["card"] in FAMILIES["blue"]


def test_a_pattern_is_shown_as_its_fabric_when_catalogue_and_photo_agree(tmp_path):
    p = tmp_path / "stripe.jpg"
    shot((20, 20, 20), stripes=(245, 245, 245)).save(p, quality=95)
    kind, card, hx, jpeg = swatch(p, "Stripe", "Black", 32)
    im = Image.open(io.BytesIO(jpeg))
    assert kind == "pattern" and card is None and im.size == (32, 32) and len(jpeg) < 4000
    assert np.asarray(im.convert("L")).std() > 40                         # the stripes survive the crop


def test_a_fine_houndstooth_check_is_still_a_pattern(tmp_path):
    """A micro-pattern averages into one colour at analysis resolution; the catalogue's "Check" keeps it a pattern,
    and the full-resolution crop keeps the pattern visible (the Hanoi Oversize Blazer case)."""
    a = np.zeros((H, W, 3), np.uint8); a[:] = BACKDROP
    yy, xx = np.mgrid[150:800, 150:450]
    a[150:800, 150:450] = np.where((((yy // 3) + (xx // 3)) % 2 == 0)[..., None], (60, 45, 40), (225, 215, 200))
    p = tmp_path / "houndstooth.jpg"
    Image.fromarray(a).save(p, quality=95)
    kind, card, hx, jpeg = swatch(p, "Check", "Dark Orange", 32)
    assert kind == "pattern" and card is None
    assert np.asarray(Image.open(io.BytesIO(jpeg)).convert("L")).std() > 25


def test_catalogue_says_solid_so_a_stripy_photo_still_gets_one_colour():
    r = analyse(shot((20, 20, 20), stripes=(60, 60, 60)), "Solid", "Black")
    assert r["kind"] == "plain" and r["card"] in FAMILIES["black"]


def test_snap_stays_inside_the_named_colour_family():
    near_grey_green = to_lab(np.array([62, 66, 62]))                    # photographs almost neutral
    assert snap(near_grey_green) in ("Dark Grey", "Charcoal")
    assert snap(near_grey_green, family_of("Dark Green")) in FAMILIES["green"]


def test_named_colour_wins_when_the_photo_agrees_and_loses_when_it_does_not():
    beige = to_lab(np.array([209, 204, 188]))                            # a light-beige coat, photographed dim
    assert snap(beige, "beige", "Light Beige") == "Light Beige"
    red = to_lab(np.array([200, 30, 40]))
    assert snap(red, "red", "Dark Red") == "Red"                         # too far from Dark Red: nearest red shade


def test_family_of_catalogue_colour_names():
    assert family_of("Greenish Khaki") == "green" and family_of("Light Beige") == "beige"
    assert family_of("Off White") == "white" and family_of("Dark Turquoise") == "turquoise"
    assert family_of("Unknown") is None and family_of("Other") is None
    assert all(n in CARD for names in FAMILIES.values() for n in names)
