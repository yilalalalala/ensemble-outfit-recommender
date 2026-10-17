"""Fabric swatches (ensemble.serving.swatches): the crop lands on the garment, not the studio backdrop, and keeps
a printed colourway's pattern."""
import io

import numpy as np
from PIL import Image

from ensemble.serving.swatches import fabric_point, swatch

PARAMS = dict(size=32, crop_frac=0.16, bg_threshold=40, erode=[11, 9, 7, 5, 3])


def _product_shot(tmp_path, garment, stripes=False, legs=False):
    """A 2:3 'product photo': grey backdrop, a garment rectangle (optionally two legs with a gap, or stripes)."""
    w, h = 600, 900
    a = np.full((h, w, 3), 236, np.uint8)
    x0, y0, x1, y1 = garment
    a[y0:y1, x0:x1] = (40, 40, 40)
    if stripes:
        a[y0:y1, x0:x1][(np.arange(y1 - y0) // 12) % 2 == 0] = (250, 250, 250)
    if legs:   # trousers: the gap between the legs is backdrop
        a[int(y0 + (y1 - y0) * .35):y1, (x0 + x1) // 2 - 30:(x0 + x1) // 2 + 30] = 236
    p = tmp_path / "shot.jpg"
    Image.fromarray(a).save(p, quality=95)
    return p


def test_point_is_inside_the_garment(tmp_path):
    p = _product_shot(tmp_path, (150, 200, 450, 800))
    x, y = fabric_point(Image.open(p).convert("RGB"), 40, PARAMS["erode"])
    assert 150 / 600 < x < 450 / 600 and 200 / 900 < y < 800 / 900


def test_point_avoids_the_gap_between_trouser_legs(tmp_path):
    p = _product_shot(tmp_path, (150, 100, 450, 850), legs=True)
    x, y = fabric_point(Image.open(p).convert("RGB"), 40, PARAMS["erode"])
    px, py = int(x * 600), int(y * 900)
    in_gap = py > 100 + 750 * .35 and 270 <= px <= 330
    assert not in_gap


def test_swatch_is_small_and_keeps_a_print(tmp_path):
    plain, _ = swatch(_product_shot(tmp_path, (150, 200, 450, 800)), **PARAMS)
    striped, hx = swatch(_product_shot(tmp_path, (150, 200, 450, 800), stripes=True), **PARAMS)
    im = Image.open(io.BytesIO(striped))
    assert im.size == (32, 32) and len(striped) < 4000 and hx.startswith("#")
    assert np.asarray(im.convert("L")).std() > 40                       # stripes survive the crop
    assert np.asarray(Image.open(io.BytesIO(plain)).convert("L")).std() < 10
