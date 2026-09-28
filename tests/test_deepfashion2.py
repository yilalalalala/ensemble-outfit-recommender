import json
from types import SimpleNamespace

import numpy as np
from PIL import Image

from ensemble.vision.deepfashion2 import DomainAdapters, _mask_crop, _metrics


def test_segmentation_background_removal():
    im = Image.new("RGB", (10, 10), "red")
    row = SimpleNamespace(x1=0, y1=0, x2=10, y2=10,
                          segmentation=json.dumps([[2, 2, 7, 2, 7, 7, 2, 7]]))
    out = _mask_crop(im, row, True)
    assert out.getpixel((0, 0)) == (255, 255, 255)
    assert out.getpixel((4, 4)) == (255, 0, 0)


def test_exact_match_metrics_are_identity_based_and_category_filtered():
    q = np.asarray([[1, 0], [0, 1]], dtype=np.float32)
    g = np.asarray([[1, 0], [0, 1], [1, 0]], dtype=np.float32)
    out = _metrics(q, g, np.asarray(["a", "b"]), np.asarray(["a", "b", "x"]),
                   np.asarray([1, 2]), np.asarray([1, 2, 2]))
    assert out["overall"]["recall@1"] == 1.0


def test_adapter_starts_as_identity_mapping():
    import torch
    model = DomainAdapters(4, 2)
    x = torch.randn(3, 4)
    expected = torch.nn.functional.normalize(x, dim=-1)
    assert torch.allclose(model.user(x), expected)
    assert torch.allclose(model.shop(x), expected)
