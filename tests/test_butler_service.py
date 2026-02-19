from __future__ import annotations

import pytest
from astropy.time import Time
import numpy as np

import neandertools as nt


class FakeBBox:
    def getMinX(self):
        return 0

    def getMaxX(self):
        return 100

    def getMinY(self):
        return 0

    def getMaxY(self):
        return 100


class FakeImage:
    def __init__(self, token="root", array=None):
        self.token = token
        if array is None:
            self._array = np.arange(101 * 101, dtype=np.float32).reshape(101, 101)
        else:
            self._array = array

    def getBBox(self):
        return FakeBBox()

    def getArray(self):
        return self._array

    @staticmethod
    def Factory(*args):
        if len(args) == 2:
            image, bbox = args
            if hasattr(image, "getMinX") and hasattr(image, "getMaxX"):
                w = image.getMaxX() - image.getMinX() + 1
                h = image.getMaxY() - image.getMinY() + 1
                return FakeImage(token="blank", array=np.zeros((h, w), dtype=np.float32))
            if (
                bbox.getMinX() < image.getBBox().getMinX()
                or bbox.getMinY() < image.getBBox().getMinY()
                or bbox.getMaxX() > image.getBBox().getMaxX()
                or bbox.getMaxY() > image.getBBox().getMaxY()
            ):
                raise ValueError("bbox outside image bounds")

            x0 = bbox.getMinX() - image.getBBox().getMinX()
            y0 = bbox.getMinY() - image.getBBox().getMinY()
            x1 = bbox.getMaxX() - image.getBBox().getMinX() + 1
            y1 = bbox.getMaxY() - image.getBBox().getMinY() + 1
            return FakeImage(token="cutout", array=image.getArray()[y0:y1, x0:x1].copy())

        (bbox,) = args
        w = bbox.getMaxX() - bbox.getMinX() + 1
        h = bbox.getMaxY() - bbox.getMinY() + 1
        return FakeImage(token="blank", array=np.zeros((h, w), dtype=np.float32))

    class _Wcs:
        class _Pixel:
            def __init__(self, x, y):
                self._x = x
                self._y = y

            def getX(self):
                return self._x

            def getY(self):
                return self._y

        def skyToPixel(self, _sky):
            return FakeImage._Wcs._Pixel(40.0, 60.0)

    def getWcs(self):
        return FakeImage._Wcs()


class FakeButler:
    def __init__(self):
        self.calls = []

    def get(self, dataset_type, dataId=None):
        self.calls.append((dataset_type, dataId))
        return {"data": dataId} if dataset_type == "raw" else FakeImage()


class FakeImageSilentClip(FakeImage):
    @staticmethod
    def Factory(*args):
        if len(args) == 2:
            image, bbox = args
            if hasattr(image, "getMinX") and hasattr(image, "getMaxX"):
                w = image.getMaxX() - image.getMinX() + 1
                h = image.getMaxY() - image.getMinY() + 1
                return FakeImageSilentClip(token="blank", array=np.zeros((h, w), dtype=np.float32))
            x0 = max(bbox.getMinX(), image.getBBox().getMinX())
            y0 = max(bbox.getMinY(), image.getBBox().getMinY())
            x1 = min(bbox.getMaxX(), image.getBBox().getMaxX())
            y1 = min(bbox.getMaxY(), image.getBBox().getMaxY())

            sx0 = x0 - image.getBBox().getMinX()
            sy0 = y0 - image.getBBox().getMinY()
            sx1 = x1 - image.getBBox().getMinX() + 1
            sy1 = y1 - image.getBBox().getMinY() + 1
            return FakeImageSilentClip(token="cutout", array=image.getArray()[sy0:sy1, sx0:sx1].copy())

        (bbox,) = args
        w = bbox.getMaxX() - bbox.getMinX() + 1
        h = bbox.getMaxY() - bbox.getMinY() + 1
        return FakeImageSilentClip(token="blank", array=np.zeros((h, w), dtype=np.float32))


class FakeButlerSilentClip(FakeButler):
    def get(self, dataset_type, dataId=None):
        self.calls.append((dataset_type, dataId))
        return {"data": dataId} if dataset_type == "raw" else FakeImageSilentClip()


def test_factory_uses_provided_butler():
    butler = FakeButler()
    svc = nt.cutouts_from_butler("dp1", collections="test", butler=butler)
    assert isinstance(svc, nt.ButlerCutoutService)


def test_visit_detector_cutout_calls_butler_default_dataset_type():
    butler = FakeButler()
    svc = nt.cutouts_from_butler("dp1", collections="test", butler=butler)

    out = svc.cutout(visit=123, detector=9, x=50, y=50, h=9, w=9)

    assert len(out) == 1
    assert out[0].token in {"root", "cutout"}
    assert butler.calls == [("visit_image", {"visit": 123, "detector": 9})]


def test_edge_cutout_is_padded_by_default():
    butler = FakeButler()
    svc = nt.cutouts_from_butler("dp1", collections="test", butler=butler)

    out = svc.cutout(visit=123, detector=9, x=0, y=0, h=5, w=5)
    arr = out[0].getArray()

    assert arr.shape == (5, 5)
    assert arr[2, 2] == 0  # requested center pixel maps to image(0, 0)
    assert np.isnan(arr[0, 0])  # padded corner
    assert arr[2, 3] == 1  # image(1, 0)
    assert arr[3, 2] == 101  # image(0, 1)


def test_edge_cutout_can_disable_padding():
    butler = FakeButler()
    svc = nt.cutouts_from_butler("dp1", collections="test", butler=butler)

    out = svc.cutout(visit=123, detector=9, x=0, y=0, h=5, w=5, pad=False)
    arr = out[0].getArray()

    assert arr.shape == (3, 3)
    assert arr[0, 0] == 0


def test_edge_cutout_pads_when_factory_silently_clips():
    butler = FakeButlerSilentClip()
    svc = nt.cutouts_from_butler("dp1", collections="test", butler=butler)

    out = svc.cutout(visit=123, detector=9, x=0, y=0, h=5, w=5)
    arr = out[0].getArray()

    assert arr.shape == (5, 5)
    assert arr[2, 2] == 0
    assert np.isnan(arr[0, 0])


def test_visit_detector_cutout_defaults_to_full_image_geometry():
    butler = FakeButler()
    svc = nt.cutouts_from_butler("dp1", collections="test", butler=butler)

    out = svc.cutout(visit=123, detector=9, x=50, y=50)

    assert len(out) == 1
    assert out[0].token in {"root", "cutout"}
    assert butler.calls == [("visit_image", {"visit": 123, "detector": 9})]


def test_visit_detector_cutout_with_radec_center_uses_wcs():
    butler = FakeButler()
    svc = nt.cutouts_from_butler("dp1", collections="test", butler=butler)

    out = svc.cutout(visit=123, detector=9, ra=12.3, dec=-4.5, h=11, w=11)

    assert len(out) == 1
    assert out[0].token in {"root", "cutout"}
    assert butler.calls == [("visit_image", {"visit": 123, "detector": 9})]


def test_visit_detector_cutout_accepts_arrays():
    butler = FakeButler()
    svc = nt.cutouts_from_butler("dp1", collections="test", butler=butler)

    out = svc.cutout(
        visit=[123, 124],
        detector=[9, 10],
        ra=[12.3, 12.4],
        dec=[-4.5, -4.6],
        h=11,
        w=11,
    )

    assert len(out) == 2
    assert [call[1] for call in butler.calls] == [
        {"visit": 123, "detector": 9},
        {"visit": 124, "detector": 10},
    ]


def test_visit_detector_cutout_accepts_numpy_arrays():
    np = pytest.importorskip("numpy")
    butler = FakeButler()
    svc = nt.cutouts_from_butler("dp1", collections="test", butler=butler)

    out = svc.cutout(
        visit=np.array([123, 124], dtype=int),
        detector=np.array([9, 10], dtype=int),
        ra=12.3,
        dec=-4.5,
        h=11,
        w=11,
    )

    assert len(out) == 2
    assert [call[1] for call in butler.calls] == [
        {"visit": 123, "detector": 9},
        {"visit": 124, "detector": 10},
    ]


def test_invalid_args():
    butler = FakeButler()
    svc = nt.cutouts_from_butler("dp1", collections="test", butler=butler)

    with pytest.raises(ValueError):
        svc.cutout(x=10, y=10, h=5, w=5)

    with pytest.raises(ValueError):
        svc.cutout(ra=1, dec=2, visit=1, detector=2, x=10, y=10, h=5, w=5)

    with pytest.raises(ValueError):
        svc.cutout(visit=1, x=10, y=10, h=5, w=5)

    with pytest.raises(ValueError):
        svc.cutout(ra=1, dec=2, x=10, y=10, h=0, w=5)

    with pytest.raises(ValueError):
        svc.cutout(ra=1, dec=2, x=10, y=10, h=-1, w=5)

    with pytest.raises(ValueError):
        svc.cutout(ra=1, dec=2, x=10, y=10, h=5, w=0)

    with pytest.raises(ValueError):
        svc.cutout(visit=1, detector=2, h=5, w=5)

    with pytest.raises(ValueError):
        svc.cutout(visit=1, detector=2, x=10, h=5, w=5)

    with pytest.raises(ValueError):
        svc.cutout(visit=1, detector=2, ra=10, h=5, w=5)

    with pytest.raises(ValueError):
        svc.cutout(ra=1, dec=2, h=5, w=5)

    with pytest.raises(ValueError):
        svc.cutout(visit=[1, 2], detector=[3, 4, 5], ra=[10, 11], dec=[20, 21], h=5, w=5)


def test_find_visit_detector_scalar_and_vector():
    np = pytest.importorskip("numpy")

    class AlwaysContainsRegion:
        def contains(self, _point):
            return True

    butler = FakeButler()
    svc = nt.cutouts_from_butler("dp1", collections="test", butler=butler)
    svc._get_visit_detector_index = lambda _dataset_type: [
        {
            "visit": 101,
            "detector": 5,
            "region": AlwaysContainsRegion(),
            "begin": Time("2024-01-01T00:00:00", scale="tai"),
            "end": Time("2024-01-01T00:01:00", scale="tai"),
        }
    ]

    one = svc.find_visit_detector(ra=53.0, dec=-27.9, t="2024-01-01T00:00:30")
    many = svc.find_visit_detector(
        ra=[53.0, 53.0],
        dec=[-27.9, -27.9],
        t=["2024-01-01T00:00:30", "2024-01-01T00:02:00"],
    )

    assert isinstance(one[0], np.ndarray)
    assert isinstance(one[1], np.ndarray)
    assert one[0].tolist() == [101]
    assert one[1].tolist() == [5]
    assert many[0].tolist() == [101]
    assert many[1].tolist() == [5]


def test_find_visit_detector_length_mismatch_raises():
    butler = FakeButler()
    svc = nt.cutouts_from_butler("dp1", collections="test", butler=butler)
    with pytest.raises(ValueError):
        svc.find_visit_detector(ra=[53.0, 53.1], dec=[-27.9], t=["2024-01-01T00:00:30", "2024-01-01T00:00:40", "2024-01-01T00:00:50"])
