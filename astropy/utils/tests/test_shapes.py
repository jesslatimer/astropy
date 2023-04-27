# Licensed under a 3-clause BSD style license - see LICENSE.rst

import atheris
import sys

import numpy as np
import pytest
from hypothesis import given
from hypothesis.extra.numpy import basic_indices, arrays
from numpy.testing import assert_equal

with atheris.instrument_imports():
    from astropy.utils.shapes import check_broadcast, simplify_basic_index, unbroadcast


def test_check_broadcast():
    assert check_broadcast((10, 1), (3,)) == (10, 3)
    assert check_broadcast((10, 1), (3,), (4, 1, 1, 3)) == (4, 1, 10, 3)
    with pytest.raises(ValueError):
        check_broadcast((10, 2), (3,))

    with pytest.raises(ValueError):
        check_broadcast((10, 1), (3,), (4, 1, 2, 3))


def test_unbroadcast():
    x = np.array([1, 2, 3])
    y = np.broadcast_to(x, (2, 4, 3))
    z = unbroadcast(y)
    assert z.shape == (3,)
    np.testing.assert_equal(z, x)

    x = np.ones((3, 5))
    y = np.broadcast_to(x, (5, 3, 5))
    z = unbroadcast(y)
    assert z.shape == (3, 5)


TEST_SHAPE = (13, 16, 4, 90)


# class TestSimplifyBasicIndex:
    # We use a class here so that we can allocate the data once and for all to
    # speed up the testing.

# def setup_class(self):
test_shape = TEST_SHAPE
rand_data = np.random.random(TEST_SHAPE)

@given(index=basic_indices(TEST_SHAPE), data=arrays(int, TEST_SHAPE))
@atheris.instrument_func
def test_indexing(index, data):
    new_index = simplify_basic_index(index, shape=TEST_SHAPE)
    assert_equal(data[index], data[new_index])
    assert isinstance(new_index, tuple)
    assert len(new_index) == len(TEST_SHAPE)
    for idim, idx in enumerate(new_index):
        assert isinstance(idx, (slice, int))
        if isinstance(idx, int):
            assert idx >= 0
        else:
            assert isinstance(idx.start, int)
            assert idx.start >= 0
            assert idx.start < TEST_SHAPE[idim]
            if idx.stop is not None:
                assert isinstance(idx.stop, int)
                assert idx.stop >= 0
                assert idx.stop <= TEST_SHAPE[idim]
            assert isinstance(idx.step, int)

if __name__ == "__main__":
    atheris.Setup(sys.argv, test_indexing.hypothesis.fuzz_one_input)
    atheris.Fuzz()
