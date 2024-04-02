import pytest

import mim_nlp


def test_tests():
    assert True
    with pytest.raises(AssertionError):
        assert False


def test_package_availability():
    assert mim_nlp is not None
