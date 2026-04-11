
import mathlib
import pytest
import sys

@pytest.mark.skipif((sys.version_info) > (3,5), reason="I dont want to run this test case")
def test_cal_total():
    total = mathlib.calc_total(4,5)
    assert total == 9

def test_calc_mult():
    total = mathlib.calc_mult(6,5)
    assert total == 30