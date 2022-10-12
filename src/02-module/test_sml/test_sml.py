from module2.sml import cc_features
from sml import synthetic_data
from unittest import TestCase
from datetime import datetime, date
import pytest
from contextlib import nullcontext as does_not_raise


@pytest.mark.parametrize(
    "credit_card_number, cash_amounts, length, delta, radius, country_code, excp",
    [("1111 2222 3333 4444", [112.10, 11.23], 1, 1, 10.0, 'US', does_not_raise())
        , ("1111 2222 3333 44", [-12.00], -1, 1, 1.0, 'IE', pytest.raises(Exception))]
)
def test_generate_atm_withdrawal(credit_card_number: str, cash_amounts: list, length: int, \
                                 delta: int, radius: float, country_code, excp):
    with excp:
        synthetic_data.generate_atm_withdrawal(credit_card_number, cash_amounts, length, delta, radius, country_code)


# add new unit test exercise as optional task for homework for week 2 lab
def test_time_delta():
    date_time1 = int((datetime(2020, 5, 17, 10, 35, 00).strftime("%Y%m%d%H%M%S")))
    date_time2 = int((datetime(2020, 5, 20, 15, 45, 00).strftime("%Y%m%d%H%M%S")))
    expected_time_delta = -3051000
    assert(cc_features.time_delta(date_time1, date_time2)) == expected_time_delta