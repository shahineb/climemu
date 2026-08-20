import pytest
from earthsampler.utils.datetime import parse_doy


class TestParseDoyInt:
    def test_valid_bounds(self):
        assert parse_doy(1) == 1
        assert parse_doy(365) == 365

    def test_mid_year(self):
        assert parse_doy(172) == 172

    def test_zero_raises(self):
        with pytest.raises(ValueError):
            parse_doy(0)

    def test_366_raises(self):
        with pytest.raises(ValueError):
            parse_doy(366)

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            parse_doy(-1)


class TestParseDoyStr:
    def test_jan_first(self):
        assert parse_doy("01/01") == 1

    def test_dec_last(self):
        assert parse_doy("31/12") == 365

    def test_jun_21(self):
        # 31 (Jan) + 28 (Feb) + 31 (Mar) + 30 (Apr) + 31 (May) + 21 = 172
        assert parse_doy("21/06") == 172

    def test_feb_28(self):
        assert parse_doy("28/02") == 59

    def test_mar_first(self):
        assert parse_doy("01/03") == 60

    def test_feb_29_raises(self):
        with pytest.raises(ValueError, match="29/02"):
            parse_doy("29/02")

    def test_invalid_month_raises(self):
        with pytest.raises(ValueError):
            parse_doy("01/13")

    def test_invalid_day_raises(self):
        with pytest.raises(ValueError):
            parse_doy("32/01")

    def test_bad_format_raises(self):
        with pytest.raises(ValueError):
            parse_doy("invalid")

    def test_non_numeric_raises(self):
        with pytest.raises(ValueError):
            parse_doy("ab/cd")


class TestParseDoyDdDashMm:
    def test_jan_first(self):
        assert parse_doy("01-01") == 1

    def test_dec_last(self):
        assert parse_doy("31-12") == 365

    def test_jun_21(self):
        assert parse_doy("21-06") == 172

    def test_feb_28(self):
        assert parse_doy("28-02") == 59

    def test_mar_first(self):
        assert parse_doy("01-03") == 60

    def test_feb_29_raises(self):
        with pytest.raises(ValueError):
            parse_doy("29-02")

    def test_invalid_month_raises(self):
        with pytest.raises(ValueError):
            parse_doy("01-13")

    def test_invalid_day_raises(self):
        with pytest.raises(ValueError):
            parse_doy("32-01")

    def test_non_numeric_raises(self):
        with pytest.raises(ValueError):
            parse_doy("ab-cd")

    def test_consistent_with_dd_slash_mm(self):
        assert parse_doy("21-06") == parse_doy("21/06")
        assert parse_doy("31-12") == parse_doy("31/12")
        assert parse_doy("01-01") == parse_doy("01/01")


class TestParseDoyType:
    def test_float_raises(self):
        with pytest.raises(TypeError):
            parse_doy(1.0)

    def test_none_raises(self):
        with pytest.raises(TypeError):
            parse_doy(None)
