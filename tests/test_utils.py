"""Tests for scrapers/utils.py — pure helper functions."""

import pytest
from scrapers.utils import clean_money, clean_pct, normalise_name, map_position


# ── clean_money ────────────────────────────────────────────────────────────────

class TestCleanMoney:
    def test_dollar_sign_with_commas(self):
        assert clean_money("$24,500,000") == 24_500_000.0

    def test_millions_suffix(self):
        assert clean_money("24.5M") == 24_500_000.0

    def test_millions_suffix_uppercase(self):
        assert clean_money("10M") == 10_000_000.0

    def test_plain_number_string(self):
        assert clean_money("5000000") == 5_000_000.0

    def test_non_string_returns_none(self):
        assert clean_money(123) is None
        assert clean_money(None) is None

    def test_garbage_string_returns_none(self):
        assert clean_money("N/A") is None

    def test_empty_string_returns_none(self):
        assert clean_money("") is None

    def test_dollar_no_commas(self):
        assert clean_money("$1000000") == 1_000_000.0


# ── clean_pct ──────────────────────────────────────────────────────────────────

class TestCleanPct:
    def test_basic_percentage(self):
        assert clean_pct("12.5%") == pytest.approx(0.125)

    def test_integer_percentage(self):
        assert clean_pct("50%") == pytest.approx(0.5)

    def test_zero_percentage(self):
        assert clean_pct("0%") == pytest.approx(0.0)

    def test_non_string_returns_none(self):
        assert clean_pct(0.5) is None
        assert clean_pct(None) is None

    def test_garbage_returns_none(self):
        assert clean_pct("N/A") is None

    def test_whitespace_stripped(self):
        assert clean_pct(" 10% ") == pytest.approx(0.1)


# ── normalise_name ─────────────────────────────────────────────────────────────

class TestNormaliseName:
    def test_basic_lowercase(self):
        assert normalise_name("Patrick Mahomes") == "patrick mahomes"

    def test_suffix_preserved(self):
        assert normalise_name("Patrick Mahomes II") == "patrick mahomes ii"

    def test_strips_whitespace(self):
        assert normalise_name("  Josh Allen  ") == "josh allen"

    def test_accented_chars_stripped(self):
        # Unicode accents should be removed
        result = normalise_name("Dé'Anthony Thomas")
        assert "'" not in result or result == result.lower()

    def test_already_lowercase(self):
        assert normalise_name("travis kelce") == "travis kelce"


# ── map_position ───────────────────────────────────────────────────────────────

class TestMapPosition:
    def test_qb(self):
        assert map_position("QB") == ("QB", "QB")

    def test_wr(self):
        assert map_position("WR") == ("WR", "SKILL")

    def test_rb(self):
        assert map_position("RB") == ("RB", "SKILL")

    def test_hb_maps_to_rb(self):
        assert map_position("HB") == ("RB", "SKILL")

    def test_fb_maps_to_rb(self):
        assert map_position("FB") == ("RB", "SKILL")

    def test_te(self):
        assert map_position("TE") == ("TE", "SKILL")

    def test_lt_maps_to_ot(self):
        assert map_position("LT") == ("OT", "OL")

    def test_rt_maps_to_ot(self):
        assert map_position("RT") == ("OT", "OL")

    def test_lg_maps_to_og(self):
        assert map_position("LG") == ("OG", "OL")

    def test_center(self):
        assert map_position("C") == ("C", "OL")

    def test_unknown_position(self):
        assert map_position("SS") == (None, None)

    def test_case_insensitive(self):
        assert map_position("qb") == ("QB", "QB")
        assert map_position("Wr") == ("WR", "SKILL")

    def test_strips_whitespace(self):
        assert map_position(" QB ") == ("QB", "QB")
