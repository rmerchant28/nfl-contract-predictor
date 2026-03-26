"""Tests for scrapers/pfr.py — _make_name_key and build_pre_contract_stats."""

import numpy as np
import pandas as pd
import pytest

from scrapers.pfr import _make_name_key, build_pre_contract_stats


# ── _make_name_key ─────────────────────────────────────────────────────────────

class TestMakeNameKey:
    def test_full_name(self):
        assert _make_name_key("Patrick Mahomes") == "p.mahomes"

    def test_abbreviated_name(self):
        assert _make_name_key("P.Mahomes") == "p.mahomes"

    def test_abbreviated_with_space(self):
        assert _make_name_key("P. Mahomes") == "p.mahomes"

    def test_lowercase(self):
        assert _make_name_key("travis kelce") == "t.kelce"

    def test_suffix_uses_last_part(self):
        # 'Patrick Mahomes II' → last part is 'ii'
        assert _make_name_key("Patrick Mahomes II") == "p.ii"

    def test_single_word(self):
        assert _make_name_key("Odell") == "odell"

    def test_empty_string(self):
        assert _make_name_key("") == ""

    def test_non_string(self):
        assert _make_name_key(None) == ""
        assert _make_name_key(123) == ""

    def test_accented_chars_stripped(self):
        key = _make_name_key("Dé'Anthony Thomas")
        assert "." in key or len(key) > 0  # should produce something


# ── build_pre_contract_stats ───────────────────────────────────────────────────

def _make_stats(seasons, player="patrick mahomes", col="pass_yards", values=None):
    """Create a minimal stats DataFrame for testing."""
    if values is None:
        values = [4000.0 + i * 100 for i in range(len(seasons))]
    df = pd.DataFrame({
        "player_name_norm": [player] * len(seasons),
        "name_key": [_make_name_key(player)] * len(seasons),
        "season": seasons,
        col: values,
        "games": [16] * len(seasons),
        "is_starter": [1] * len(seasons),
    })
    return df


class TestBuildPreContractStats:
    def test_empty_for_unknown_player(self):
        stats = _make_stats([2020, 2021, 2022], player="patrick mahomes")
        result = build_pre_contract_stats(stats, 2023, "unknown player")
        assert result == {}

    def test_mean_computed_correctly(self):
        stats = _make_stats([2020, 2021, 2022], values=[3000.0, 4000.0, 5000.0])
        result = build_pre_contract_stats(stats, 2023, "patrick mahomes",
                                          numeric_cols=["pass_yards"])
        assert result["pass_yards_mean"] == pytest.approx(4000.0)

    def test_last_is_most_recent_season(self):
        stats = _make_stats([2020, 2021, 2022], values=[3000.0, 4000.0, 5000.0])
        result = build_pre_contract_stats(stats, 2023, "patrick mahomes",
                                          numeric_cols=["pass_yards"])
        assert result["pass_yards_last"] == pytest.approx(5000.0)

    def test_max_value(self):
        stats = _make_stats([2020, 2021, 2022], values=[3000.0, 5500.0, 5000.0])
        result = build_pre_contract_stats(stats, 2023, "patrick mahomes",
                                          numeric_cols=["pass_yards"])
        assert result["pass_yards_max"] == pytest.approx(5500.0)

    def test_peak_decline_at_peak_is_zero(self):
        """Player's last season equals their best → peak_decline = 0.0"""
        stats = _make_stats([2020, 2021, 2022], values=[3000.0, 4000.0, 5000.0])
        result = build_pre_contract_stats(stats, 2023, "patrick mahomes",
                                          numeric_cols=["pass_yards"])
        assert result["pass_yards_peak_decline"] == pytest.approx(0.0)

    def test_peak_decline_when_declining(self):
        """Player's last season is half their peak → peak_decline = 0.5"""
        stats = _make_stats([2020, 2021, 2022], values=[5000.0, 5000.0, 2500.0])
        result = build_pre_contract_stats(stats, 2023, "patrick mahomes",
                                          numeric_cols=["pass_yards"])
        assert result["pass_yards_peak_decline"] == pytest.approx(0.5)

    def test_gap_years_zero_when_active_last_season(self):
        """Signing in 2023 after playing 2022 → gap_years = 0"""
        stats = _make_stats([2020, 2021, 2022])
        result = build_pre_contract_stats(stats, 2023, "patrick mahomes",
                                          numeric_cols=["pass_yards"])
        assert result["gap_years"] == 0

    def test_gap_years_one_when_missed_a_season(self):
        """Last active season was 2021, signing in 2023 → gap_years = 1"""
        stats = _make_stats([2019, 2020, 2021])
        result = build_pre_contract_stats(stats, 2023, "patrick mahomes",
                                          numeric_cols=["pass_yards"])
        assert result["gap_years"] == 1

    def test_starter_seasons_counting(self):
        """3 seasons all as starter → starter_seasons = 3"""
        stats = _make_stats([2020, 2021, 2022])
        stats["is_starter"] = [1, 1, 1]
        result = build_pre_contract_stats(stats, 2023, "patrick mahomes",
                                          numeric_cols=["pass_yards"])
        assert result["starter_seasons"] == 3

    def test_recent_demotion_flag(self):
        """Was starter 2 of 3 seasons but not in last → recent_demotion = 1"""
        stats = _make_stats([2020, 2021, 2022])
        stats["is_starter"] = [1, 1, 0]
        result = build_pre_contract_stats(stats, 2023, "patrick mahomes",
                                          numeric_cols=["pass_yards"])
        assert result["recent_demotion"] == 1

    def test_no_recent_demotion_when_still_starting(self):
        """Was starter all 3 seasons → recent_demotion = 0"""
        stats = _make_stats([2020, 2021, 2022])
        stats["is_starter"] = [1, 1, 1]
        result = build_pre_contract_stats(stats, 2023, "patrick mahomes",
                                          numeric_cols=["pass_yards"])
        assert result["recent_demotion"] == 0

    def test_window_limits_to_correct_seasons(self):
        """3-season window before signing_year=2023 → uses 2020, 2021, 2022 only"""
        stats = _make_stats([2019, 2020, 2021, 2022], values=[9999.0, 3000.0, 4000.0, 5000.0])
        result = build_pre_contract_stats(stats, 2023, "patrick mahomes",
                                          window=3, numeric_cols=["pass_yards"])
        # 2019 should be excluded; mean of 3000, 4000, 5000
        assert result["pass_yards_mean"] == pytest.approx(4000.0)

    def test_trend_positive(self):
        """Monotonically increasing values → positive trend"""
        stats = _make_stats([2020, 2021, 2022], values=[3000.0, 4000.0, 5000.0])
        result = build_pre_contract_stats(stats, 2023, "patrick mahomes",
                                          numeric_cols=["pass_yards"])
        assert result["pass_yards_trend"] > 0

    def test_trend_negative(self):
        """Monotonically decreasing values → negative trend"""
        stats = _make_stats([2020, 2021, 2022], values=[5000.0, 4000.0, 3000.0])
        result = build_pre_contract_stats(stats, 2023, "patrick mahomes",
                                          numeric_cols=["pass_yards"])
        assert result["pass_yards_trend"] < 0

    def test_single_season_no_trend_key(self):
        """Only 1 season in window → no trend key (need ≥2 points)"""
        stats = _make_stats([2022], values=[4000.0])
        result = build_pre_contract_stats(stats, 2023, "patrick mahomes",
                                          numeric_cols=["pass_yards"])
        assert "pass_yards_trend" not in result

    def test_name_key_fallback_matching(self):
        """Abbreviated name in data ('p.mahomes') should match full query name"""
        stats = pd.DataFrame({
            "player_name_norm": ["p.mahomes"],  # abbreviated — won't exact-match
            "name_key": ["p.mahomes"],
            "season": [2022],
            "pass_yards": [5250.0],
            "games": [17],
            "is_starter": [1],
        })
        result = build_pre_contract_stats(stats, 2023, "patrick mahomes",
                                          numeric_cols=["pass_yards"])
        assert result.get("pass_yards_mean") == pytest.approx(5250.0)
