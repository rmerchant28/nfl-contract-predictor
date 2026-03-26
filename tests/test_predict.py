"""Tests for predict.py — _normalise, _name_key, find_comps."""

import json
from pathlib import Path

import pandas as pd
import pytest

from predict import _normalise, _name_key, find_comps


# ── _normalise ─────────────────────────────────────────────────────────────────

class TestNormalise:
    def test_lowercase(self):
        assert _normalise("Patrick Mahomes") == "patrick mahomes"

    def test_strips_whitespace(self):
        assert _normalise("  Josh Allen  ") == "josh allen"

    def test_accented_chars_stripped(self):
        result = _normalise("Dé'Anthony Thomas")
        assert result == result.lower()

    def test_already_normalised(self):
        assert _normalise("travis kelce") == "travis kelce"


# ── _name_key ──────────────────────────────────────────────────────────────────

class TestNameKey:
    def test_full_name(self):
        assert _name_key("Patrick Mahomes") == "p.mahomes"

    def test_abbreviated_with_dot(self):
        assert _name_key("P.Mahomes") == "p.mahomes"

    def test_abbreviated_with_space(self):
        assert _name_key("P. Mahomes") == "p.mahomes"

    def test_case_insensitive(self):
        assert _name_key("TRAVIS KELCE") == "t.kelce"

    def test_single_name(self):
        # Only one part → return as-is (normalised)
        assert _name_key("Odell") == "odell"

    def test_consistency_with_full_vs_abbreviated(self):
        """Full name and abbreviated form produce the same key."""
        assert _name_key("Josh Allen") == _name_key("J.Allen")


# ── find_comps ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def contracts_csv(tmp_path):
    """Write a minimal contracts_with_cap_pct.csv and patch ROOT."""
    data = pd.DataFrame({
        "player_name": ["A", "B", "C", "D"],
        "team":        ["KC", "BUF", "SF", "DAL"],
        "position":    ["QB", "QB", "QB", "WR"],
        "signing_year": [2020, 2021, 2022, 2022],
        "apy":         [40_000_000, 45_000_000, 50_000_000, 30_000_000],
        "apy_pct_cap": [0.20, 0.22, 0.24, 0.15],
        "contract_years": [4, 4, 5, 3],
    })
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    csv_path = raw_dir / "contracts_with_cap_pct.csv"
    data.to_csv(csv_path, index=False)

    import predict
    original_root = predict.ROOT
    predict.ROOT = tmp_path
    yield tmp_path
    predict.ROOT = original_root


class TestFindComps:
    def test_returns_dataframe(self, contracts_csv):
        result = find_comps("QB", predicted_cap_pct=21.0, signing_year=2023, n=3)
        assert isinstance(result, pd.DataFrame)

    def test_returns_correct_columns(self, contracts_csv):
        result = find_comps("QB", predicted_cap_pct=21.0, signing_year=2023, n=3)
        assert list(result.columns) == ["Player", "Team", "Year", "APY", "Cap %", "Years"]

    def test_filters_by_position(self, contracts_csv):
        """Should only return QB comps, not WR."""
        result = find_comps("QB", predicted_cap_pct=21.0, signing_year=2023, n=5)
        # All 3 QB rows are in year < 2023
        assert len(result) == 3

    def test_filters_by_signing_year(self, contracts_csv):
        """Comps must be from years strictly before signing_year."""
        result = find_comps("QB", predicted_cap_pct=21.0, signing_year=2022, n=5)
        # Only 2020 and 2021 rows pass the < 2022 filter
        assert len(result) == 2
        assert all(result["Year"] < 2022)

    def test_closest_cap_pct_returned_first(self, contracts_csv):
        """Row closest to target cap % should appear first (nsmallest ordering)."""
        result = find_comps("QB", predicted_cap_pct=22.0, signing_year=2023, n=3)
        # 2021 row has apy_pct_cap=0.22 → diff=0, should be first
        assert result["Year"].iloc[0] == 2021

    def test_n_limits_results(self, contracts_csv):
        result = find_comps("QB", predicted_cap_pct=21.0, signing_year=2023, n=1)
        assert len(result) == 1

    def test_returns_empty_when_no_csv(self, tmp_path, monkeypatch):
        """If contracts CSV doesn't exist, return empty DataFrame."""
        import predict
        monkeypatch.setattr(predict, "ROOT", tmp_path)  # no csv in tmp_path
        result = find_comps("QB", predicted_cap_pct=21.0, signing_year=2023)
        assert result.empty

    def test_returns_empty_when_no_matching_position(self, contracts_csv):
        result = find_comps("TE", predicted_cap_pct=10.0, signing_year=2023)
        assert result.empty

    def test_apy_formatted_as_dollars(self, contracts_csv):
        result = find_comps("QB", predicted_cap_pct=21.0, signing_year=2023, n=3)
        # APY column should start with '$'
        assert all(str(v).startswith("$") or v == "—" for v in result["APY"])
