"""Tests for scrapers/overthecap.py — add_cap_percentage."""

import numpy as np
import pandas as pd
import pytest

from scrapers.overthecap import add_cap_percentage


def _contracts(**kwargs):
    base = {
        "player_name": ["Patrick Mahomes"],
        "position": ["QB"],
        "apy": [45_000_000.0],
        "guaranteed": [180_000_000.0],
        "signing_year": [2022],
    }
    base.update(kwargs)
    return pd.DataFrame(base)


def _caps(year_cap_pairs):
    return pd.DataFrame({"year": [y for y, _ in year_cap_pairs],
                         "salary_cap": [c for _, c in year_cap_pairs]})


class TestAddCapPercentage:
    def test_apy_pct_cap_computed(self):
        contracts = _contracts(apy=[20_820_000.0], signing_year=[2022])
        cap_history = _caps([(2022, 208_200_000)])
        result = add_cap_percentage(contracts, cap_history)
        assert "apy_pct_cap" in result.columns
        assert result["apy_pct_cap"].iloc[0] == pytest.approx(0.1)

    def test_guaranteed_pct_cap_computed(self):
        contracts = _contracts(guaranteed=[104_100_000.0], signing_year=[2022])
        cap_history = _caps([(2022, 208_200_000)])
        result = add_cap_percentage(contracts, cap_history)
        assert "guaranteed_pct_cap" in result.columns
        assert result["guaranteed_pct_cap"].iloc[0] == pytest.approx(0.5)

    def test_missing_cap_year_gives_nan(self):
        """Contract signed in a year with no cap data → both pct columns are NaN."""
        contracts = _contracts(signing_year=[1990])
        cap_history = _caps([(2022, 208_200_000)])
        result = add_cap_percentage(contracts, cap_history)
        assert pd.isna(result["apy_pct_cap"].iloc[0])
        assert pd.isna(result["guaranteed_pct_cap"].iloc[0])

    def test_multiple_years(self):
        contracts = pd.DataFrame({
            "player_name": ["A", "B"],
            "position": ["QB", "WR"],
            "apy": [18_820_000.0, 22_000_000.0],
            "guaranteed": [50_000_000.0, 60_000_000.0],
            "signing_year": [2021, 2022],
        })
        cap_history = _caps([(2021, 182_500_000), (2022, 208_200_000)])
        result = add_cap_percentage(contracts, cap_history)

        expected_2021 = 18_820_000.0 / 182_500_000
        expected_2022 = 22_000_000.0 / 208_200_000
        assert result.loc[result["signing_year"] == 2021, "apy_pct_cap"].iloc[0] == pytest.approx(expected_2021)
        assert result.loc[result["signing_year"] == 2022, "apy_pct_cap"].iloc[0] == pytest.approx(expected_2022)

    def test_cap_that_year_column_present(self):
        """Merged DataFrame should contain the raw cap column for downstream use."""
        contracts = _contracts()
        cap_history = _caps([(2022, 208_200_000)])
        result = add_cap_percentage(contracts, cap_history)
        assert "cap_that_year" in result.columns
        assert result["cap_that_year"].iloc[0] == 208_200_000
