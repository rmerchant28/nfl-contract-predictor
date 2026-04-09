"""Tests for scrapers/features.py — _age_at_signing and add_market_context."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from scrapers.features import _age_at_signing, add_market_context


def _contracts(**kwargs):
    base = {
        "player_name": ["Patrick Mahomes"],
        "player_name_norm": ["patrick mahomes"],
        "signing_year": [2022],
    }
    base.update(kwargs)
    return pd.DataFrame(base)


class TestAgeAtSigning:
    def test_uses_birth_year_column_when_present(self):
        """If birth_year column exists, compute age_at_signing directly."""
        contracts = _contracts(birth_year=[1995])
        result = _age_at_signing(contracts)
        assert result["age_at_signing"].iloc[0] == 27.0

    def test_birth_year_column_takes_priority_over_nflreadpy(self):
        """birth_year column present → result is correct without hitting the network."""
        contracts = _contracts(birth_year=[1990])
        # nflreadpy is imported locally in the function body, so we verify by
        # checking the return value — if the birth_year path works, age = 32.
        result = _age_at_signing(contracts)
        assert result["age_at_signing"].iloc[0] == 32.0

    def test_nflreadpy_match_computes_age(self):
        """When nflreadpy is available and player matches, compute age_at_signing."""
        contracts = _contracts()

        fake_players = pd.DataFrame({
            "display_name": ["Patrick Mahomes"],
            "birth_date": ["1995-09-17"],
        })

        mock_nfl = MagicMock()
        mock_nfl.load_players.return_value.to_pandas.return_value = fake_players

        with patch.dict("sys.modules", {"nflreadpy": mock_nfl}):
            # Re-import to pick up the patched module
            import importlib
            import scrapers.features as feat_mod
            importlib.reload(feat_mod)
            result = feat_mod._age_at_signing(contracts.copy())

        # Age = 2022 - 1995 = 27
        assert result["age_at_signing"].iloc[0] == pytest.approx(27.0)

    def test_unmatched_player_gets_nan(self):
        """Player not in nflreadpy → age_at_signing is NaN."""
        contracts = _contracts(player_name_norm=["nobody unknown"])

        fake_players = pd.DataFrame({
            "display_name": ["Patrick Mahomes"],
            "birth_date": ["1995-09-17"],
        })

        mock_nfl = MagicMock()
        mock_nfl.load_players.return_value.to_pandas.return_value = fake_players

        with patch.dict("sys.modules", {"nflreadpy": mock_nfl}):
            import importlib
            import scrapers.features as feat_mod
            importlib.reload(feat_mod)
            result = feat_mod._age_at_signing(contracts.copy())

        assert pd.isna(result["age_at_signing"].iloc[0])

    def test_nflreadpy_unavailable_gives_nan(self):
        """If nflreadpy raises ImportError, age_at_signing is NaN (no crash)."""
        contracts = _contracts()

        with patch.dict("sys.modules", {"nflreadpy": None}):
            import importlib
            import scrapers.features as feat_mod
            importlib.reload(feat_mod)
            result = feat_mod._age_at_signing(contracts.copy())

        assert "age_at_signing" in result.columns
        assert pd.isna(result["age_at_signing"].iloc[0])


def _market_df(**extra):
    """Minimal DataFrame for add_market_context tests."""
    base = {
        "position":    ["QB", "QB", "QB", "QB"],
        "signing_year": [2020,  2020,  2021,  2022],
        "apy":         [25.0,  20.0,  30.0,  28.0],
        "apy_pct_cap": [0.10,  0.08,  0.12,  0.11],
    }
    base.update(extra)
    return pd.DataFrame(base)


class TestAddMarketContext:
    def test_top_contract_pct_uses_only_prior_years(self):
        """top_contract_pct for year Y must equal the max apy_pct_cap from years < Y."""
        df = _market_df()
        result = add_market_context(df)

        # Year 2020 has no prior data → NaN
        y2020 = result[result["signing_year"] == 2020]
        assert y2020["top_contract_pct"].isna().all()

        # Year 2021 should see max from 2020 only: max(0.10, 0.08) = 0.10
        y2021 = result[result["signing_year"] == 2021]
        assert np.allclose(y2021["top_contract_pct"].values, 0.10)

        # Year 2022 should see max from 2020+2021: max(0.10, 0.08, 0.12) = 0.12
        y2022 = result[result["signing_year"] == 2022]
        assert np.allclose(y2022["top_contract_pct"].values, 0.12)

    def test_top_contract_pct_does_not_use_same_year(self):
        """A record-setting contract in year Y must NOT appear in top_contract_pct for year Y."""
        df = _market_df()
        result = add_market_context(df)
        # The year-2021 contract has apy_pct_cap=0.12 (record), but year-2021's
        # top_contract_pct should only reflect year-2020 data (0.10).
        y2021 = result[result["signing_year"] == 2021]
        assert np.allclose(y2021["top_contract_pct"].values, 0.10)

    def test_years_since_reset_uses_only_prior_years(self):
        """years_since_reset for year Y must be based on record set in a year < Y."""
        df = _market_df()
        result = add_market_context(df)

        # Year 2020: no prior data → NaN
        y2020 = result[result["signing_year"] == 2020]
        assert y2020["years_since_reset"].isna().all()

        # Year 2021: last reset was 2020 (prior max=0.10) → 2021-2020 = 1
        y2021 = result[result["signing_year"] == 2021]
        assert np.allclose(y2021["years_since_reset"].values, 1.0)

        # Year 2022: last reset was 2021 (0.12 > 0.10) → 2022-2021 = 1
        y2022 = result[result["signing_year"] == 2022]
        assert np.allclose(y2022["years_since_reset"].values, 1.0)

    def test_empty_df_returns_empty(self):
        result = add_market_context(pd.DataFrame())
        assert result.empty

    def test_missing_column_returns_unchanged(self):
        df = _market_df().drop(columns=["apy_pct_cap"])
        result = add_market_context(df)
        assert "top_contract_pct" not in result.columns

    def test_multiple_positions_are_independent(self):
        """Context features are computed per-position independently."""
        df = pd.DataFrame({
            "position":    ["QB", "QB", "WR", "WR"],
            "signing_year": [2020,  2021,  2020,  2021],
            "apy":         [25.0,  30.0,  15.0,  18.0],
            "apy_pct_cap": [0.10,  0.12,  0.06,  0.07],
        })
        result = add_market_context(df)

        # WR 2021 sees only WR 2020 data (0.06), not QB data
        wr_2021 = result[(result["position"] == "WR") & (result["signing_year"] == 2021)]
        assert np.allclose(wr_2021["top_contract_pct"].values, 0.06)
