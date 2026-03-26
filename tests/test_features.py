"""Tests for scrapers/features.py — _age_at_signing."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from scrapers.features import _age_at_signing


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
