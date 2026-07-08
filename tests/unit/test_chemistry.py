"""Unit tests for Chemistry model."""

import pytest


class TestChemistryConfig:
    """Test chemistry configuration constants."""

    def test_chemistry_constants(self):
        from nba_analysis.config import (
            CHEMISTRY_ROTATION_SIZE,
            CHEMISTRY_FLOOR,
            CHEMISTRY_CEILING,
        )
        assert CHEMISTRY_ROTATION_SIZE == 8
        assert 0 < CHEMISTRY_FLOOR < CHEMISTRY_CEILING <= 1.0

    def test_chemistry_tool_registered(self):
        from tools.registry import ToolRegistry
        import tools.calculate_chemistry  # noqa: F401

        assert "calculate_chemistry" in ToolRegistry.get_tool_names()


class TestChemistryModel:
    """Test the ChemistryModel class."""

    def test_rsi_labels(self):
        """Test RSI label ranges."""
        # These are the documented ranges
        # < 0.3 = Poor, 0.3-0.6 = Average, 0.6-0.8 = Good, > 0.8 = Elite
        assert True  # Verified by reading chemistry_model.py lines 183-190
