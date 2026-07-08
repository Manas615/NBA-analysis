"""Unit tests for Salary Cap validation."""

import pytest


class TestSalaryCapValidation:
    """Tests for the salary cap tool and validation logic."""

    def test_salary_validation_schema(self):
        """Test that salary_validation tool is registered."""
        from tools.registry import ToolRegistry
        import tools.salary_validation  # noqa: F401

        assert "salary_validation" in ToolRegistry.get_tool_names()

    def test_salary_format(self):
        """Test salary formatting."""
        from salary_cap import SalaryCapValidator

        validator = SalaryCapValidator()
        assert validator.format_salary(1_000_000) == "$1.0M"
        assert validator.format_salary(50_000_000) == "$50.0M"
        assert validator.format_salary(500_000) == "$500K"
        assert validator.format_salary(100) == "$100"


class TestSalaryCapConstants:
    """Test salary cap configuration constants."""

    def test_cap_values(self):
        from nba_analysis.config import SALARY_CAP_AMOUNT, LUXURY_TAX_THRESHOLD
        assert SALARY_CAP_AMOUNT == 140_588_000
        assert LUXURY_TAX_THRESHOLD == 170_814_000

    def test_match_factor(self):
        from nba_analysis.config import SALARY_CAP_MATCH_FACTOR
        assert SALARY_CAP_MATCH_FACTOR == 1.25
