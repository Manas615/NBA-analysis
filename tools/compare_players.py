"""
Tool: compare_players — Compare two NBA players side-by-side.

Provides statistical comparison, impact differential,
trade value assessment, and positional analysis.
"""

from __future__ import annotations

import pandas as pd

from tools.registry import tool
from nba_analysis.data_loader import get_player_stats_with_predictions
from nba_analysis.config import PLAYER_MODEL_FEATURES


@tool(
    name="compare_players",
    description=(
        "Compare two NBA players side-by-side. Returns statistical comparison, "
        "impact differential, trade value assessment, and determines which player "
        "provides more value in various categories."
    ),
)
def compare_players(player_a: str, player_b: str) -> dict:
    """Compare two NBA players across all available metrics."""
    stats = get_player_stats_with_predictions()

    def find_player(name: str) -> pd.Series | None:
        mask = stats["full_name"].str.lower().str.contains(
            name.strip().lower(), na=False
        )
        matches = stats[mask]
        return matches.iloc[0] if len(matches) > 0 else None

    p_a = find_player(player_a)
    p_b = find_player(player_b)

    if p_a is None:
        return {"success": False, "error": f"Player '{player_a}' not found"}
    if p_b is None:
        return {"success": False, "error": f"Player '{player_b}' not found"}

    # Comparison categories
    stat_cols = {
        "points": "Points",
        "assists": "Assists",
        "reboundsTotal": "Rebounds",
        "numMinutes": "Minutes",
        "trueShootingPercentage": "True Shooting %",
        "usagePercentage": "Usage Rate",
        "netRating": "Net Rating",
        "playerImpactEstimate": "Player Impact",
        "estimatedOffensiveRating": "Offensive Rating",
        "estimatedDefensiveRating": "Defensive Rating",
    }

    comparison = {}
    advantages_a = []
    advantages_b = []

    for col, label in stat_cols.items():
        val_a = float(p_a.get(col, 0) if pd.notna(p_a.get(col, 0)) else 0)
        val_b = float(p_b.get(col, 0) if pd.notna(p_b.get(col, 0)) else 0)

        # For defensive rating, lower is better
        if col == "estimatedDefensiveRating":
            better = "A" if val_a < val_b else ("B" if val_b < val_a else "Tie")
        else:
            better = "A" if val_a > val_b else ("B" if val_b > val_a else "Tie")

        comparison[label] = {
            "player_a": round(val_a, 3),
            "player_b": round(val_b, 3),
            "difference": round(val_a - val_b, 3),
            "advantage": better,
        }

        if better == "A":
            advantages_a.append(label)
        elif better == "B":
            advantages_b.append(label)

    # Future impact comparison
    impact_a = float(p_a.get("future_impact", 0))
    impact_b = float(p_b.get("future_impact", 0))

    name_a = str(p_a.get("full_name", ""))
    name_b = str(p_b.get("full_name", ""))

    # Salary comparison
    try:
        from salary_cap import SalaryCapValidator
        validator = SalaryCapValidator()
        salary_a = validator.get_player_salary(name_a)
        salary_b = validator.get_player_salary(name_b)
        salary_comparison = {
            "player_a": validator.format_salary(salary_a),
            "player_b": validator.format_salary(salary_b),
            "difference": validator.format_salary(abs(salary_a - salary_b)),
            "better_value": "A" if impact_a / max(salary_a, 1) > impact_b / max(salary_b, 1) else "B",
        }
    except Exception:
        salary_comparison = None

    # Overall verdict
    if impact_a > impact_b * 1.1:
        verdict = f"{name_a} is the clearly better player"
    elif impact_b > impact_a * 1.1:
        verdict = f"{name_b} is the clearly better player"
    else:
        verdict = f"{name_a} and {name_b} are comparable players"

    return {
        "success": True,
        "player_a": {
            "name": name_a,
            "team": str(p_a.get("playerteamName", "")),
            "future_impact": round(impact_a, 4),
        },
        "player_b": {
            "name": name_b,
            "team": str(p_b.get("playerteamName", "")),
            "future_impact": round(impact_b, 4),
        },
        "statistical_comparison": comparison,
        "advantages_player_a": advantages_a,
        "advantages_player_b": advantages_b,
        "impact_differential": round(impact_a - impact_b, 4),
        "salary_comparison": salary_comparison,
        "verdict": verdict,
        "explanation": (
            f"{name_a} vs {name_b}: {verdict}. "
            f"Impact: {impact_a:.4f} vs {impact_b:.4f} (diff: {impact_a - impact_b:+.4f}). "
            f"{name_a} leads in {len(advantages_a)} categories, "
            f"{name_b} leads in {len(advantages_b)} categories."
        ),
        "confidence": 0.85,
    }
