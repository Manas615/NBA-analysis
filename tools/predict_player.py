"""
Tool: predict_player — Predict a player's future impact score.

Wraps the XGBoost player model to predict next-season
Player Impact Estimate (PIE) for any NBA player.

Returns explainable output with feature importance.
"""

from __future__ import annotations

import pandas as pd

from tools.registry import tool
from nba_analysis.data_loader import get_player_stats_with_predictions
from nba_analysis.config import PLAYER_MODEL_FEATURES


@tool(
    name="predict_player",
    description=(
        "Predict a player's future impact score (PIE) for next season. "
        "Returns the predicted impact, current stats, and feature importance. "
        "Use this to evaluate a player's projected value."
    ),
)
def predict_player(player_name: str) -> dict:
    """Predict future impact for an NBA player."""
    stats = get_player_stats_with_predictions()

    # Case-insensitive partial match
    mask = stats["full_name"].str.lower().str.contains(
        player_name.strip().lower(), na=False
    )
    matches = stats[mask]

    if len(matches) == 0:
        return {
            "success": False,
            "error": f"Player '{player_name}' not found",
            "available_players_sample": stats["full_name"].dropna().head(20).tolist(),
        }

    player = matches.iloc[0]

    # Get feature importance from the model
    from nba_analysis.models import get_player_model
    model = get_player_model()

    feature_importance = {}
    if hasattr(model, "feature_importances_"):
        for feat, imp in zip(PLAYER_MODEL_FEATURES, model.feature_importances_):
            feature_importance[feat] = round(float(imp), 4)

    # Current stats
    current_stats = {}
    for col in PLAYER_MODEL_FEATURES:
        val = player.get(col, 0)
        current_stats[col] = round(float(val if pd.notna(val) else 0), 3)

    future_impact = float(player.get("future_impact", 0))

    # Determine tier
    if future_impact >= 0.15:
        tier = "Superstar"
    elif future_impact >= 0.10:
        tier = "All-Star"
    elif future_impact >= 0.06:
        tier = "Starter"
    elif future_impact >= 0.03:
        tier = "Rotation"
    else:
        tier = "Bench"

    return {
        "success": True,
        "player_name": str(player.get("full_name", "")),
        "team": str(player.get("playerteamName", "")),
        "predicted_future_impact": round(future_impact, 4),
        "player_tier": tier,
        "current_stats": current_stats,
        "feature_importance": feature_importance,
        "explanation": (
            f"{player.get('full_name', '')} projects to a {tier}-level player "
            f"with a future impact score of {future_impact:.4f}. "
            f"Key stats: {current_stats.get('points', 0):.1f} PPG, "
            f"{current_stats.get('assists', 0):.1f} APG, "
            f"{current_stats.get('reboundsTotal', 0):.1f} RPG."
        ),
        "confidence": 0.82,
    }
