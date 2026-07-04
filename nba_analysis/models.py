"""
Lazy model loading utilities with caching.

Each model is loaded at most once and cached for the lifetime
of the process.  Missing models raise clear error messages.
"""

import os
import joblib

from nba_analysis.config import (
    PLAYER_MODEL_PATH,
    MATCHUP_MODEL_PATH,
    CHEMISTRY_MODEL_PATH,
    INJURY_MODEL_PATH,
)

# ── Module-level cache ────────────────────────────────────────
_model_cache: dict[str, object] = {}


def _load_model(path: str, name: str) -> object:
    """Load a model from disk with caching."""
    if path in _model_cache:
        return _model_cache[path]

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{name} not found at '{path}'. "
            f"Run the corresponding training script first."
        )

    model = joblib.load(path)
    _model_cache[path] = model
    return model


def get_player_model():
    """
    Load the player future-impact prediction model.

    Model: XGBRegressor
    Input features: points, assists, reboundsTotal, turnovers,
                    numMinutes, trueShootingPercentage,
                    usagePercentage, netRating, playerImpactEstimate
    Output: predicted next-season Player Impact Estimate (PIE)
    """
    return _load_model(PLAYER_MODEL_PATH, "Player model")


def get_matchup_model():
    """
    Load the matchup win-probability model.

    Model: Pipeline(StandardScaler + LogisticRegression)
    Input features: netRating (differential),
                    estimatedOffensiveRating,
                    estimatedDefensiveRating
    Output: predict_proba()[0][1] → P(team_a wins)
    """
    return _load_model(MATCHUP_MODEL_PATH, "Matchup model")


def get_chemistry_model():
    """
    Load the lineup chemistry model.

    Model: XGBRegressor (trained by train_chemistry_model.py)
    Input: pairwise chemistry features
    Output: performance residual (actual - expected)
    """
    return _load_model(CHEMISTRY_MODEL_PATH, "Chemistry model")


def get_injury_model():
    """
    Load the injury prediction model.

    Model: XGBRegressor (trained by train_injury_model.py)
    Input: age proxy, minutes, usage, historical availability
    Output: predicted games missed next season
    """
    return _load_model(INJURY_MODEL_PATH, "Injury model")


def clear_model_cache() -> None:
    """Clear all cached models."""
    _model_cache.clear()
