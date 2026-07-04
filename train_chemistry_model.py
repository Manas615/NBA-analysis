"""
Train the Lineup Chemistry Model.

Novel contribution: Roster Synergy Index (RSI)

Most trade simulators assume player contributions are additive.
This model captures non-linear interaction effects by learning
which player combinations over/under-perform their raw talent.

Training approach:
1. For each team-season, compute pairwise chemistry features
2. Target = actual_win_pct - expected_win_pct_from_talent
3. This "residual" captures chemistry effects
4. Train XGBoost: chemistry_features → performance_residual

Usage:
    python train_chemistry_model.py
"""

import os
import pandas as pd
import numpy as np
import joblib
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from nba_analysis.data_loader import load_dataset, get_multi_season_stats
from nba_analysis.config import (
    MODEL_DIR,
    CHEMISTRY_MODEL_PATH,
    CHEMISTRY_ROTATION_SIZE,
)


def compute_pairwise_features(
    player_a: pd.Series,
    player_b: pd.Series,
) -> dict:
    """Compute chemistry features between two players."""
    usage_a = float(player_a.get("usagePercentage", 0) or 0)
    usage_b = float(player_b.get("usagePercentage", 0) or 0)
    assist_a = float(player_a.get("assists", 0) or 0)
    assist_b = float(player_b.get("assists", 0) or 0)
    ts_a = float(player_a.get("trueShootingPercentage", 0) or 0)
    ts_b = float(player_b.get("trueShootingPercentage", 0) or 0)
    min_a = float(player_a.get("numMinutes", 0) or 0)
    min_b = float(player_b.get("numMinutes", 0) or 0)
    pts_a = float(player_a.get("points", 0) or 0)
    pts_b = float(player_b.get("points", 0) or 0)

    return {
        "usage_conflict": abs(usage_a - usage_b),
        "combined_usage": usage_a + usage_b,
        "assist_synergy": (
            assist_a * (1 - min(usage_b / 100, 1))
            + assist_b * (1 - min(usage_a / 100, 1))
        ),
        "shooting_spacing": ts_a + ts_b,
        "minutes_overlap": (
            min(min_a, min_b) / max(min_a, min_b)
            if max(min_a, min_b) > 0 else 0
        ),
        "scoring_balance": (
            1 - abs(pts_a - pts_b) / max(pts_a + pts_b, 1)
        ),
    }


def build_team_chemistry_features(
    team_players: pd.DataFrame,
) -> dict | None:
    """Compute aggregated chemistry features for a team rotation."""
    top = team_players.nlargest(
        min(CHEMISTRY_ROTATION_SIZE, len(team_players)),
        "numMinutes",
    )

    if len(top) < 2:
        return None

    all_pair_features = []
    for (_, a), (_, b) in combinations(top.iterrows(), 2):
        all_pair_features.append(compute_pairwise_features(a, b))

    if not all_pair_features:
        return None

    pair_df = pd.DataFrame(all_pair_features)

    result = {}
    for col in pair_df.columns:
        result[f"{col}_mean"] = pair_df[col].mean()
        result[f"{col}_std"] = pair_df[col].std()

    result["roster_size"] = len(top)
    result["usage_spread"] = top["usagePercentage"].fillna(0).std()
    result["minutes_concentration"] = (
        top["numMinutes"].fillna(0).max()
        / top["numMinutes"].fillna(1).sum()
        if top["numMinutes"].fillna(0).sum() > 0 else 0
    )

    return result


def main():
    print("Loading multi-season data...")
    multi = get_multi_season_stats()

    # Load raw data for win information
    raw = load_dataset()
    raw["season"] = raw["gameDateTimeEst"].dt.year

    # Compute actual win% per team-season
    if "win" not in raw.columns:
        print("No 'win' column found, cannot compute performance residuals")
        print("Creating a default model...")
        _save_default_model()
        return

    team_wins = (
        raw.groupby(["playerteamName", "season"])
        .agg(actual_win_pct=("win", "mean"))
        .reset_index()
    )

    # Expected win% from talent
    talent = (
        multi.groupby(["playerteamName", "season"])
        .agg(
            mean_impact=("playerImpactEstimate", "mean"),
            mean_net_rating=("netRating", "mean"),
        )
        .reset_index()
    )

    team_data = team_wins.merge(
        talent, on=["playerteamName", "season"], how="inner"
    )

    # Fit linear model for expected win%
    lr = LinearRegression()
    X_talent = team_data[["mean_impact", "mean_net_rating"]].fillna(0)
    lr.fit(X_talent, team_data["actual_win_pct"])
    team_data["expected_win_pct"] = lr.predict(X_talent)

    # Performance residual = actual - expected (chemistry effect)
    team_data["chemistry_residual"] = (
        team_data["actual_win_pct"] - team_data["expected_win_pct"]
    )

    print(f"Team-seasons with win data: {len(team_data)}")

    # Build chemistry features for each team-season
    print("Computing roster chemistry features...")
    rows = []
    for _, team_row in team_data.iterrows():
        team = team_row["playerteamName"]
        season = team_row["season"]

        team_players = multi[
            (multi["playerteamName"] == team)
            & (multi["season"] == season)
        ]

        features = build_team_chemistry_features(team_players)
        if features is not None:
            features["chemistry_residual"] = team_row["chemistry_residual"]
            rows.append(features)

    df = pd.DataFrame(rows)
    print(f"Training samples with chemistry features: {len(df)}")

    if len(df) < 20:
        print("Not enough training data. Saving default model...")
        _save_default_model()
        return

    feature_cols = [c for c in df.columns if c != "chemistry_residual"]
    X = df[feature_cols].fillna(0)
    y = df["chemistry_residual"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=3,
        random_state=42,
    )

    print("Training chemistry model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"\nModel Results")
    print(f"MAE: {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")

    print(f"\nFeature Importance:")
    for feat, imp in sorted(
        zip(feature_cols, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    )[:8]:
        print(f"  {feat}: {imp:.3f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, CHEMISTRY_MODEL_PATH)
    print(f"\nModel saved to {CHEMISTRY_MODEL_PATH}")


def _save_default_model():
    """Save a minimal default model when insufficient training data."""
    X = pd.DataFrame(
        np.zeros((10, 15)),
        columns=[
            "usage_conflict_mean", "usage_conflict_std",
            "combined_usage_mean", "combined_usage_std",
            "assist_synergy_mean", "assist_synergy_std",
            "shooting_spacing_mean", "shooting_spacing_std",
            "minutes_overlap_mean", "minutes_overlap_std",
            "scoring_balance_mean", "scoring_balance_std",
            "roster_size", "usage_spread", "minutes_concentration",
        ],
    )
    y = pd.Series(np.zeros(10))

    model = XGBRegressor(n_estimators=10, max_depth=2, random_state=42)
    model.fit(X, y)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, CHEMISTRY_MODEL_PATH)
    print(f"Default model saved to {CHEMISTRY_MODEL_PATH}")


if __name__ == "__main__":
    main()
