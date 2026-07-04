"""
Train the Injury Prediction Model.

Predicts how many games a player will miss next season based on:
- Age proxy (seasons in league)
- Minutes per game (workload)
- Usage percentage (physical toll)
- Historical availability (games_played / 82)
- Points per game (star player workload)

Target: games_missed_next_season = 82 - games_played_next_season

Usage:
    python train_injury_model.py
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

from nba_analysis.data_loader import get_multi_season_stats
from nba_analysis.config import INJURY_MODEL_PATH, MODEL_DIR, GAMES_PER_SEASON


def build_injury_features(multi_season: pd.DataFrame) -> pd.DataFrame:
    """Build injury prediction features from multi-season data."""
    df = multi_season.copy()

    # Availability ratio: games_played / expected 82
    df["availability_ratio"] = (
        df["games_played"].clip(upper=GAMES_PER_SEASON) / GAMES_PER_SEASON
    )

    # Career length as age proxy
    df["seasons_in_league"] = df.groupby("personId").cumcount() + 1

    # Historical average availability (expanding mean of past seasons)
    df["hist_availability"] = (
        df.groupby("personId")["availability_ratio"]
        .transform(lambda x: x.expanding().mean().shift(1))
    )

    # Feature columns
    df["minutes_load"] = df["numMinutes"].fillna(0)
    df["usage_burden"] = df["usagePercentage"].fillna(0)
    df["scoring_load"] = df["points"].fillna(0)

    # Target: games missed NEXT season
    df["next_games_played"] = (
        df.groupby("personId")["games_played"].shift(-1)
    )
    df["games_missed_next"] = (
        GAMES_PER_SEASON
        - df["next_games_played"].clip(upper=GAMES_PER_SEASON)
    )

    # Drop rows without target or historical data
    df = df.dropna(subset=["games_missed_next", "hist_availability"])

    return df


def main():
    print("Loading multi-season data...")
    multi_season = get_multi_season_stats()
    print(f"Total player-seasons: {len(multi_season)}")

    print("Building injury features...")
    df = build_injury_features(multi_season)
    print(f"Training samples: {len(df)}")

    features = [
        "seasons_in_league",
        "minutes_load",
        "usage_burden",
        "scoring_load",
        "availability_ratio",
        "hist_availability",
    ]

    X = df[features].fillna(0)
    y = df["games_missed_next"].clip(lower=0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=5,
        random_state=42,
    )

    print("Training injury prediction model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"\nModel Results")
    print(f"MAE: {mae:.2f} games")
    print(f"R2 Score: {r2:.4f}")

    print(f"\nFeature Importance:")
    for feat, imp in sorted(
        zip(features, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    ):
        print(f"  {feat}: {imp:.3f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, INJURY_MODEL_PATH)
    print(f"\nModel saved to {INJURY_MODEL_PATH}")


if __name__ == "__main__":
    main()
