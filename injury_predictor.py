"""
Injury Risk Prediction and Impact Adjustment.

Predicts the probability of a player missing significant games
next season and adjusts their future_impact accordingly.

Research angle: Trade value changes based on injury risk,
making championship simulations more realistic.

Usage:
    python injury_predictor.py
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass

from nba_analysis.config import (
    GAMES_PER_SEASON,
    INJURY_SEVERITY_WEIGHT,
    HIGH_INJURY_RISK_THRESHOLD,
)
from nba_analysis.data_loader import (
    get_player_stats_with_predictions,
    get_multi_season_stats,
)
from nba_analysis.models import get_injury_model


@dataclass
class PlayerInjuryRisk:
    """Injury risk assessment for a single player."""
    player_name: str
    team: str
    predicted_games_missed: float
    injury_risk: float
    risk_level: str
    original_impact: float
    adjusted_impact: float


class InjuryPredictor:
    """Predict player injury risk and adjust future impact."""

    def __init__(self):
        self.model = get_injury_model()

    def _enrich_with_history(
        self, stats: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add historical features needed for injury prediction."""
        df = stats.copy()

        try:
            multi = get_multi_season_stats()

            career = multi.groupby("personId").agg(
                seasons_in_league=("season", "nunique"),
                hist_availability=(
                    "games_played",
                    lambda x: (
                        x.clip(upper=GAMES_PER_SEASON) / GAMES_PER_SEASON
                    ).mean(),
                ),
            ).reset_index()

            df = df.merge(career, on="personId", how="left")
            df["seasons_in_league"] = df["seasons_in_league"].fillna(3)
            df["hist_availability"] = df["hist_availability"].fillna(0.75)
        except Exception:
            df["seasons_in_league"] = 5
            df["hist_availability"] = 0.75

        return df

    def predict_injury_risk(self, player_stats: dict) -> float:
        """Predict injury risk probability for a single player."""
        feature_cols = [
            "seasons_in_league",
            "minutes_load",
            "usage_burden",
            "scoring_load",
            "availability_ratio",
            "hist_availability",
        ]

        features = pd.DataFrame([player_stats])
        for col in feature_cols:
            if col not in features.columns:
                features[col] = 0

        predicted_missed = float(
            self.model.predict(features[feature_cols])[0]
        )
        predicted_missed = max(0, min(predicted_missed, GAMES_PER_SEASON))

        risk = predicted_missed / GAMES_PER_SEASON
        return min(risk, 1.0)

    def get_risk_level(self, risk: float) -> str:
        """Categorize risk level."""
        if risk >= HIGH_INJURY_RISK_THRESHOLD:
            return "High"
        elif risk >= 0.20:
            return "Moderate"
        return "Low"

    def adjust_future_impact(
        self,
        player_stats: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Adjust future_impact by injury risk for all players.

        adjusted_impact = future_impact * (1 - injury_risk * severity_weight)
        """
        df = self._enrich_with_history(player_stats)

        feature_cols = [
            "seasons_in_league",
            "minutes_load",
            "usage_burden",
            "scoring_load",
            "availability_ratio",
            "hist_availability",
        ]

        # Build feature matrix
        feature_df = pd.DataFrame()
        feature_df["seasons_in_league"] = df.get("seasons_in_league", 5)
        feature_df["minutes_load"] = df["numMinutes"].fillna(0)
        feature_df["usage_burden"] = df["usagePercentage"].fillna(0)
        feature_df["scoring_load"] = df["points"].fillna(0)
        feature_df["availability_ratio"] = (
            df["games_played"].clip(upper=GAMES_PER_SEASON) / GAMES_PER_SEASON
            if "games_played" in df.columns
            else 0.75
        )
        feature_df["hist_availability"] = df.get("hist_availability", 0.75)

        # Predict games missed
        predicted_missed = self.model.predict(
            feature_df[feature_cols].fillna(0)
        )
        predicted_missed = np.clip(predicted_missed, 0, GAMES_PER_SEASON)

        df["injury_risk"] = (predicted_missed / GAMES_PER_SEASON).clip(0, 1)
        df["predicted_games_missed"] = predicted_missed

        # Adjust future impact
        if "future_impact" in df.columns:
            df["original_future_impact"] = df["future_impact"].copy()
            df["future_impact"] = df["future_impact"] * (
                1 - df["injury_risk"] * INJURY_SEVERITY_WEIGHT
            )

        return df

    def get_team_injury_report(
        self,
        team_name: str,
    ) -> list[PlayerInjuryRisk]:
        """Generate injury risk report for an entire team roster."""
        stats = get_player_stats_with_predictions()
        stats = self._enrich_with_history(stats)

        team = stats[
            stats["playerteamName"].str.lower()
            == team_name.strip().lower()
        ]

        if len(team) == 0:
            raise ValueError(f"Team '{team_name}' not found")

        adjusted = self.adjust_future_impact(team)

        report = []
        for _, row in adjusted.iterrows():
            risk = float(row.get("injury_risk", 0))
            report.append(
                PlayerInjuryRisk(
                    player_name=row["full_name"],
                    team=row["playerteamName"],
                    predicted_games_missed=float(
                        row.get("predicted_games_missed", 0)
                    ),
                    injury_risk=risk,
                    risk_level=self.get_risk_level(risk),
                    original_impact=float(
                        row.get(
                            "original_future_impact",
                            row.get("future_impact", 0),
                        )
                    ),
                    adjusted_impact=float(row.get("future_impact", 0)),
                )
            )

        return sorted(report, key=lambda x: x.injury_risk, reverse=True)


# ── CLI Entry Point ──────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("NBA Injury Risk Predictor")
    print("=" * 55)

    predictor = InjuryPredictor()

    team = input("\nEnter team name: ").strip()
    report = predictor.get_team_injury_report(team)

    print(f"\nInjury Risk Report: {team}")
    print("-" * 55)
    print(
        f"{'Player':<25} {'Risk':>8} {'Level':<10} {'Games Missed':>12}"
    )
    print("-" * 55)

    for player in report:
        print(
            f"{player.player_name:<25} "
            f"{player.injury_risk:>7.1%} "
            f"{player.risk_level:<10} "
            f"{player.predicted_games_missed:>11.1f}"
        )
