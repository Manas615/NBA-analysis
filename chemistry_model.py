"""
Lineup Chemistry Model — Roster Synergy Index (RSI).

Novel metric for team chemistry that captures playstyle compatibility.

Most trade simulators assume:
    Player A + Player B = additive improvement

Reality:
    Bad chemistry can reduce performance.
    Example: Ball-dominant PG + Ball-dominant SG = lower efficiency

The RSI quantifies "fit" by measuring:
1. Shooting Spacing — how well players spread the floor
2. Assist Synergy — playmaking compatibility
3. Usage Conflict — ball-dominance clashes

RSI ∈ [0, 1] where:
    < 0.3 = Poor chemistry (usage conflicts)
    0.3-0.6 = Average chemistry
    0.6-0.8 = Good chemistry (complementary styles)
    > 0.8 = Elite chemistry (rare)

Usage:
    python chemistry_model.py
"""

import pandas as pd
import numpy as np
from itertools import combinations
from dataclasses import dataclass

from nba_analysis.config import (
    CHEMISTRY_ROTATION_SIZE,
    CHEMISTRY_FLOOR,
    CHEMISTRY_CEILING,
)
from nba_analysis.data_loader import get_player_stats_with_predictions


@dataclass
class PairwiseChemistry:
    """Chemistry score between two players."""
    player_a: str
    player_b: str
    usage_conflict: float
    assist_synergy: float
    shooting_spacing: float
    minutes_overlap: float
    scoring_balance: float
    overall_score: float


@dataclass
class RosterSynergy:
    """Full roster chemistry analysis."""
    team: str
    rsi: float
    rsi_label: str
    strength_multiplier: float
    top_pair: PairwiseChemistry | None
    worst_pair: PairwiseChemistry | None
    pair_details: list[PairwiseChemistry]


class ChemistryModel:
    """
    Roster Synergy Index (RSI) — Novel chemistry metric.

    Computes pairwise chemistry scores between rotation players
    and aggregates into a single team-level metric.
    """

    def compute_pairwise_chemistry(
        self,
        player_a: pd.Series,
        player_b: pd.Series,
    ) -> PairwiseChemistry:
        """Compute chemistry score between two players."""
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

        # 1. Usage conflict (0 = no conflict, 1 = max conflict)
        max_usage = max(usage_a, usage_b)
        min_usage = min(usage_a, usage_b)
        usage_conflict = 0.0
        if max_usage > 0:
            usage_conflict = (
                min(min_usage / 30.0, 1.0)
                * min(max_usage / 30.0, 1.0)
            )

        # 2. Assist synergy (0 = no synergy, 1 = max synergy)
        assist_total = assist_a + assist_b
        assist_synergy = min(assist_total / 15.0, 1.0)

        # 3. Shooting spacing (0 = poor, 1 = elite)
        avg_ts = (ts_a + ts_b) / 2
        shooting_spacing = min(avg_ts / 0.60, 1.0) if avg_ts > 0 else 0.5

        # 4. Minutes overlap
        minutes_overlap = (
            min(min_a, min_b) / max(min_a, min_b)
            if max(min_a, min_b) > 0 else 0
        )

        # 5. Scoring balance
        total_pts = pts_a + pts_b
        scoring_balance = (
            1 - abs(pts_a - pts_b) / total_pts
            if total_pts > 0 else 0.5
        )

        # Overall chemistry score (weighted)
        overall = (
            0.25 * shooting_spacing
            + 0.30 * assist_synergy
            + 0.15 * scoring_balance
            + 0.10 * minutes_overlap
            - 0.20 * usage_conflict
        )
        overall = max(0.0, min(1.0, overall))

        return PairwiseChemistry(
            player_a=str(player_a.get("full_name", "Unknown")),
            player_b=str(player_b.get("full_name", "Unknown")),
            usage_conflict=round(usage_conflict, 3),
            assist_synergy=round(assist_synergy, 3),
            shooting_spacing=round(shooting_spacing, 3),
            minutes_overlap=round(minutes_overlap, 3),
            scoring_balance=round(scoring_balance, 3),
            overall_score=round(overall, 3),
        )

    def compute_roster_synergy_index(
        self,
        roster: pd.DataFrame,
    ) -> RosterSynergy:
        """
        Compute RSI for a team roster.

        RSI = mean pairwise chemistry across top rotation players.
        """
        team_name = (
            roster["playerteamName"].iloc[0]
            if len(roster) > 0 else "Unknown"
        )

        top = roster.nlargest(
            min(CHEMISTRY_ROTATION_SIZE, len(roster)),
            "numMinutes",
        )

        if len(top) < 2:
            return RosterSynergy(
                team=team_name, rsi=0.5, rsi_label="Average",
                strength_multiplier=0.85,
                top_pair=None, worst_pair=None, pair_details=[],
            )

        pairs = []
        for (_, a), (_, b) in combinations(top.iterrows(), 2):
            pairs.append(self.compute_pairwise_chemistry(a, b))

        if not pairs:
            return RosterSynergy(
                team=team_name, rsi=0.5, rsi_label="Average",
                strength_multiplier=0.85,
                top_pair=None, worst_pair=None, pair_details=[],
            )

        scores = [p.overall_score for p in pairs]
        rsi = float(np.mean(scores))

        if rsi >= 0.8:
            label = "Elite"
        elif rsi >= 0.6:
            label = "Good"
        elif rsi >= 0.3:
            label = "Average"
        else:
            label = "Poor"

        multiplier = (
            CHEMISTRY_FLOOR
            + (CHEMISTRY_CEILING - CHEMISTRY_FLOOR) * rsi
        )

        sorted_pairs = sorted(
            pairs, key=lambda p: p.overall_score, reverse=True
        )

        return RosterSynergy(
            team=team_name,
            rsi=round(rsi, 3),
            rsi_label=label,
            strength_multiplier=round(multiplier, 3),
            top_pair=sorted_pairs[0],
            worst_pair=sorted_pairs[-1],
            pair_details=sorted_pairs,
        )

    def chemistry_adjusted_strength(
        self,
        roster: pd.DataFrame,
    ) -> dict:
        """Calculate team strength adjusted by chemistry."""
        synergy = self.compute_roster_synergy_index(roster)
        mult = synergy.strength_multiplier

        top_players = (
            roster.sort_values("future_impact", ascending=False).head(10)
            if "future_impact" in roster.columns
            else roster.head(10)
        )

        return {
            "net_rating": float(
                top_players["netRating"].fillna(0).mean()
            ) * mult,
            "offensive_rating": float(
                top_players["estimatedOffensiveRating"].fillna(0).mean()
            ) * mult,
            "defensive_rating": float(
                top_players["estimatedDefensiveRating"].fillna(0).mean()
            ) * mult,
            "rsi": synergy.rsi,
            "rsi_label": synergy.rsi_label,
            "multiplier": mult,
        }

    def compare_trade_chemistry(
        self,
        roster_before: pd.DataFrame,
        roster_after: pd.DataFrame,
    ) -> dict:
        """Compare chemistry before and after a trade."""
        before = self.compute_roster_synergy_index(roster_before)
        after = self.compute_roster_synergy_index(roster_after)

        return {
            "rsi_before": before.rsi,
            "rsi_after": after.rsi,
            "rsi_change": round(after.rsi - before.rsi, 3),
            "label_before": before.rsi_label,
            "label_after": after.rsi_label,
            "multiplier_before": before.strength_multiplier,
            "multiplier_after": after.strength_multiplier,
        }


# ── CLI Entry Point ──────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("NBA Roster Chemistry Analysis")
    print("Roster Synergy Index (RSI)")
    print("=" * 55)

    model = ChemistryModel()
    stats = get_player_stats_with_predictions()

    team = input("\nEnter team name: ").strip()

    roster = stats[
        stats["playerteamName"].str.lower() == team.lower()
    ]

    if len(roster) == 0:
        print(f'Team "{team}" not found.')
        teams = sorted(stats["playerteamName"].dropna().unique())
        print(f"Available: {teams}")
    else:
        synergy = model.compute_roster_synergy_index(roster)

        print(f"\n{synergy.team} Roster Chemistry")
        print("-" * 55)
        print(f"RSI: {synergy.rsi:.3f} ({synergy.rsi_label})")
        print(f"Strength Multiplier: {synergy.strength_multiplier:.3f}")

        if synergy.top_pair:
            print(f"\nBest Pair:")
            p = synergy.top_pair
            print(f"  {p.player_a} + {p.player_b}: {p.overall_score:.3f}")

        if synergy.worst_pair:
            print(f"\nWorst Pair:")
            p = synergy.worst_pair
            print(f"  {p.player_a} + {p.player_b}: {p.overall_score:.3f}")

        print(f"\nAll Pair Scores:")
        for p in synergy.pair_details[:10]:
            print(
                f"  {p.player_a:<20} + {p.player_b:<20}"
                f" = {p.overall_score:.3f}"
            )
