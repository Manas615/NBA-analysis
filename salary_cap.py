"""
Salary Cap Validation Module.

Validates NBA trades against the CBA salary matching rules.
For teams over the salary cap, incoming salary cannot exceed
125% of outgoing salary.

Generates synthetic salary estimates from player stats when
real salary data is not available.

Usage:
    python salary_cap.py
"""

import os
import pandas as pd
import numpy as np
from dataclasses import dataclass

from nba_analysis.config import (
    SALARY_CAP_MATCH_FACTOR,
    SALARY_CAP_AMOUNT,
    LUXURY_TAX_THRESHOLD,
    SALARY_CAP_CSV,
    DATA_DIR,
)
from nba_analysis.data_loader import get_player_stats_with_predictions


@dataclass
class TradeValidation:
    """Result of salary cap trade validation."""
    is_valid: bool
    team_a_outgoing_salary: float
    team_b_outgoing_salary: float
    team_a_max_incoming: float
    team_b_max_incoming: float
    team_a_incoming_salary: float
    team_b_incoming_salary: float
    reason: str = ""
    details: str = ""


class SalaryCapValidator:
    """Validate trades against NBA salary cap rules."""

    def __init__(self):
        self.salary_data = self._load_salaries()

    def _load_salaries(self) -> pd.DataFrame:
        """Load salary data from CSV, or generate synthetic estimates."""
        if os.path.exists(SALARY_CAP_CSV):
            return pd.read_csv(SALARY_CAP_CSV)

        # Generate realistic salary estimates from player stats
        print("Generating salary estimates from player statistics...")
        stats = get_player_stats_with_predictions()

        rng = np.random.default_rng(42)
        salaries = []

        for _, player in stats.iterrows():
            impact = max(float(player.get("future_impact", 0) or 0), 0)
            minutes = max(float(player.get("numMinutes", 0) or 0), 0)
            points = max(float(player.get("points", 0) or 0), 0)

            # Base salary scale: min ~$1M, max ~$50M
            impact_factor = min(impact / 0.20, 1.0)
            minutes_factor = min(minutes / 36.0, 1.0)
            points_factor = min(points / 30.0, 1.0)

            estimated_salary = (
                1_000_000
                + impact_factor * 35_000_000
                + minutes_factor * 8_000_000
                + points_factor * 6_000_000
            )
            estimated_salary *= rng.uniform(0.85, 1.15)
            estimated_salary = int(round(estimated_salary, -4))

            salaries.append({
                "player_name": player["full_name"],
                "team": player["playerteamName"],
                "salary": estimated_salary,
                "season": "2024-25",
            })

        df = pd.DataFrame(salaries)

        os.makedirs(DATA_DIR, exist_ok=True)
        df.to_csv(SALARY_CAP_CSV, index=False)
        print(f"Salary data cached to {SALARY_CAP_CSV}")

        return df

    def get_player_salary(self, player_name: str) -> float:
        """Look up a player's current salary."""
        match = self.salary_data[
            self.salary_data["player_name"].str.lower()
            == player_name.strip().lower()
        ]
        if len(match) == 0:
            # Try partial match
            try:
                match = self.salary_data[
                    self.salary_data["player_name"].str.lower().str.contains(
                        player_name.strip().lower(), na=False
                    )
                ]
            except Exception:
                pass
        if len(match) == 0:
            return 0.0
        return float(match.iloc[0]["salary"])

    def format_salary(self, amount: float) -> str:
        """Format salary as human-readable string."""
        if amount >= 1_000_000:
            return f"${amount / 1_000_000:.1f}M"
        elif amount >= 1_000:
            return f"${amount / 1_000:.0f}K"
        return f"${amount:,.0f}"

    def validate_trade(
        self,
        team_a_players: list[str],
        team_b_players: list[str],
    ) -> TradeValidation:
        """
        Validate a trade against NBA salary matching rules.

        NBA CBA Rule: For teams over the salary cap, incoming
        salary cannot exceed 125% of outgoing salary.
        """
        team_a_outgoing = sum(
            self.get_player_salary(p) for p in team_a_players
        )
        team_b_outgoing = sum(
            self.get_player_salary(p) for p in team_b_players
        )

        team_a_max_incoming = team_a_outgoing * SALARY_CAP_MATCH_FACTOR
        team_b_max_incoming = team_b_outgoing * SALARY_CAP_MATCH_FACTOR

        # Team A receives Team B's players
        team_a_incoming = team_b_outgoing
        # Team B receives Team A's players
        team_b_incoming = team_a_outgoing

        is_valid = True
        reasons = []

        if team_a_incoming > team_a_max_incoming and team_a_outgoing > 0:
            is_valid = False
            reasons.append(
                f"Team A receiving {self.format_salary(team_a_incoming)} "
                f"but can only take {self.format_salary(team_a_max_incoming)} "
                f"(125% of {self.format_salary(team_a_outgoing)} outgoing)"
            )

        if team_b_incoming > team_b_max_incoming and team_b_outgoing > 0:
            is_valid = False
            reasons.append(
                f"Team B receiving {self.format_salary(team_b_incoming)} "
                f"but can only take {self.format_salary(team_b_max_incoming)} "
                f"(125% of {self.format_salary(team_b_outgoing)} outgoing)"
            )

        if team_a_outgoing == 0 and team_b_outgoing == 0:
            is_valid = False
            reasons.append("No player salaries found for either side")

        details_lines = [
            f"Team A sends: {', '.join(team_a_players)}",
            f"  Total outgoing: {self.format_salary(team_a_outgoing)}",
            f"  Max incoming (125%): {self.format_salary(team_a_max_incoming)}",
            f"  Actual incoming: {self.format_salary(team_a_incoming)}",
            "",
            f"Team B sends: {', '.join(team_b_players)}",
            f"  Total outgoing: {self.format_salary(team_b_outgoing)}",
            f"  Max incoming (125%): {self.format_salary(team_b_max_incoming)}",
            f"  Actual incoming: {self.format_salary(team_b_incoming)}",
        ]

        return TradeValidation(
            is_valid=is_valid,
            team_a_outgoing_salary=team_a_outgoing,
            team_b_outgoing_salary=team_b_outgoing,
            team_a_max_incoming=team_a_max_incoming,
            team_b_max_incoming=team_b_max_incoming,
            team_a_incoming_salary=team_a_incoming,
            team_b_incoming_salary=team_b_incoming,
            reason=" | ".join(reasons) if reasons else "Trade is salary-cap compliant",
            details="\n".join(details_lines),
        )

    def suggest_salary_filler(
        self,
        team_name: str,
        target_salary: float,
        exclude_players: list[str] | None = None,
    ) -> list[dict]:
        """Suggest players to include as salary filler."""
        if exclude_players is None:
            exclude_players = []

        exclude_lower = [p.lower() for p in exclude_players]

        team_players = self.salary_data[
            (self.salary_data["team"].str.lower() == team_name.strip().lower())
            & (~self.salary_data["player_name"].str.lower().isin(exclude_lower))
        ].sort_values("salary", ascending=False)

        suggestions = []
        for _, player in team_players.iterrows():
            if player["salary"] <= target_salary * 1.1:
                suggestions.append({
                    "player_name": player["player_name"],
                    "salary": player["salary"],
                    "salary_formatted": self.format_salary(player["salary"]),
                    "gap_remaining": target_salary - player["salary"],
                })

        return suggestions[:5]


# ── CLI Entry Point ──────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("NBA Salary Cap Trade Validator")
    print("=" * 50)

    validator = SalaryCapValidator()

    print("\nEnter trade details:")
    player_a = input("Player from Team A: ").strip()
    player_b = input("Player from Team B: ").strip()

    salary_a = validator.get_player_salary(player_a)
    salary_b = validator.get_player_salary(player_b)

    print(f"\n{player_a}: {validator.format_salary(salary_a)}")
    print(f"{player_b}: {validator.format_salary(salary_b)}")

    result = validator.validate_trade([player_a], [player_b])

    print(f"\n{'=' * 50}")
    print(f"Trade Valid: {'✅ YES' if result.is_valid else '❌ NO'}")
    print(f"Reason: {result.reason}")
    print(f"\nDetails:")
    print(result.details)
