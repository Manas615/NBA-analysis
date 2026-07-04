"""
NBA Trade Simulator — Matchup Simulation CLI.

Simulates 1000 games between two teams to estimate win probability.
Now backed by the nba_analysis package.

Usage:
    python trade_simulator.py
"""

import numpy as np

from nba_analysis.data_loader import get_player_stats_with_predictions
from nba_analysis.team_utils import (
    get_team_roster,
    calculate_team_strength,
    predict_win_probability,
)


def simulate_matchup(
    team_a: str,
    team_b: str,
    simulations: int = 1000,
) -> None:
    """Simulate N games between two teams."""
    player_stats = get_player_stats_with_predictions()

    roster_a = get_team_roster(team_a, player_stats)
    roster_b = get_team_roster(team_b, player_stats)

    strength_a = calculate_team_strength(roster_a)
    strength_b = calculate_team_strength(roster_b)

    win_prob = predict_win_probability(strength_a, strength_b)
    prob = win_prob / 100.0  # Convert to 0-1

    results = np.random.binomial(1, prob, simulations)
    team_a_pct = np.mean(results) * 100

    print("\n" + "=" * 50)
    print("  Simulation Results")
    print("=" * 50)
    print(f"\n  {team_a} vs {team_b}")
    print(f"  ({simulations:,} simulations)")
    print(f"\n  {team_a} Win Probability: {team_a_pct:.2f}%")
    print(f"  {team_b} Win Probability: {100 - team_a_pct:.2f}%")
    print("=" * 50)


def main():
    print("\n" + "=" * 50)
    print("  NBA Matchup Simulator")
    print("=" * 50)

    print("\nLoading data...")
    player_stats = get_player_stats_with_predictions()

    teams = sorted(
        player_stats["playerteamName"].dropna().unique()
    )
    print("\nAvailable Teams:")
    for t in teams:
        print(f"  - {t}")

    team_a = input("\nEnter Team A: ").strip()
    team_b = input("Enter Team B: ").strip()

    simulate_matchup(team_a, team_b)


if __name__ == "__main__":
    main()
