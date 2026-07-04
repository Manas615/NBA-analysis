"""
NBA Trade Engine — CLI Interface.

Simulates a player-for-player trade and shows the impact on
win probability.  Now backed by the nba_analysis package.

Usage:
    python trade_engine.py
"""

from nba_analysis.data_loader import get_player_stats_with_predictions
from nba_analysis.team_utils import execute_trade


def main():
    print("\n" + "=" * 50)
    print("  NBA Trade Engine")
    print("=" * 50)

    # Load data
    print("\nLoading player data...")
    player_stats = get_player_stats_with_predictions()
    print("Data loaded.\n")

    # Show available teams
    teams = sorted(
        player_stats["playerteamName"].dropna().unique()
    )
    print("Available Teams:")
    for t in teams:
        print(f"  - {t}")

    # Get trade details
    print("\nEnter Trade Details")
    team_a = input("Team A: ").strip()
    player_a = input("Player from Team A: ").strip()
    team_b = input("Team B: ").strip()
    player_b = input("Player from Team B: ").strip()

    # Execute trade
    result = execute_trade(
        team_a, player_a,
        team_b, player_b,
        player_stats=player_stats,
    )

    if not result.is_valid:
        print(f"\n❌ Trade Error: {result.error}")
        return

    # Display results
    print("\n" + "=" * 50)
    print("  TRADE RESULT")
    print("=" * 50)
    print(f"\n  {result.player_a} ({result.team_a})")
    print(f"    ↕")
    print(f"  {result.player_b} ({result.team_b})")
    print(f"\n  Before Trade:")
    print(f"    {result.team_a} Win Probability: {result.before_prob_a:.2f}%")
    print(f"\n  After Trade:")
    print(f"    {result.team_a} Win Probability: {result.after_prob_a:.2f}%")
    print(f"\n  Net Change: {result.net_change:+.2f}%")
    print("=" * 50)


if __name__ == "__main__":
    main()
