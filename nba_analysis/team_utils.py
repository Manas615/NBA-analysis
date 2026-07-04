"""
Team roster manipulation, strength calculation, and trade execution.

Extracted from the monolithic trade_engine.py / trade_simulator.py
into reusable functions that return structured data instead of
printing to stdout.
"""

from dataclasses import dataclass, field
import pandas as pd
import numpy as np

from nba_analysis.config import MATCHUP_MODEL_FEATURES
from nba_analysis.data_loader import get_player_stats_with_predictions
from nba_analysis.models import get_matchup_model


# ── Data Classes ──────────────────────────────────────────────

@dataclass
class TeamStrength:
    """Aggregated team strength metrics."""
    net_rating: float
    offensive_rating: float
    defensive_rating: float


@dataclass
class TradeResult:
    """Structured result of a trade simulation."""
    team_a: str
    team_b: str
    player_a: str
    player_b: str
    before_prob_a: float  # Team A win prob before trade
    after_prob_a: float   # Team A win prob after trade
    net_change: float     # Change in win prob for Team A
    roster_a_after: pd.DataFrame = field(repr=False)
    roster_b_after: pd.DataFrame = field(repr=False)
    is_valid: bool = True
    error: str = ""


# ── Core Functions ────────────────────────────────────────────

def get_team_roster(
    team_name: str,
    player_stats: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Get the roster for a team.

    Parameters
    ----------
    team_name : str
        Team name (case-insensitive).
    player_stats : pd.DataFrame or None
        Pre-loaded player stats.  If None, loads from data_loader.

    Returns
    -------
    pd.DataFrame
        Player stats for the team's roster.
    """
    if player_stats is None:
        player_stats = get_player_stats_with_predictions()

    roster = player_stats[
        player_stats["playerteamName"].str.lower()
        == team_name.strip().lower()
    ].copy()

    if len(roster) == 0:
        raise ValueError(
            f"Team '{team_name}' not found. "
            f"Available teams: {sorted(player_stats['playerteamName'].dropna().unique())}"
        )

    return roster


def calculate_team_strength(
    roster: pd.DataFrame,
    top_n: int = 10,
) -> TeamStrength:
    """
    Calculate team strength from the top-N players by future impact.

    Parameters
    ----------
    roster : pd.DataFrame
        Player stats for a team (must have 'future_impact' column).
    top_n : int
        Number of top players to consider (default 10).

    Returns
    -------
    TeamStrength
        Aggregated strength metrics.
    """
    top_players = roster.sort_values(
        "future_impact", ascending=False
    ).head(top_n)

    return TeamStrength(
        net_rating=top_players["netRating"].fillna(0).mean(),
        offensive_rating=top_players["estimatedOffensiveRating"].fillna(0).mean(),
        defensive_rating=top_players["estimatedDefensiveRating"].fillna(0).mean(),
    )


def predict_win_probability(
    team_a_strength: TeamStrength,
    team_b_strength: TeamStrength,
) -> float:
    """
    Predict Team A's win probability against Team B.

    Uses the trained matchup model (LogisticRegression) on the
    differential between team strength metrics.

    Returns
    -------
    float
        Win probability for Team A as a percentage (0-100).
    """
    matchup_model = get_matchup_model()

    matchup_df = pd.DataFrame([{
        "netRating": (
            team_a_strength.net_rating
            - team_b_strength.net_rating
        ),
        "estimatedOffensiveRating": team_a_strength.offensive_rating,
        "estimatedDefensiveRating": team_a_strength.defensive_rating,
    }]).fillna(0)

    prob = matchup_model.predict_proba(matchup_df)[0][1]
    return prob * 100


def find_player(
    roster: pd.DataFrame,
    player_name: str,
) -> pd.DataFrame:
    """
    Find a player by name in a roster (case-insensitive).

    Returns
    -------
    pd.DataFrame
        Matching player row(s).

    Raises
    ------
    ValueError
        If the player is not found.
    """
    match = roster[
        roster["full_name"].str.lower()
        == player_name.strip().lower()
    ]

    if len(match) == 0:
        available = roster["full_name"].tolist()
        raise ValueError(
            f"Player '{player_name}' not found on roster. "
            f"Available: {available}"
        )

    return match


def execute_trade(
    team_a: str,
    player_a: str,
    team_b: str,
    player_b: str,
    player_stats: pd.DataFrame | None = None,
) -> TradeResult:
    """
    Simulate a trade and return structured results.

    Parameters
    ----------
    team_a, team_b : str
        Team names.
    player_a : str
        Player leaving Team A (going to Team B).
    player_b : str
        Player leaving Team B (going to Team A).
    player_stats : pd.DataFrame or None
        Pre-loaded player stats with predictions.

    Returns
    -------
    TradeResult
        Complete trade analysis including before/after probabilities.
    """
    if player_stats is None:
        player_stats = get_player_stats_with_predictions()

    try:
        roster_a = get_team_roster(team_a, player_stats)
        roster_b = get_team_roster(team_b, player_stats)
    except ValueError as e:
        return TradeResult(
            team_a=team_a, team_b=team_b,
            player_a=player_a, player_b=player_b,
            before_prob_a=0, after_prob_a=0, net_change=0,
            roster_a_after=pd.DataFrame(),
            roster_b_after=pd.DataFrame(),
            is_valid=False,
            error=str(e),
        )

    # Calculate pre-trade strength
    before_a = calculate_team_strength(roster_a)
    before_b = calculate_team_strength(roster_b)
    before_prob = predict_win_probability(before_a, before_b)

    # Find players
    try:
        p_a = find_player(roster_a, player_a)
        p_b = find_player(roster_b, player_b)
    except ValueError as e:
        return TradeResult(
            team_a=team_a, team_b=team_b,
            player_a=player_a, player_b=player_b,
            before_prob_a=before_prob, after_prob_a=before_prob,
            net_change=0,
            roster_a_after=roster_a,
            roster_b_after=roster_b,
            is_valid=False,
            error=str(e),
        )

    # Execute swap
    roster_a_new = roster_a[
        roster_a["full_name"].str.lower() != player_a.strip().lower()
    ]
    roster_b_new = roster_b[
        roster_b["full_name"].str.lower() != player_b.strip().lower()
    ]

    roster_a_new = pd.concat([roster_a_new, p_b], ignore_index=True)
    roster_b_new = pd.concat([roster_b_new, p_a], ignore_index=True)

    # Calculate post-trade strength
    after_a = calculate_team_strength(roster_a_new)
    after_b = calculate_team_strength(roster_b_new)
    after_prob = predict_win_probability(after_a, after_b)

    change = after_prob - before_prob

    return TradeResult(
        team_a=team_a,
        team_b=team_b,
        player_a=player_a,
        player_b=player_b,
        before_prob_a=before_prob,
        after_prob_a=after_prob,
        net_change=change,
        roster_a_after=roster_a_new,
        roster_b_after=roster_b_new,
    )


def swap_players_on_rosters(
    player_stats: pd.DataFrame,
    team_a: str,
    player_a: str,
    team_b: str,
    player_b: str,
) -> pd.DataFrame:
    """
    Return a modified copy of the full player_stats DataFrame
    with the two players swapped between teams.

    Useful for re-running simulations with modified rosters.
    """
    stats = player_stats.copy()

    mask_a = (
        (stats["full_name"].str.lower() == player_a.strip().lower())
        & (stats["playerteamName"].str.lower() == team_a.strip().lower())
    )
    mask_b = (
        (stats["full_name"].str.lower() == player_b.strip().lower())
        & (stats["playerteamName"].str.lower() == team_b.strip().lower())
    )

    # Swap team assignments
    stats.loc[mask_a, "playerteamName"] = team_b
    stats.loc[mask_b, "playerteamName"] = team_a

    return stats
