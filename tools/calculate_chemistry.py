"""
Tool: calculate_chemistry — Compute lineup chemistry and Roster Synergy Index.

Wraps the ChemistryModel to evaluate how well players fit together.
Returns spacing, offensive/defensive fit, usage conflicts, and RSI score.
"""

from __future__ import annotations

from tools.registry import tool
from nba_analysis.data_loader import get_player_stats_with_predictions
from nba_analysis.team_utils import get_team_roster
from nba_analysis.config import resolve_team_name


@tool(
    name="calculate_chemistry",
    description=(
        "Calculate lineup chemistry and Roster Synergy Index (RSI) for a team. "
        "Evaluates spacing, offensive fit, defensive fit, usage conflicts, "
        "playmaking compatibility, and returns an overall chemistry score. "
        "Can compare chemistry before and after a trade."
    ),
)
def calculate_chemistry(
    team: str,
    trade_player_out: str = "",
    trade_player_in: str = "",
    trade_team_in: str = "",
) -> dict:
    """Calculate lineup chemistry for a team, optionally with a trade scenario."""
    from chemistry_model import ChemistryModel

    team_resolved = resolve_team_name(team)
    stats = get_player_stats_with_predictions()
    chem = ChemistryModel()

    try:
        roster = get_team_roster(team_resolved, stats)
    except ValueError as e:
        return {"success": False, "error": str(e)}

    synergy_before = chem.compute_roster_synergy_index(roster)

    result = {
        "success": True,
        "team": team_resolved,
        "rsi": synergy_before.rsi,
        "rsi_label": synergy_before.rsi_label,
        "strength_multiplier": synergy_before.strength_multiplier,
        "best_pair": None,
        "worst_pair": None,
        "pair_count": len(synergy_before.pair_details),
    }

    if synergy_before.top_pair:
        result["best_pair"] = {
            "players": f"{synergy_before.top_pair.player_a} + {synergy_before.top_pair.player_b}",
            "score": synergy_before.top_pair.overall_score,
            "usage_conflict": synergy_before.top_pair.usage_conflict,
            "assist_synergy": synergy_before.top_pair.assist_synergy,
            "shooting_spacing": synergy_before.top_pair.shooting_spacing,
        }

    if synergy_before.worst_pair:
        result["worst_pair"] = {
            "players": f"{synergy_before.worst_pair.player_a} + {synergy_before.worst_pair.player_b}",
            "score": synergy_before.worst_pair.overall_score,
            "usage_conflict": synergy_before.worst_pair.usage_conflict,
            "assist_synergy": synergy_before.worst_pair.assist_synergy,
            "shooting_spacing": synergy_before.worst_pair.shooting_spacing,
        }

    # Trade comparison
    if trade_player_out and trade_player_in and trade_team_in:
        import pandas as pd
        trade_team_resolved = resolve_team_name(trade_team_in)

        try:
            incoming_roster = get_team_roster(trade_team_resolved, stats)
            incoming_player = incoming_roster[
                incoming_roster["full_name"].str.lower().str.contains(
                    trade_player_in.strip().lower(), na=False
                )
            ]

            if len(incoming_player) == 0:
                result["trade_error"] = f"Player '{trade_player_in}' not found on {trade_team_resolved}"
            else:
                roster_after = roster[
                    ~roster["full_name"].str.lower().str.contains(
                        trade_player_out.strip().lower(), na=False
                    )
                ]
                roster_after = pd.concat(
                    [roster_after, incoming_player.head(1)],
                    ignore_index=True,
                )

                synergy_after = chem.compute_roster_synergy_index(roster_after)

                result["trade_comparison"] = {
                    "rsi_before": synergy_before.rsi,
                    "rsi_after": synergy_after.rsi,
                    "rsi_change": round(synergy_after.rsi - synergy_before.rsi, 3),
                    "label_before": synergy_before.rsi_label,
                    "label_after": synergy_after.rsi_label,
                    "multiplier_before": synergy_before.strength_multiplier,
                    "multiplier_after": synergy_after.strength_multiplier,
                }
        except Exception as e:
            result["trade_error"] = str(e)

    # Explanation
    result["explanation"] = (
        f"{team_resolved} has {synergy_before.rsi_label} chemistry (RSI: {synergy_before.rsi:.3f}). "
        f"Strength multiplier: {synergy_before.strength_multiplier:.3f}. "
        + (
            f"Best pair: {synergy_before.top_pair.player_a} + {synergy_before.top_pair.player_b} "
            f"({synergy_before.top_pair.overall_score:.3f}). "
            if synergy_before.top_pair else ""
        )
        + (
            f"Worst pair: {synergy_before.worst_pair.player_a} + {synergy_before.worst_pair.player_b} "
            f"({synergy_before.worst_pair.overall_score:.3f})."
            if synergy_before.worst_pair else ""
        )
    )
    result["confidence"] = 0.75

    return result
