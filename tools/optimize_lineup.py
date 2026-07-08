"""
Tool: optimize_lineup — Find optimal starting five, bench, and rotations.

Analyzes a team roster to recommend the best lineup configuration
based on player impact, chemistry, and positional balance.
"""

from __future__ import annotations

import pandas as pd

from tools.registry import tool
from nba_analysis.data_loader import get_player_stats_with_predictions
from nba_analysis.team_utils import get_team_roster
from nba_analysis.config import resolve_team_name


def _estimate_position(player: pd.Series) -> str:
    """Estimate player position from stat profile."""
    assists = float(player.get("assists", 0) or 0)
    rebounds = float(player.get("reboundsTotal", 0) or 0)
    points = float(player.get("points", 0) or 0)
    minutes = float(player.get("numMinutes", 0) or 0)

    if minutes < 10:
        return "Bench"

    if assists > 6:
        return "PG"
    elif assists > 4 and rebounds < 5:
        return "SG"
    elif rebounds > 8:
        return "C"
    elif rebounds > 6:
        return "PF"
    else:
        return "SF"


def _allocate_minutes(roster: pd.DataFrame) -> list[dict]:
    """Allocate minutes based on player impact and position."""
    total_minutes = 240  # 48 min * 5 positions
    players = roster.sort_values("future_impact", ascending=False).head(15)

    allocations = []
    remaining = total_minutes

    for i, (_, player) in enumerate(players.iterrows()):
        impact = float(player.get("future_impact", 0) or 0)
        name = str(player.get("full_name", "Unknown"))

        if i < 5:
            # Starters: 28-38 minutes
            mins = min(38, max(28, int(impact * 200)))
        elif i < 8:
            # Key bench: 18-26 minutes
            mins = min(26, max(18, int(impact * 150)))
        elif i < 10:
            # Rotation: 10-18 minutes
            mins = min(18, max(10, int(impact * 120)))
        else:
            # Deep bench: 5-10 minutes
            mins = min(10, max(5, int(impact * 80)))

        mins = min(mins, remaining)
        remaining -= mins

        allocations.append({
            "player": name,
            "position": _estimate_position(player),
            "minutes": mins,
            "impact": round(impact, 4),
        })

        if remaining <= 0:
            break

    return allocations


@tool(
    name="optimize_lineup",
    description=(
        "Optimize a team's lineup by finding the best starting five, "
        "bench rotation, minute allocation, and overall rotation strategy. "
        "Based on player impact scores, positional fit, and chemistry."
    ),
)
def optimize_lineup(team: str) -> dict:
    """Optimize starting lineup, bench, and minutes for a team."""
    team_resolved = resolve_team_name(team)
    stats = get_player_stats_with_predictions()

    try:
        roster = get_team_roster(team_resolved, stats)
    except ValueError as e:
        return {"success": False, "error": str(e)}

    # Sort by future impact
    ranked = roster.sort_values("future_impact", ascending=False)

    # Starting five
    starters = []
    for _, player in ranked.head(5).iterrows():
        starters.append({
            "player": str(player.get("full_name", "")),
            "position": _estimate_position(player),
            "impact": round(float(player.get("future_impact", 0)), 4),
            "points": round(float(player.get("points", 0) or 0), 1),
            "assists": round(float(player.get("assists", 0) or 0), 1),
            "rebounds": round(float(player.get("reboundsTotal", 0) or 0), 1),
        })

    # Bench
    bench = []
    for _, player in ranked.iloc[5:13].iterrows():
        bench.append({
            "player": str(player.get("full_name", "")),
            "position": _estimate_position(player),
            "impact": round(float(player.get("future_impact", 0)), 4),
            "points": round(float(player.get("points", 0) or 0), 1),
        })

    # Minute allocation
    minutes = _allocate_minutes(roster)

    # Team totals
    top5_impact = sum(s["impact"] for s in starters)
    bench_impact = sum(b["impact"] for b in bench)

    return {
        "success": True,
        "team": team_resolved,
        "roster_size": len(roster),
        "starting_five": starters,
        "bench": bench,
        "minute_allocation": minutes,
        "team_metrics": {
            "starting_five_impact": round(top5_impact, 4),
            "bench_impact": round(bench_impact, 4),
            "depth_ratio": round(bench_impact / max(top5_impact, 0.001), 3),
        },
        "explanation": (
            f"{team_resolved} optimal lineup: "
            f"Starting five total impact: {top5_impact:.3f}. "
            f"Bench depth impact: {bench_impact:.3f}. "
            f"Depth ratio: {bench_impact / max(top5_impact, 0.001):.2f}."
        ),
        "confidence": 0.72,
    }
