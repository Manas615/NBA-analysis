"""
Tool: recommend_draft_pick — Recommend draft picks based on team needs.

Analyzes current roster gaps and recommends prospects based on
projected ceiling, player comparisons, and positional needs.
"""

from __future__ import annotations

from tools.registry import tool
from nba_analysis.data_loader import get_player_stats_with_predictions
from nba_analysis.team_utils import get_team_roster
from nba_analysis.config import resolve_team_name


# Simulated draft prospect data (2025 class)
DRAFT_PROSPECTS = [
    {
        "name": "Cooper Flagg",
        "school": "Duke",
        "position": "SF/PF",
        "projected_pick": 1,
        "ceiling": "Superstar — two-way franchise player",
        "comparison": "Jayson Tatum / Paul George",
        "projected_impact": 0.14,
        "strengths": ["Two-way versatility", "Shot creation", "Defensive motor"],
        "weaknesses": ["Three-point consistency", "Frame durability"],
    },
    {
        "name": "Dylan Harper",
        "school": "Rutgers",
        "position": "SG/PG",
        "projected_pick": 2,
        "ceiling": "All-Star — elite scorer and playmaker",
        "comparison": "Dwyane Wade / Jalen Brunson",
        "projected_impact": 0.12,
        "strengths": ["Scoring instinct", "Playmaking", "Mid-range game"],
        "weaknesses": ["Defensive effort", "Shot selection"],
    },
    {
        "name": "Ace Bailey",
        "school": "Rutgers",
        "position": "SF/SG",
        "projected_pick": 3,
        "ceiling": "All-Star — versatile wing scorer",
        "comparison": "Brandon Ingram / Tracy McGrady",
        "projected_impact": 0.11,
        "strengths": ["Length", "Scoring versatility", "Transition game"],
        "weaknesses": ["Physicality", "Defensive consistency"],
    },
    {
        "name": "VJ Edgecombe",
        "school": "Baylor",
        "position": "SG",
        "projected_pick": 4,
        "ceiling": "All-Star — explosive two-way guard",
        "comparison": "Jalen Green / Derrick Rose",
        "projected_impact": 0.10,
        "strengths": ["Athleticism", "Finishing", "Defensive upside"],
        "weaknesses": ["Jump shot", "Decision making"],
    },
    {
        "name": "Kon Knueppel",
        "school": "Duke",
        "position": "SG/SF",
        "projected_pick": 5,
        "ceiling": "Starter — elite shooter and connector",
        "comparison": "Klay Thompson / Khris Middleton",
        "projected_impact": 0.09,
        "strengths": ["Three-point shooting", "Basketball IQ", "Off-ball movement"],
        "weaknesses": ["Lateral quickness", "Shot creation"],
    },
    {
        "name": "Kasparas Jakucionis",
        "school": "Illinois",
        "position": "PG",
        "projected_pick": 6,
        "ceiling": "Starter — crafty floor general",
        "comparison": "Ricky Rubio / Luka Doncic (lite)",
        "projected_impact": 0.08,
        "strengths": ["Court vision", "Pick-and-roll mastery", "Size for position"],
        "weaknesses": ["Athleticism", "Finishing at rim"],
    },
    {
        "name": "Liam McNeeley",
        "school": "UConn",
        "position": "SF",
        "projected_pick": 7,
        "ceiling": "Starter — 3-and-D wing",
        "comparison": "Mikal Bridges / Otto Porter Jr.",
        "projected_impact": 0.07,
        "strengths": ["Three-point shooting", "Defensive versatility", "Length"],
        "weaknesses": ["Ball handling", "Self creation"],
    },
    {
        "name": "Tre Johnson",
        "school": "Texas",
        "position": "SG",
        "projected_pick": 8,
        "ceiling": "All-Star — bucket-getting scorer",
        "comparison": "Bradley Beal / Devin Booker",
        "projected_impact": 0.10,
        "strengths": ["Shot making", "Scoring instinct", "Free throw drawing"],
        "weaknesses": ["Defensive engagement", "Playmaking"],
    },
    {
        "name": "Jeremiah Fears",
        "school": "Oklahoma",
        "position": "PG/SG",
        "projected_pick": 9,
        "ceiling": "Starter — explosive combo guard",
        "comparison": "De'Aaron Fox / Ja Morant (lite)",
        "projected_impact": 0.08,
        "strengths": ["Speed", "Penetration", "Clutch scoring"],
        "weaknesses": ["Turnover prone", "Jump shot consistency"],
    },
    {
        "name": "Nolan Traore",
        "school": "International (France)",
        "position": "PG",
        "projected_pick": 10,
        "ceiling": "Starter — athletic playmaker",
        "comparison": "Dennis Schröder / Elfrid Payton",
        "projected_impact": 0.07,
        "strengths": ["Athleticism", "Transition game", "Passing"],
        "weaknesses": ["Jump shot", "Half-court offense"],
    },
]


@tool(
    name="recommend_draft_pick",
    description=(
        "Recommend the best draft pick for a team based on draft position, "
        "team needs, and prospect analysis. Returns the recommended prospect, "
        "their expected ceiling, player comparison, and fit analysis."
    ),
)
def recommend_draft_pick(
    team: str,
    draft_position: int = 1,
) -> dict:
    """Recommend draft pick based on team needs and draft position."""
    team_resolved = resolve_team_name(team)
    stats = get_player_stats_with_predictions()

    try:
        roster = get_team_roster(team_resolved, stats)
    except ValueError as e:
        return {"success": False, "error": str(e)}

    # Analyze team needs by position
    position_counts: dict[str, int] = {"PG": 0, "SG": 0, "SF": 0, "PF": 0, "C": 0}
    position_impact: dict[str, float] = {"PG": 0, "SG": 0, "SF": 0, "PF": 0, "C": 0}

    for _, player in roster.iterrows():
        assists = float(player.get("assists", 0) or 0)
        rebounds = float(player.get("reboundsTotal", 0) or 0)
        impact = float(player.get("future_impact", 0) or 0)

        if assists > 6:
            pos = "PG"
        elif assists > 4 and rebounds < 5:
            pos = "SG"
        elif rebounds > 8:
            pos = "C"
        elif rebounds > 6:
            pos = "PF"
        else:
            pos = "SF"

        position_counts[pos] += 1
        position_impact[pos] += impact

    # Identify weakest positions
    needs = sorted(position_impact.items(), key=lambda x: x[1])
    top_needs = [n[0] for n in needs[:2]]

    # Available prospects at this draft position
    available = [
        p for p in DRAFT_PROSPECTS
        if p["projected_pick"] >= draft_position
    ]

    if not available:
        available = DRAFT_PROSPECTS[-3:]

    # Score prospects by team fit
    recommendations = []
    for prospect in available[:5]:
        prospect_positions = prospect["position"].split("/")
        need_match = any(p in top_needs for p in prospect_positions)

        fit_score = prospect["projected_impact"]
        if need_match:
            fit_score *= 1.3  # Boost for positional need

        recommendations.append({
            "prospect": prospect["name"],
            "school": prospect["school"],
            "position": prospect["position"],
            "projected_pick": prospect["projected_pick"],
            "ceiling": prospect["ceiling"],
            "player_comparison": prospect["comparison"],
            "projected_impact": round(prospect["projected_impact"], 4),
            "fit_score": round(fit_score, 4),
            "fills_need": need_match,
            "strengths": prospect["strengths"],
            "weaknesses": prospect["weaknesses"],
        })

    recommendations.sort(key=lambda x: x["fit_score"], reverse=True)

    best = recommendations[0] if recommendations else None

    return {
        "success": True,
        "team": team_resolved,
        "draft_position": draft_position,
        "team_needs": top_needs,
        "position_analysis": {
            "counts": position_counts,
            "impact_by_position": {
                k: round(v, 4) for k, v in position_impact.items()
            },
        },
        "recommended_pick": best,
        "all_recommendations": recommendations,
        "explanation": (
            f"For {team_resolved} at pick #{draft_position}: "
            f"Top needs are {', '.join(top_needs)}. "
            f"Recommendation: {best['prospect']} ({best['position']}) — "
            f"{best['ceiling']}. Comp: {best['player_comparison']}."
            if best else f"No prospects available at pick #{draft_position}."
        ),
        "confidence": 0.60,
    }
