"""
Central configuration for the NBA Trade Analysis System.

Contains all constants, team assignments, feature definitions,
and tunable parameters used across the system.
"""

# ── Dataset ────────────────────────────────────────────────────
KAGGLE_DATASET = "eoinamoore/historical-nba-data-and-player-box-scores"
DATASET_CSV = "PlayerStatisticsExtended.csv"

# ── Feature Definitions ───────────────────────────────────────
NUMERIC_COLS = [
    "points",
    "assists",
    "reboundsTotal",
    "turnovers",
    "numMinutes",
    "trueShootingPercentage",
    "usagePercentage",
    "netRating",
    "playerImpactEstimate",
    "estimatedOffensiveRating",
    "estimatedDefensiveRating",
]

PLAYER_MODEL_FEATURES = [
    "points",
    "assists",
    "reboundsTotal",
    "turnovers",
    "numMinutes",
    "trueShootingPercentage",
    "usagePercentage",
    "netRating",
    "playerImpactEstimate",
]

MATCHUP_MODEL_FEATURES = [
    "netRating",
    "estimatedOffensiveRating",
    "estimatedDefensiveRating",
]

# ── NBA Team Assignments ──────────────────────────────────────
EASTERN_CONFERENCE = [
    "76ers",
    "Bucks",
    "Bulls",
    "Cavaliers",
    "Celtics",
    "Hawks",
    "Heat",
    "Hornets",
    "Knicks",
    "Magic",
    "Nets",
    "Pacers",
    "Pistons",
    "Raptors",
    "Wizards",
]

WESTERN_CONFERENCE = [
    "Clippers",
    "Grizzlies",
    "Jazz",
    "Kings",
    "Lakers",
    "Mavericks",
    "Nuggets",
    "Pelicans",
    "Rockets",
    "Spurs",
    "Suns",
    "Thunder",
    "Timberwolves",
    "Trail Blazers",
    "Warriors",
]

ALL_TEAMS = sorted(EASTERN_CONFERENCE + WESTERN_CONFERENCE)

# Map team name variants to canonical names
TEAM_NAME_ALIASES = {
    "sixers": "76ers",
    "philadelphia": "76ers",
    "philly": "76ers",
    "milwaukee": "Bucks",
    "chicago": "Bulls",
    "cleveland": "Cavaliers",
    "cavs": "Cavaliers",
    "boston": "Celtics",
    "atlanta": "Hawks",
    "miami": "Heat",
    "charlotte": "Hornets",
    "new york": "Knicks",
    "ny": "Knicks",
    "orlando": "Magic",
    "brooklyn": "Nets",
    "indiana": "Pacers",
    "detroit": "Pistons",
    "toronto": "Raptors",
    "washington": "Wizards",
    "la clippers": "Clippers",
    "memphis": "Grizzlies",
    "utah": "Jazz",
    "sacramento": "Kings",
    "la lakers": "Lakers",
    "los angeles lakers": "Lakers",
    "los angeles clippers": "Clippers",
    "dallas": "Mavericks",
    "mavs": "Mavericks",
    "denver": "Nuggets",
    "new orleans": "Pelicans",
    "houston": "Rockets",
    "san antonio": "Spurs",
    "phoenix": "Suns",
    "okc": "Thunder",
    "oklahoma city": "Thunder",
    "minnesota": "Timberwolves",
    "wolves": "Timberwolves",
    "portland": "Trail Blazers",
    "blazers": "Trail Blazers",
    "golden state": "Warriors",
    "gsw": "Warriors",
    "dubs": "Warriors",
}

# ── Season & Playoff Constants ────────────────────────────────
GAMES_PER_SEASON = 82
PLAYOFF_TEAMS_PER_CONF = 8
PLAYOFF_SERIES_LENGTH = 7
HOME_COURT_ADVANTAGE = 0.03  # +3% win probability for home team

# Intra-conference games: 52, Inter-conference games: 30
INTRA_CONFERENCE_GAMES = 52
INTER_CONFERENCE_GAMES = 30

# ── Salary Cap Constants ──────────────────────────────────────
SALARY_CAP_MATCH_FACTOR = 1.25  # 125% rule
SALARY_CAP_AMOUNT = 140_588_000  # 2024-25 salary cap
LUXURY_TAX_THRESHOLD = 170_814_000  # 2024-25 luxury tax

# ── Chemistry Model Constants ─────────────────────────────────
CHEMISTRY_ROTATION_SIZE = 8  # Top 8 players by minutes for RSI
CHEMISTRY_FLOOR = 0.70  # Minimum multiplier (worst chemistry)
CHEMISTRY_CEILING = 1.00  # Maximum multiplier (perfect chemistry)

# ── Injury Model Constants ────────────────────────────────────
INJURY_SEVERITY_WEIGHT = 0.70  # Discount factor for injury risk
HIGH_INJURY_RISK_THRESHOLD = 0.40  # Above this = high risk

# ── Simulation Defaults ──────────────────────────────────────
DEFAULT_SIMULATIONS = 1000
FAST_SIMULATIONS = 100  # For quick estimates

# ── Model Paths ───────────────────────────────────────────────
MODEL_DIR = "models"
PLAYER_MODEL_PATH = "models/player_model.pkl"
MATCHUP_MODEL_PATH = "models/matchup_model.pkl"
CHEMISTRY_MODEL_PATH = "models/chemistry_model.pkl"
INJURY_MODEL_PATH = "models/injury_model.pkl"

# ── Data Paths ────────────────────────────────────────────────
DATA_DIR = "data"
SALARY_CAP_CSV = "data/salary_cap.csv"


def get_conference(team_name: str) -> str:
    """Return 'East' or 'West' for a given team name."""
    if team_name in EASTERN_CONFERENCE:
        return "East"
    elif team_name in WESTERN_CONFERENCE:
        return "West"
    else:
        raise ValueError(f"Unknown team: {team_name}")


def resolve_team_name(name: str) -> str:
    """Resolve a team name alias to its canonical name."""
    lower = name.strip().lower()
    if lower in TEAM_NAME_ALIASES:
        return TEAM_NAME_ALIASES[lower]
    # Check if it's already a canonical name (case-insensitive)
    for team in ALL_TEAMS:
        if team.lower() == lower:
            return team
    return name  # Return as-is if no match
