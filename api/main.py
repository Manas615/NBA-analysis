"""
NBA Trade Analysis — LLM Natural Language Interface.

FastAPI application that accepts natural language trade queries
and routes them through the full analysis pipeline:

  LLM Parse → Entity Extraction → Salary Validation →
  Chemistry Analysis → Injury Adjustment → Championship Simulation →
  JSON Response

Usage:
    uvicorn api.main:app --reload
    # or
    python -m api.main

Endpoints:
    POST /simulate_trade    — Natural language trade query
    POST /trade             — Structured trade request
    GET  /team/{name}/odds  — Championship odds for a team
    GET  /team/{name}/roster — Roster with injury risks
    GET  /player/{name}     — Player profile
    GET  /teams             — List all teams
"""

import os
import sys
import json
import re
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add parent dir to path so nba_analysis package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nba_analysis.config import (
    ALL_TEAMS,
    TEAM_NAME_ALIASES,
    resolve_team_name,
    FAST_SIMULATIONS,
)
from nba_analysis.data_loader import (
    get_player_stats_with_predictions,
    get_all_teams,
)
from nba_analysis.team_utils import (
    execute_trade,
    get_team_roster,
    calculate_team_strength,
)


# ── Pydantic Models ──────────────────────────────────────────

class NLQueryRequest(BaseModel):
    """Natural language trade query."""
    query: str = Field(
        ...,
        description="Natural language trade query",
        examples=["What if Lakers trade LeBron for Giannis?"],
    )
    simulations: int = Field(
        default=FAST_SIMULATIONS,
        description="Number of championship simulations",
        ge=10,
        le=5000,
    )


class StructuredTradeRequest(BaseModel):
    """Structured trade request."""
    team_a: str
    player_a: str
    team_b: str
    player_b: str
    simulations: int = Field(default=FAST_SIMULATIONS, ge=10, le=5000)


class TradeResponse(BaseModel):
    """Full trade analysis response."""
    trade_valid: bool
    salary_valid: bool | None = None
    salary_details: str | None = None
    win_change: str
    championship_odds_a: str | None = None
    championship_odds_b: str | None = None
    chemistry_before: float | None = None
    chemistry_after: float | None = None
    chemistry_change: float | None = None
    injury_risk_summary: dict | None = None
    details: dict


# ── Trade Query Parser (LLM or Rule-Based) ───────────────────

class TradeQueryParser:
    """
    Parse natural language trade queries into structured parameters.

    Uses LLM (Google Gemini) when available, falls back to
    rule-based extraction.
    """

    def __init__(self):
        self.llm_available = False
        self.llm_model = None
        self._try_init_llm()
        self._load_player_names()

    def _try_init_llm(self):
        """Try to initialize the Gemini LLM."""
        try:
            from dotenv import load_dotenv
            # Load from project root .env
            env_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                ".env"
            )
            load_dotenv(env_path)

            api_key = os.getenv("LLM_API_KEY", "")
            if not api_key or api_key == "your_key_here":
                print("LLM: No API key found, using rule-based parser")
                return

            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model_name = os.getenv("LLM_MODEL", "gemini-2.0-flash")
            self.llm_model = genai.GenerativeModel(model_name)
            self.llm_available = True
            print(f"LLM: Initialized {model_name}")

        except ImportError:
            print("LLM: google-generativeai not installed, using rule-based parser")
        except Exception as e:
            print(f"LLM: Init failed ({e}), using rule-based parser")

    def _load_player_names(self):
        """Load player names for entity matching."""
        try:
            stats = get_player_stats_with_predictions()
            self.player_names = stats["full_name"].dropna().unique().tolist()
            self.team_names = stats["playerteamName"].dropna().unique().tolist()
            # Build player→team lookup
            self.player_team_map = {}
            for _, row in stats.iterrows():
                name = row.get("full_name", "")
                team = row.get("playerteamName", "")
                if name and team:
                    self.player_team_map[name.lower()] = team
        except Exception:
            self.player_names = []
            self.team_names = []
            self.player_team_map = {}

    def _parse_with_llm(self, query: str) -> dict | None:
        """Use LLM to extract trade entities."""
        if not self.llm_available or not self.llm_model:
            return None

        prompt = f"""Extract the trade details from this NBA trade query.
Return ONLY valid JSON with these fields:
- team_a: the first team mentioned
- player_a: player from team_a being traded
- team_b: the second team mentioned
- player_b: player from team_b being traded

Valid teams: {self.team_names[:30]}

If you cannot extract all 4 fields, set the missing ones to null.

Query: "{query}"

Respond ONLY with the JSON object, no explanation:"""

        try:
            response = self.llm_model.generate_content(prompt)
            text = response.text.strip()

            # Clean up markdown code blocks
            if text.startswith("```"):
                text = re.sub(r"```\w*\n?", "", text).strip()

            return json.loads(text)
        except Exception as e:
            print(f"LLM parse error: {e}")
            return None

    def _parse_with_rules(self, query: str) -> dict:
        """Rule-based entity extraction fallback."""
        query_lower = query.lower()
        result = {
            "team_a": None,
            "player_a": None,
            "team_b": None,
            "player_b": None,
        }

        # Find team names in query
        found_teams = []
        for team in self.team_names:
            if team.lower() in query_lower:
                found_teams.append(team)

        # Also check aliases
        for alias, canonical in TEAM_NAME_ALIASES.items():
            if alias in query_lower and canonical not in found_teams:
                found_teams.append(canonical)

        if len(found_teams) >= 2:
            result["team_a"] = found_teams[0]
            result["team_b"] = found_teams[1]

        # Find player names in query
        found_players = []
        for name in self.player_names:
            # Check full name
            if name.lower() in query_lower:
                found_players.append(name)
                continue
            # Check last name only (for common references)
            parts = name.split()
            if len(parts) >= 2:
                last = parts[-1].lower()
                if len(last) > 3 and last in query_lower:
                    # Verify it's a word boundary
                    pattern = r'\b' + re.escape(last) + r'\b'
                    if re.search(pattern, query_lower):
                        found_players.append(name)

        # Remove duplicates, keep order
        seen = set()
        unique_players = []
        for p in found_players:
            if p.lower() not in seen:
                seen.add(p.lower())
                unique_players.append(p)

        if len(unique_players) >= 2:
            result["player_a"] = unique_players[0]
            result["player_b"] = unique_players[1]

            # Auto-assign teams from player→team map
            if not result["team_a"]:
                t = self.player_team_map.get(unique_players[0].lower())
                if t:
                    result["team_a"] = t
            if not result["team_b"]:
                t = self.player_team_map.get(unique_players[1].lower())
                if t:
                    result["team_b"] = t

        elif len(unique_players) == 1:
            result["player_a"] = unique_players[0]
            if not result["team_a"]:
                t = self.player_team_map.get(unique_players[0].lower())
                if t:
                    result["team_a"] = t

        return result

    def parse(self, query: str) -> dict:
        """Parse a natural language trade query."""
        # Try LLM first
        llm_result = self._parse_with_llm(query)
        if llm_result and all(llm_result.get(k) for k in ["team_a", "player_a", "team_b", "player_b"]):
            # Resolve team names
            for key in ["team_a", "team_b"]:
                if llm_result.get(key):
                    llm_result[key] = resolve_team_name(llm_result[key])
            return llm_result

        # Fall back to rules
        return self._parse_with_rules(query)


# ── Application Setup ─────────────────────────────────────────

# Global instances (initialized at startup)
_parser: TradeQueryParser | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize expensive resources at startup."""
    global _parser
    print("Initializing NBA Trade Analysis API...")
    print("Loading player data...")
    get_player_stats_with_predictions()
    print("Initializing query parser...")
    _parser = TradeQueryParser()
    print("API ready!")
    yield
    print("Shutting down API...")


app = FastAPI(
    title="NBA Trade Analysis API",
    description=(
        "AI-driven NBA trade simulator with chemistry-aware "
        "roster optimization and championship forecasting."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helper Functions ──────────────────────────────────────────

def _run_full_analysis(
    team_a: str,
    player_a: str,
    team_b: str,
    player_b: str,
    simulations: int = FAST_SIMULATIONS,
) -> dict:
    """Run the complete trade analysis pipeline."""
    result = {
        "trade_valid": False,
        "salary_valid": None,
        "salary_details": None,
        "win_change": "N/A",
        "championship_odds_a": None,
        "championship_odds_b": None,
        "chemistry_before": None,
        "chemistry_after": None,
        "chemistry_change": None,
        "injury_risk_summary": None,
        "details": {},
    }

    # Step 1: Execute base trade simulation
    trade = execute_trade(team_a, player_a, team_b, player_b)
    if not trade.is_valid:
        result["details"] = {"error": trade.error}
        return result

    result["trade_valid"] = True
    result["win_change"] = f"{trade.net_change:+.1f}%"
    result["details"]["before_prob"] = round(trade.before_prob_a, 2)
    result["details"]["after_prob"] = round(trade.after_prob_a, 2)
    result["details"]["team_a"] = team_a
    result["details"]["team_b"] = team_b
    result["details"]["player_a"] = player_a
    result["details"]["player_b"] = player_b

    # Step 2: Salary cap validation (if module available)
    try:
        from salary_cap import SalaryCapValidator
        validator = SalaryCapValidator()
        validation = validator.validate_trade([player_a], [player_b])
        result["salary_valid"] = validation.is_valid
        result["salary_details"] = validation.reason
        result["details"]["salary"] = {
            "team_a_outgoing": validation.team_a_outgoing_salary,
            "team_b_outgoing": validation.team_b_outgoing_salary,
            "is_valid": validation.is_valid,
        }
    except Exception as e:
        result["salary_details"] = f"Salary validation unavailable: {e}"

    # Step 3: Chemistry analysis (if module available)
    try:
        from chemistry_model import ChemistryModel
        chem = ChemistryModel()
        stats = get_player_stats_with_predictions()

        roster_a_before = get_team_roster(team_a, stats)
        synergy_before = chem.compute_roster_synergy_index(roster_a_before)
        synergy_after = chem.compute_roster_synergy_index(trade.roster_a_after)

        result["chemistry_before"] = synergy_before.rsi
        result["chemistry_after"] = synergy_after.rsi
        result["chemistry_change"] = round(
            synergy_after.rsi - synergy_before.rsi, 3
        )
        result["details"]["chemistry"] = {
            "rsi_before": synergy_before.rsi,
            "rsi_after": synergy_after.rsi,
            "label_before": synergy_before.rsi_label,
            "label_after": synergy_after.rsi_label,
        }
    except Exception as e:
        result["details"]["chemistry_error"] = str(e)

    # Step 4: Injury risk summary (if module available)
    try:
        from injury_predictor import InjuryPredictor
        ip = InjuryPredictor()
        report_a = ip.get_team_injury_report(team_a)
        high_risk = [
            {"player": p.player_name, "risk": round(p.injury_risk, 3)}
            for p in report_a[:3]
        ]
        result["injury_risk_summary"] = {
            "team": team_a,
            "high_risk_players": high_risk,
        }
    except Exception as e:
        result["details"]["injury_error"] = str(e)

    # Step 5: Championship simulation (if available)
    try:
        from championship_simulator import ChampionshipSimulator
        sim = ChampionshipSimulator(
            n_simulations=simulations,
        )
        sim_results = sim.simulate_with_trade(
            team_a, player_a, team_b, player_b,
            verbose=False,
        )
        focus = sim_results.get("focus_teams", {})
        for team, data in focus.items():
            if team.lower() == team_a.lower():
                result["championship_odds_a"] = f"{data['champ_after']:.1%}"
                result["details"]["championship_a"] = data
            elif team.lower() == team_b.lower():
                result["championship_odds_b"] = f"{data['champ_after']:.1%}"
                result["details"]["championship_b"] = data
    except Exception as e:
        result["details"]["championship_error"] = str(e)

    return result


# ── Endpoints ─────────────────────────────────────────────────

@app.get("/")
async def root():
    """API health check."""
    return {
        "service": "NBA Trade Analysis API",
        "version": "2.0.0",
        "status": "operational",
        "endpoints": [
            "POST /simulate_trade",
            "POST /trade",
            "GET /team/{name}/odds",
            "GET /team/{name}/roster",
            "GET /player/{name}",
            "GET /teams",
        ],
    }


@app.get("/teams")
async def list_teams():
    """List all available teams."""
    try:
        teams = get_all_teams()
        return {"teams": teams, "count": len(teams)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/simulate_trade")
async def simulate_trade_nl(request: NLQueryRequest):
    """
    Natural language trade simulation.

    Accepts queries like:
      "What if Lakers trade LeBron for Giannis?"
      "Trade Steph Curry to the Celtics for Jayson Tatum"
    """
    global _parser
    if _parser is None:
        _parser = TradeQueryParser()

    # Parse the query
    parsed = _parser.parse(request.query)

    # Validate extraction
    missing = [
        k for k in ["team_a", "player_a", "team_b", "player_b"]
        if not parsed.get(k)
    ]
    if missing:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Could not extract all trade details from query",
                "missing_fields": missing,
                "parsed": parsed,
                "hint": (
                    "Try a more specific query like: "
                    "'What if Lakers trade LeBron James for Bucks Giannis Antetokounmpo?'"
                ),
            },
        )

    # Run analysis
    result = _run_full_analysis(
        team_a=parsed["team_a"],
        player_a=parsed["player_a"],
        team_b=parsed["team_b"],
        player_b=parsed["player_b"],
        simulations=request.simulations,
    )

    result["parsed_query"] = parsed
    return result


@app.post("/trade")
async def simulate_trade_structured(request: StructuredTradeRequest):
    """Structured trade simulation with explicit parameters."""
    team_a = resolve_team_name(request.team_a)
    team_b = resolve_team_name(request.team_b)

    result = _run_full_analysis(
        team_a=team_a,
        player_a=request.player_a,
        team_b=team_b,
        player_b=request.player_b,
        simulations=request.simulations,
    )

    return result


@app.get("/team/{team_name}/odds")
async def get_team_odds(team_name: str, simulations: int = FAST_SIMULATIONS):
    """Get championship odds for a specific team."""
    team = resolve_team_name(team_name)

    try:
        from championship_simulator import ChampionshipSimulator
        sim = ChampionshipSimulator(n_simulations=simulations)
        results = sim.run(verbose=False)
        odds = results.get_team(team)

        return {
            "team": odds.team,
            "conference": odds.conference,
            "expected_wins": round(odds.expected_wins, 1),
            "expected_losses": round(odds.expected_losses, 1),
            "playoff_probability": f"{odds.playoff_probability:.1%}",
            "conf_finals_probability": f"{odds.conf_finals_probability:.1%}",
            "finals_probability": f"{odds.finals_probability:.1%}",
            "championship_probability": f"{odds.championship_probability:.1%}",
            "simulations": simulations,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/team/{team_name}/roster")
async def get_team_roster_info(team_name: str):
    """Get team roster with player stats and injury risks."""
    team = resolve_team_name(team_name)

    try:
        stats = get_player_stats_with_predictions()
        roster = get_team_roster(team, stats)

        players = []
        for _, row in roster.sort_values("future_impact", ascending=False).iterrows():
            player_info = {
                "name": row.get("full_name", ""),
                "points": round(row.get("points", 0), 1),
                "assists": round(row.get("assists", 0), 1),
                "rebounds": round(row.get("reboundsTotal", 0), 1),
                "minutes": round(row.get("numMinutes", 0), 1),
                "future_impact": round(row.get("future_impact", 0), 4),
            }
            players.append(player_info)

        # Try to add chemistry
        chemistry_info = None
        try:
            from chemistry_model import ChemistryModel
            chem = ChemistryModel()
            synergy = chem.compute_roster_synergy_index(roster)
            chemistry_info = {
                "rsi": synergy.rsi,
                "label": synergy.rsi_label,
                "multiplier": synergy.strength_multiplier,
            }
        except Exception:
            pass

        return {
            "team": team,
            "roster_size": len(players),
            "players": players,
            "chemistry": chemistry_info,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/player/{player_name}")
async def get_player_profile(player_name: str):
    """Get player profile with stats, impact, and injury risk."""
    try:
        stats = get_player_stats_with_predictions()

        # Find player (case-insensitive, partial match)
        match = stats[
            stats["full_name"].str.lower().str.contains(
                player_name.strip().lower()
            )
        ]

        if len(match) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"Player '{player_name}' not found",
            )

        player = match.iloc[0]

        profile = {
            "name": player.get("full_name", ""),
            "team": player.get("playerteamName", ""),
            "stats": {
                "points": round(player.get("points", 0), 1),
                "assists": round(player.get("assists", 0), 1),
                "rebounds": round(player.get("reboundsTotal", 0), 1),
                "minutes": round(player.get("numMinutes", 0), 1),
                "true_shooting": round(player.get("trueShootingPercentage", 0), 3),
                "usage_rate": round(player.get("usagePercentage", 0), 1),
                "net_rating": round(player.get("netRating", 0), 1),
                "pie": round(player.get("playerImpactEstimate", 0), 4),
            },
            "future_impact": round(player.get("future_impact", 0), 4),
        }

        # Try to add injury risk
        try:
            from injury_predictor import InjuryPredictor
            ip = InjuryPredictor()
            risk = ip.predict_injury_risk({
                "seasons_in_league": 5,
                "minutes_load": player.get("numMinutes", 0),
                "usage_burden": player.get("usagePercentage", 0),
                "scoring_load": player.get("points", 0),
                "availability_ratio": 0.8,
                "hist_availability": 0.8,
            })
            profile["injury_risk"] = round(risk, 3)
            profile["injury_level"] = ip.get_risk_level(risk)
        except Exception:
            pass

        # Try salary
        try:
            from salary_cap import SalaryCapValidator
            v = SalaryCapValidator()
            salary = v.get_player_salary(player.get("full_name", ""))
            profile["salary"] = v.format_salary(salary)
        except Exception:
            pass

        return profile
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Main ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
