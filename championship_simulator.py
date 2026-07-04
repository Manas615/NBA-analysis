"""
Championship Odds Simulation — Monte Carlo Season Simulator.

Simulates full 82-game NBA seasons thousands of times to estimate:
  - Expected wins per team
  - Playoff probability
  - Conference Finals probability
  - Championship probability

Research novelty: Team strength is computed using predicted future
impact (from the player model), not static ratings.  After a trade,
strengths are dynamically recalculated, making championship odds
**trade-aware** — the core contribution.

Usage:
    python championship_simulator.py
    python championship_simulator.py --team Warriors --sims 1000
"""

import argparse
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from nba_analysis.config import (
    ALL_TEAMS,
    EASTERN_CONFERENCE,
    WESTERN_CONFERENCE,
    GAMES_PER_SEASON,
    PLAYOFF_TEAMS_PER_CONF,
    HOME_COURT_ADVANTAGE,
    DEFAULT_SIMULATIONS,
    INTRA_CONFERENCE_GAMES,
    INTER_CONFERENCE_GAMES,
    get_conference,
)
from nba_analysis.data_loader import get_player_stats_with_predictions
from nba_analysis.models import get_matchup_model
from nba_analysis.team_utils import (
    get_team_roster,
    calculate_team_strength,
    swap_players_on_rosters,
    TeamStrength,
)


# ── Data Classes ──────────────────────────────────────────────

@dataclass
class TeamOdds:
    """Championship simulation results for a single team."""
    team: str
    conference: str
    expected_wins: float
    expected_losses: float
    win_std: float
    playoff_probability: float
    second_round_probability: float
    conf_finals_probability: float
    finals_probability: float
    championship_probability: float


@dataclass
class SimulationResults:
    """Full simulation results across all teams."""
    n_simulations: int
    team_odds: dict[str, TeamOdds]
    champion_distribution: dict[str, int]

    def get_team(self, team: str) -> TeamOdds:
        """Get results for a specific team."""
        key = team.strip()
        if key in self.team_odds:
            return self.team_odds[key]
        # Try case-insensitive
        for k, v in self.team_odds.items():
            if k.lower() == key.lower():
                return v
        raise KeyError(f"Team '{team}' not found in results")

    def print_team(self, team: str) -> None:
        """Pretty-print results for one team."""
        odds = self.get_team(team)
        print(f"\n{'=' * 50}")
        print(f"  {odds.team} ({odds.conference}ern Conference)")
        print(f"{'=' * 50}")
        print(f"  Expected Record:          {odds.expected_wins:.1f} - {odds.expected_losses:.1f}")
        print(f"  Win Std Dev:              ±{odds.win_std:.1f}")
        print(f"  Playoff Probability:      {odds.playoff_probability:.1%}")
        print(f"  Second Round:             {odds.second_round_probability:.1%}")
        print(f"  Conference Finals:        {odds.conf_finals_probability:.1%}")
        print(f"  NBA Finals:               {odds.finals_probability:.1%}")
        print(f"  Championship Probability: {odds.championship_probability:.1%}")
        print(f"{'=' * 50}")

    def print_rankings(self, top_n: int = 30) -> None:
        """Print championship odds leaderboard."""
        ranked = sorted(
            self.team_odds.values(),
            key=lambda t: t.championship_probability,
            reverse=True,
        )[:top_n]

        print(f"\n{'=' * 72}")
        print(f"  Championship Odds ({self.n_simulations:,} simulations)")
        print(f"{'=' * 72}")
        print(
            f"  {'Rank':<5} {'Team':<22} {'Wins':>5} "
            f"{'Playoffs':>9} {'Conf F':>8} {'Finals':>8} {'Champ':>8}"
        )
        print(f"  {'-' * 66}")

        for i, t in enumerate(ranked, 1):
            print(
                f"  {i:<5} {t.team:<22} {t.expected_wins:>5.1f} "
                f"{t.playoff_probability:>8.1%} "
                f"{t.conf_finals_probability:>7.1%} "
                f"{t.finals_probability:>7.1%} "
                f"{t.championship_probability:>7.1%}"
            )
        print(f"{'=' * 72}")


# ── Schedule Generator ────────────────────────────────────────

class ScheduleGenerator:
    """
    Generate a simplified 82-game schedule for each team.

    Uses weighted round-robin: teams play more intra-conference
    games (52) than inter-conference games (30), matching the
    real NBA scheduling pattern.
    """

    def __init__(self, teams: list[str]):
        self.teams = teams
        self.east = [t for t in teams if t in EASTERN_CONFERENCE]
        self.west = [t for t in teams if t in WESTERN_CONFERENCE]

    def generate(self, rng: np.random.Generator | None = None) -> list[tuple[str, str]]:
        """
        Generate a full season schedule as (home, away) tuples.

        Each team will play approximately 82 games total, with
        ~52 intra-conference and ~30 inter-conference.
        """
        if rng is None:
            rng = np.random.default_rng()

        games = []

        # Intra-conference: each team plays every conf opponent ~3.7 times
        # 14 opponents × 3.7 ≈ 52 games
        for conf in [self.east, self.west]:
            for i, team_a in enumerate(conf):
                for team_b in conf[i + 1:]:
                    # ~3-4 games between each pair
                    n_games = rng.choice([3, 4], p=[0.3, 0.7])
                    for g in range(n_games):
                        # Alternate home/away
                        if g % 2 == 0:
                            games.append((team_a, team_b))
                        else:
                            games.append((team_b, team_a))

        # Inter-conference: each team plays every cross-conf opponent 2 times
        # 15 opponents × 2 = 30 games
        for east_team in self.east:
            for west_team in self.west:
                n_games = 2
                for g in range(n_games):
                    if g % 2 == 0:
                        games.append((east_team, west_team))
                    else:
                        games.append((west_team, east_team))

        # Shuffle for realism
        rng.shuffle(games)
        return games


# ── Core Simulator ────────────────────────────────────────────

class ChampionshipSimulator:
    """
    Monte Carlo NBA season & playoff simulator.

    For each simulation:
      1. Generate an 82-game schedule
      2. Simulate every game using the matchup model
      3. Compute standings
      4. Simulate playoffs (best-of-7 per round)
      5. Record the champion

    After N simulations, probabilities are derived from
    frequency counts.
    """

    def __init__(
        self,
        n_simulations: int = DEFAULT_SIMULATIONS,
        player_stats: pd.DataFrame | None = None,
        seed: int | None = None,
    ):
        self.n_simulations = n_simulations
        self.rng = np.random.default_rng(seed)
        self.matchup_model = get_matchup_model()

        # Load player stats and compute team strengths
        if player_stats is None:
            player_stats = get_player_stats_with_predictions()
        self.player_stats = player_stats

        # Identify teams present in the data
        available_teams = set(
            player_stats["playerteamName"].dropna().unique()
        )
        self.teams = sorted(available_teams & set(ALL_TEAMS))

        if len(self.teams) < 2:
            raise ValueError(
                f"Need at least 2 NBA teams in data, found: {self.teams}"
            )

        # Pre-compute team strengths
        self._team_strengths: dict[str, TeamStrength] = {}
        for team in self.teams:
            try:
                roster = get_team_roster(team, self.player_stats)
                self._team_strengths[team] = calculate_team_strength(roster)
            except ValueError:
                pass  # Team not in data

        self.schedule_gen = ScheduleGenerator(self.teams)

    def _get_win_probability(
        self,
        home_team: str,
        away_team: str,
    ) -> float:
        """
        Get P(home_team wins) using the matchup model.

        Includes home-court advantage adjustment.
        """
        home_str = self._team_strengths.get(home_team)
        away_str = self._team_strengths.get(away_team)

        if home_str is None or away_str is None:
            return 0.5  # Unknown team → coin flip

        # Build matchup features
        matchup_df = pd.DataFrame([{
            "netRating": home_str.net_rating - away_str.net_rating,
            "estimatedOffensiveRating": home_str.offensive_rating,
            "estimatedDefensiveRating": home_str.defensive_rating,
        }]).fillna(0)

        prob = float(
            self.matchup_model.predict_proba(matchup_df)[0][1]
        )

        # Apply home-court advantage
        prob = min(prob + HOME_COURT_ADVANTAGE, 0.95)

        return prob

    def simulate_season(self) -> dict[str, int]:
        """
        Simulate one full season.

        Returns
        -------
        dict[str, int]
            {team_name: total_wins}
        """
        wins = Counter({team: 0 for team in self.teams})
        schedule = self.schedule_gen.generate(self.rng)

        for home, away in schedule:
            if home not in self._team_strengths or away not in self._team_strengths:
                continue

            prob = self._get_win_probability(home, away)
            if self.rng.random() < prob:
                wins[home] += 1
            else:
                wins[away] += 1

        return dict(wins)

    def _seed_conference(
        self,
        standings: dict[str, int],
        conference: list[str],
    ) -> list[str]:
        """
        Get the top 8 teams in a conference by wins, sorted descending.

        Returns ordered list of seeded teams (1-seed first).
        """
        conf_teams = [
            (team, standings.get(team, 0))
            for team in conference
            if team in self._team_strengths
        ]
        conf_teams.sort(key=lambda x: x[1], reverse=True)

        return [
            team for team, _ in conf_teams[:PLAYOFF_TEAMS_PER_CONF]
        ]

    def simulate_series(
        self,
        higher_seed: str,
        lower_seed: str,
    ) -> str:
        """
        Simulate a best-of-7 playoff series.

        Higher seed gets home-court advantage (games 1, 2, 5, 7 at home).
        """
        wins_higher = 0
        wins_lower = 0
        wins_needed = 4

        # Home-court pattern: H H A A H A H
        home_games = [True, True, False, False, True, False, True]

        for game_idx in range(7):
            if wins_higher >= wins_needed or wins_lower >= wins_needed:
                break

            is_home = home_games[game_idx]
            if is_home:
                prob = self._get_win_probability(higher_seed, lower_seed)
            else:
                prob = 1 - self._get_win_probability(lower_seed, higher_seed)

            if self.rng.random() < prob:
                wins_higher += 1
            else:
                wins_lower += 1

        return higher_seed if wins_higher >= wins_needed else lower_seed

    def simulate_playoffs(
        self,
        standings: dict[str, int],
    ) -> dict:
        """
        Simulate the full NBA playoffs.

        Bracket: 1v8, 2v7, 3v6, 4v5 in each conference,
        then conference semis, conference finals, NBA Finals.

        Returns
        -------
        dict with keys:
            'champion': str
            'finalist_east': str
            'finalist_west': str
            'conf_finalists_east': list[str]
            'conf_finalists_west': list[str]
            'second_round_east': list[str]
            'second_round_west': list[str]
            'playoff_teams_east': list[str]
            'playoff_teams_west': list[str]
        """
        east_seeds = self._seed_conference(standings, EASTERN_CONFERENCE)
        west_seeds = self._seed_conference(standings, WESTERN_CONFERENCE)

        result = {
            "playoff_teams_east": list(east_seeds),
            "playoff_teams_west": list(west_seeds),
            "second_round_east": [],
            "second_round_west": [],
            "conf_finalists_east": [],
            "conf_finalists_west": [],
            "finalist_east": "",
            "finalist_west": "",
            "champion": "",
        }

        def run_bracket(seeds: list[str], conf_key: str) -> str:
            """Run a conference bracket and return the conference champion."""
            if len(seeds) < 8:
                # Pad with available teams
                while len(seeds) < 8:
                    seeds.append(seeds[-1])

            # First round: 1v8, 2v7, 3v6, 4v5
            r1_winners = [
                self.simulate_series(seeds[0], seeds[7]),
                self.simulate_series(seeds[1], seeds[6]),
                self.simulate_series(seeds[2], seeds[5]),
                self.simulate_series(seeds[3], seeds[4]),
            ]
            result[f"second_round_{conf_key}"] = list(r1_winners)

            # Second round (Conference Semis)
            r2_winners = [
                self.simulate_series(r1_winners[0], r1_winners[3]),
                self.simulate_series(r1_winners[1], r1_winners[2]),
            ]
            result[f"conf_finalists_{conf_key}"] = list(r2_winners)

            # Conference Finals
            conf_champ = self.simulate_series(r2_winners[0], r2_winners[1])
            result[f"finalist_{conf_key}"] = conf_champ

            return conf_champ

        # Run both conference brackets
        east_champ = run_bracket(east_seeds, "east")
        west_champ = run_bracket(west_seeds, "west")

        # NBA Finals
        champion = self.simulate_series(east_champ, west_champ)
        result["champion"] = champion

        return result

    def run(self, verbose: bool = True) -> SimulationResults:
        """
        Run the full Monte Carlo simulation.

        Returns
        -------
        SimulationResults
            Complete results with per-team odds.
        """
        champion_counts = Counter()
        finals_counts = Counter()
        conf_finals_counts = Counter()
        second_round_counts = Counter()
        playoff_counts = Counter()
        win_totals: dict[str, list[int]] = defaultdict(list)

        if verbose:
            print(f"\nRunning {self.n_simulations:,} season simulations...")
            print(f"Teams: {len(self.teams)}")

        for i in range(self.n_simulations):
            if verbose and (i + 1) % max(1, self.n_simulations // 10) == 0:
                pct = (i + 1) / self.n_simulations * 100
                print(f"  Progress: {pct:.0f}%")

            # 1. Simulate regular season
            standings = self.simulate_season()
            for team, wins in standings.items():
                win_totals[team].append(wins)

            # 2. Track playoff teams
            east_seeds = self._seed_conference(standings, EASTERN_CONFERENCE)
            west_seeds = self._seed_conference(standings, WESTERN_CONFERENCE)
            for team in east_seeds + west_seeds:
                playoff_counts[team] += 1

            # 3. Simulate playoffs
            playoff_result = self.simulate_playoffs(standings)

            # Track advancement
            for team in playoff_result.get("second_round_east", []):
                second_round_counts[team] += 1
            for team in playoff_result.get("second_round_west", []):
                second_round_counts[team] += 1

            for team in playoff_result.get("conf_finalists_east", []):
                conf_finals_counts[team] += 1
            for team in playoff_result.get("conf_finalists_west", []):
                conf_finals_counts[team] += 1

            east_finalist = playoff_result.get("finalist_east", "")
            west_finalist = playoff_result.get("finalist_west", "")
            if east_finalist:
                finals_counts[east_finalist] += 1
            if west_finalist:
                finals_counts[west_finalist] += 1

            champion = playoff_result.get("champion", "")
            if champion:
                champion_counts[champion] += 1

        # Compute probabilities
        n = self.n_simulations
        team_odds = {}

        for team in self.teams:
            wins_list = win_totals.get(team, [0])

            try:
                conf = get_conference(team)
            except ValueError:
                conf = "Unknown"

            team_odds[team] = TeamOdds(
                team=team,
                conference=conf,
                expected_wins=float(np.mean(wins_list)),
                expected_losses=GAMES_PER_SEASON - float(np.mean(wins_list)),
                win_std=float(np.std(wins_list)),
                playoff_probability=playoff_counts[team] / n,
                second_round_probability=second_round_counts[team] / n,
                conf_finals_probability=conf_finals_counts[team] / n,
                finals_probability=finals_counts[team] / n,
                championship_probability=champion_counts[team] / n,
            )

        if verbose:
            print("Simulation complete!")

        return SimulationResults(
            n_simulations=n,
            team_odds=team_odds,
            champion_distribution=dict(champion_counts),
        )

    def simulate_with_trade(
        self,
        team_a: str,
        player_a: str,
        team_b: str,
        player_b: str,
        verbose: bool = True,
    ) -> dict:
        """
        Run simulation before and after a trade.

        Returns a dict comparing the two scenarios.
        """
        # Run baseline simulation
        if verbose:
            print("\n=== BASELINE (before trade) ===")
        baseline = self.run(verbose=verbose)

        # Swap players and re-create simulator
        new_stats = swap_players_on_rosters(
            self.player_stats, team_a, player_a, team_b, player_b
        )

        if verbose:
            print(f"\n=== AFTER TRADE: {player_a} ↔ {player_b} ===")
        post_trade = ChampionshipSimulator(
            n_simulations=self.n_simulations,
            player_stats=new_stats,
            seed=None,
        ).run(verbose=verbose)

        # Compare
        comparison = {}
        for team in self.teams:
            try:
                before = baseline.get_team(team)
                after = post_trade.get_team(team)
            except KeyError:
                continue

            comparison[team] = {
                "wins_before": before.expected_wins,
                "wins_after": after.expected_wins,
                "wins_change": after.expected_wins - before.expected_wins,
                "playoff_before": before.playoff_probability,
                "playoff_after": after.playoff_probability,
                "champ_before": before.championship_probability,
                "champ_after": after.championship_probability,
                "champ_change": (
                    after.championship_probability
                    - before.championship_probability
                ),
            }

        # Focus on the two trading teams
        focus = {}
        for team in [team_a, team_b]:
            for k, v in comparison.items():
                if k.lower() == team.lower():
                    focus[k] = v
                    break

        return {
            "baseline": baseline,
            "post_trade": post_trade,
            "comparison": comparison,
            "focus_teams": focus,
        }


# ── CLI Entry Point ───────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="NBA Championship Odds Simulator"
    )
    parser.add_argument(
        "--team",
        type=str,
        default=None,
        help="Show detailed results for a specific team",
    )
    parser.add_argument(
        "--sims",
        type=int,
        default=DEFAULT_SIMULATIONS,
        help=f"Number of simulations (default: {DEFAULT_SIMULATIONS})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=30,
        help="Show top N teams in rankings (default: 30)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("  NBA Championship Odds Simulator")
    print("  Monte Carlo Season Simulation")
    print("=" * 60)

    sim = ChampionshipSimulator(
        n_simulations=args.sims,
        seed=args.seed,
    )

    results = sim.run(verbose=True)

    # Print rankings
    results.print_rankings(top_n=args.top)

    # Print specific team if requested
    if args.team:
        try:
            results.print_team(args.team)
        except KeyError:
            print(f"\nTeam '{args.team}' not found.")
            print(f"Available: {sim.teams}")


if __name__ == "__main__":
    main()
