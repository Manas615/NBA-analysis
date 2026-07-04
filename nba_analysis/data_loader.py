"""
Shared data loading with caching.

Replaces the duplicated kagglehub + pd.read_csv blocks that were
copy-pasted across trade_engine.py, trade_simulator.py, and the
training scripts.  The dataset is loaded once and cached in memory.
"""

import os
import pandas as pd
import numpy as np
import kagglehub

from nba_analysis.config import (
    KAGGLE_DATASET,
    DATASET_CSV,
    NUMERIC_COLS,
    PLAYER_MODEL_FEATURES,
)

# ── Module-level cache ────────────────────────────────────────
_dataset_cache: pd.DataFrame | None = None
_season_stats_cache: dict[int | None, pd.DataFrame] = {}
_player_predictions_cache: pd.DataFrame | None = None


def load_dataset(force_reload: bool = False) -> pd.DataFrame:
    """
    Load the full PlayerStatisticsExtended dataset from Kaggle.

    Results are cached in memory so subsequent calls return instantly.

    Returns
    -------
    pd.DataFrame
        The full dataset with numeric columns coerced and
        gameDateTimeEst parsed as datetime.
    """
    global _dataset_cache

    if _dataset_cache is not None and not force_reload:
        return _dataset_cache

    path = kagglehub.dataset_download(KAGGLE_DATASET)
    csv_file = os.path.join(path, DATASET_CSV)

    print("Loading dataset...")
    df = pd.read_csv(csv_file, low_memory=False)

    # Parse date column
    df["gameDateTimeEst"] = pd.to_datetime(
        df["gameDateTimeEst"], errors="coerce"
    )

    # Coerce numeric columns
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Also coerce 'win' if present
    if "win" in df.columns:
        df["win"] = pd.to_numeric(df["win"], errors="coerce")

    _dataset_cache = df
    return df


def get_latest_season_year(df: pd.DataFrame | None = None) -> int:
    """Return the most recent season year in the dataset."""
    if df is None:
        df = load_dataset()
    return int(df["gameDateTimeEst"].dt.year.max())


def get_season_stats(
    season: int | None = None,
    force_reload: bool = False,
) -> pd.DataFrame:
    """
    Compute per-player season averages for a given season.

    Parameters
    ----------
    season : int or None
        The calendar year of the season.  If None, uses the latest
        season available in the dataset.
    force_reload : bool
        Force re-computation even if cached.

    Returns
    -------
    pd.DataFrame
        One row per player with columns:
        personId, firstName, lastName, playerteamName,
        full_name, and averages for all numeric stat columns.
    """
    if season in _season_stats_cache and not force_reload:
        return _season_stats_cache[season].copy()

    df = load_dataset()

    if season is None:
        season = get_latest_season_year(df)

    season_df = df[df["gameDateTimeEst"].dt.year == season].copy()

    if len(season_df) == 0:
        raise ValueError(
            f"No data found for season year {season}"
        )

    # Aggregate per player
    agg_dict = {
        col: "mean"
        for col in NUMERIC_COLS
        if col in season_df.columns
    }

    # Also count games played
    agg_dict["gameDateTimeEst"] = "count"

    player_stats = (
        season_df.groupby(
            ["personId", "firstName", "lastName", "playerteamName"],
            as_index=False,
        )
        .agg(agg_dict)
        .rename(columns={"gameDateTimeEst": "games_played"})
    )

    # Add full name
    player_stats["full_name"] = (
        player_stats["firstName"] + " " + player_stats["lastName"]
    )

    _season_stats_cache[season] = player_stats
    return player_stats.copy()


def get_all_teams(season: int | None = None) -> list[str]:
    """Return sorted list of all team names in the dataset for a season."""
    stats = get_season_stats(season)
    return sorted(stats["playerteamName"].dropna().unique().tolist())


def get_player_stats_with_predictions(
    season: int | None = None,
    force_reload: bool = False,
) -> pd.DataFrame:
    """
    Load season stats and append 'future_impact' predictions
    from the player model.

    Returns
    -------
    pd.DataFrame
        Season stats with an additional 'future_impact' column.
    """
    global _player_predictions_cache

    if _player_predictions_cache is not None and not force_reload:
        return _player_predictions_cache.copy()

    from nba_analysis.models import get_player_model

    stats = get_season_stats(season, force_reload=force_reload)
    model = get_player_model()

    X = stats[PLAYER_MODEL_FEATURES].fillna(0)
    stats["future_impact"] = model.predict(X)

    _player_predictions_cache = stats
    return stats.copy()


def get_multi_season_stats() -> pd.DataFrame:
    """
    Compute per-player per-season averages across ALL seasons.

    Used by training scripts for injury and chemistry models.

    Returns
    -------
    pd.DataFrame
        One row per (personId, season) with stat averages
        and a 'games_played' count.
    """
    df = load_dataset()
    df["season"] = df["gameDateTimeEst"].dt.year

    agg_dict = {
        col: "mean"
        for col in NUMERIC_COLS
        if col in df.columns
    }
    agg_dict["gameDateTimeEst"] = "count"

    multi = (
        df.groupby(
            ["personId", "firstName", "lastName", "playerteamName", "season"],
            as_index=False,
        )
        .agg(agg_dict)
        .rename(columns={"gameDateTimeEst": "games_played"})
    )

    multi["full_name"] = (
        multi["firstName"] + " " + multi["lastName"]
    )

    return multi.sort_values(["personId", "season"]).reset_index(drop=True)


def clear_cache() -> None:
    """Clear all module-level caches."""
    global _dataset_cache, _season_stats_cache, _player_predictions_cache
    _dataset_cache = None
    _season_stats_cache.clear()
    _player_predictions_cache = None
