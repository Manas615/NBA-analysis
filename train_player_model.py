import pandas as pd
import numpy as np
import os
import joblib
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

path = kagglehub.dataset_download(
    "eoinamoore/historical-nba-data-and-player-box-scores"
)

print("Dataset Path:", path)

csv_file = os.path.join(
    path,
    "PlayerStatisticsExtended.csv"
)

print("Loading:", csv_file)

df = pd.read_csv(
    csv_file,
    low_memory=False
)

print("Cleaning data...")

df["gameDateTimeEst"] = pd.to_datetime(
    df["gameDateTimeEst"],
    errors="coerce"
)

df["season"] = df["gameDateTimeEst"].dt.year

df["numMinutes"] = (
    pd.to_numeric(
        df["numMinutes"],
        errors="coerce"
    )
)

numeric_columns = [
    "points",
    "assists",
    "reboundsTotal",
    "turnovers",
    "trueShootingPercentage",
    "usagePercentage",
    "netRating",
    "playerImpactEstimate"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

print("Creating season-level player stats...")

player_stats = (
    df.groupby(
        ["personId", "season"],
        as_index=False
    )
    .agg({
        "points": "mean",
        "assists": "mean",
        "reboundsTotal": "mean",
        "turnovers": "mean",
        "numMinutes": "mean",
        "trueShootingPercentage": "mean",
        "usagePercentage": "mean",
        "netRating": "mean",
        "playerImpactEstimate": "mean"
    })
)

player_stats = player_stats.sort_values(
    ["personId", "season"]
)

player_stats["next_year_pie"] = (
    player_stats.groupby("personId")
    ["playerImpactEstimate"]
    .shift(-1)
)

player_stats = player_stats.dropna(
    subset=["next_year_pie"]
)

features = [
    "points",
    "assists",
    "reboundsTotal",
    "turnovers",
    "numMinutes",
    "trueShootingPercentage",
    "usagePercentage",
    "netRating",
    "playerImpactEstimate"
]

X = player_stats[features].fillna(0)
y = player_stats["next_year_pie"]

print("Training samples:", len(X))

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)

print("Training model...")

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(
    y_test,
    predictions
)

r2 = r2_score(
    y_test,
    predictions
)

print("\nModel Results")
print("MAE:", round(mae, 4))
print("R2 Score:", round(r2, 4))

os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    model,
    "models/player_model.pkl"
)

print("\nModel Saved")
print("models/player_model.pkl")
