import pandas as pd
import os
import joblib
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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

numeric_cols = [
    "netRating",
    "estimatedOffensiveRating",
    "estimatedDefensiveRating",
    "win"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

df = df.dropna(
    subset=[
        "netRating",
        "estimatedOffensiveRating",
        "estimatedDefensiveRating",
        "win"
    ]
)

print("Remaining rows:", len(df))

X = df[[
    "netRating",
    "estimatedOffensiveRating",
    "estimatedDefensiveRating"
]]

y = df["win"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000
        )
    )
])

print("Training matchup model...")

model.fit(
    X_train,
    y_train
)

predictions = model.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\nModel Results")
print(
    "Accuracy:",
    round(accuracy, 4)
)

os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    model,
    "models/matchup_model.pkl"
)

print("\nModel Saved")
print("models/matchup_model.pkl")
