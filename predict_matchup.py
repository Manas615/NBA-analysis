import joblib
import pandas as pd

model = joblib.load(
    "models/matchup_model.pkl"
)

team_a = pd.DataFrame([{
    "netRating": 7.5,
    "estimatedOffensiveRating": 118,
    "estimatedDefensiveRating": 110
}])

probability = model.predict_proba(
    team_a
)[0][1]

print(
    f"Win Probability: {probability * 100:.2f}%"
)
