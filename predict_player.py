import joblib
import pandas as pd

model = joblib.load(
    "models/player_model.pkl"
)

sample_player = pd.DataFrame([{
    "points": 28,
    "assists": 7,
    "reboundsTotal": 8,
    "turnovers": 3,
    "numMinutes": 35,
    "trueShootingPercentage": 0.61,
    "usagePercentage": 29,
    "netRating": 7.5,
    "playerImpactEstimate": 0.18
}])

prediction = model.predict(
    sample_player
)

print(
    "Predicted Next Season PIE:",
    round(prediction[0], 4)
)
