# AI-Powered NBA Trade Simulator

An AI-powered NBA trade simulator that predicts player performance, evaluates trade impact, analyzes roster chemistry, validates salary cap rules, and estimates season wins and championship probabilities using machine learning.

## Features

- Player performance prediction using XGBoost
- Team matchup prediction using Logistic Regression
- Trade impact analysis
- Roster strength evaluation
- Chemistry-aware lineup scoring
- Injury risk prediction
- Salary cap validation
- Championship probability simulation using Monte Carlo
- FastAPI backend
- Agentic AI interface for natural language trade queries

## Tech Stack

- Python
- FastAPI
- XGBoost
- Scikit-learn
- Pandas
- NumPy
- Joblib
- NBA API
- KaggleHub

## Project Structure

```
nba-analysis/
│── api/
│── models/
│── championship_simulator.py
│── chemistry_model.py
│── injury_predictor.py
│── salary_cap.py
│── trade_engine.py
│── trade_simulator.py
│── train_player_model.py
│── train_matchup_model.py
│── train_chemistry_model.py
│── train_injury_model.py
│── requirements.txt
```

## Installation

Clone the repository.

```bash
git clone https://github.com/Manas615/NBA-analysis.git
cd NBA-analysis
```

Create a virtual environment.

```bash
python -m venv venv
```

Activate the environment.

Linux/macOS

```bash
source venv/bin/activate
```

Windows

```bash
venv\Scripts\activate
```

Install dependencies.

```bash
pip install -r requirements.txt
```

## Running the Project

Train the models.

```bash
python train_player_model.py
python train_matchup_model.py
python train_chemistry_model.py
python train_injury_model.py
```

Run the trade simulator.

```bash
python trade_simulator.py
```

Run the FastAPI server.

```bash
uvicorn api.main:app --reload
```

## Future Improvements

- Agentic AI workflow for autonomous trade analysis
- Multi-agent decision making
- Live NBA statistics integration
- Player contract and salary optimization
- Interactive web dashboard
- Explainable AI for trade recommendations

## License

This project is for educational and research purposes.
