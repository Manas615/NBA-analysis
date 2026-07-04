import pandas as pd
import kagglehub
import os

path = kagglehub.dataset_download(
    "eoinamoore/historical-nba-data-and-player-box-scores"
)

files = os.listdir(path)
print(files)

csv_file = os.path.join(path, files[0])

df = pd.read_csv(csv_file)

print(df.head())
print(df.columns)
