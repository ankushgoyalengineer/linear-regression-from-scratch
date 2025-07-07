import pandas as pd
from model import train_model

df = pd.read_csv('boston.csv')
X = df['RM'].values
y = df['MEDV'].values

theta0, theta1, mean, std = train_model(X, y)

# Save values manually or print them for later use
print(f"theta0 = {theta0}")
print(f"theta1 = {theta1}")
print(f"mean = {mean}")
print(f"std = {std}")
