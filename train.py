import pandas as pd

# Load data
df = pd.read_csv("data/quotes.csv")

# Rename columns for clarity
df.columns = ["quote", "author"]

# Drop empty values (safety)
df = df.dropna()

print("Total samples:", len(df))
print(df.head())

