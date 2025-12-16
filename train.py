import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("data/quotes.csv")
df.columns = ["quote", "author"]
df = df.dropna()

X_text = df["quote"]
y = df["author"]

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=500
)

X = vectorizer.fit_transform(X_text)

print("Feature matrix shape:", X.shape)
