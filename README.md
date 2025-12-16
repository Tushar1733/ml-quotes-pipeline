# Quotes Author Classification (ML Pipeline)

This project builds a simple machine learning pipeline to predict
the author of a quote using text features.

## Dataset
- Scraped from https://quotes.toscrape.com
- Contains quote text and author names

## Tech Stack
- Python
- pandas
- scikit-learn

## Pipeline
1. Load and clean dataset
2. Convert text to TF-IDF features
3. Train Logistic Regression model
4. Evaluate using accuracy

## How to Run
```bash
pip install -r requirements.txt
python train.py

## Model Performance Notes

The classification accuracy is relatively low (~10%). This is expected due to:
- A large number of author classes
- Very few quotes per author
- Short quote length and overlapping writing styles

The goal of this project is to demonstrate a clean and correct
machine learning pipeline rather than to optimize model accuracy.
