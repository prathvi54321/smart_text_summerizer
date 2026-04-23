import pandas as pd

# Load the raw data
# encoding='latin-1' is used because news data often has special characters
df = pd.read_csv("news_summary.csv", encoding='latin-1')

# The dataset has many columns. We only want 'ctext' (Article) and 'text' (Summary)
# We drop any rows that are empty
clean_df = df[['text', 'ctext']].dropna()

# Rename them so it's easier to understand
clean_df.columns = ['summary', 'article']

# We will take only the first 100 rows for a "proof of concept" training
# This makes training fast so you can show it to your teacher quickly
clean_df.head(100).to_csv("manual_training_data.csv", index=False)

print("✅ Step 2 Complete: 'manual_training_data.csv' is ready!")