import pandas as pd
import numpy as np

print("📢 Script started...")

# Load the dataset
try:
    df = pd.read_csv("College_data(1).csv")
    print("✅ CSV file loaded successfully!")
except FileNotFoundError:
    print("❌ Error: CSV file not found!")
    exit()
except pd.errors.EmptyDataError:
    print("❌ Error: CSV file is empty!")
    exit()

# Handle missing values and format fees
df.replace("--", np.nan, inplace=True)

# Ensure UG_fee and PG_fee are strings before replacing commas
df["UG_fee"] = df["UG_fee"].astype(str).str.replace(",", "", regex=True).astype(float)
df["PG_fee"] = df["PG_fee"].astype(str).str.replace(",", "", regex=True).astype(float)

df.fillna({"UG_fee": df["UG_fee"].median(), "PG_fee": df["PG_fee"].median()}, inplace=True)

# Convert ratings to numeric values
num_cols = ["Rating", "Academic", "Accommodation", "Faculty", "Infrastructure", "Placement", "Social_Life"]

try:
    df[num_cols] = df[num_cols].astype(float)
    print("✅ Numeric conversion successful!")
except KeyError as e:
    print(f"❌ Error: Missing column {e}")
    exit()

df.dropna(inplace=True)
df.to_csv("cleaned_college_data.csv", index=False)
print("✅ Data preprocessing completed! Cleaned data saved as 'cleaned_college_data.csv'")
