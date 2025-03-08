import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import pickle

# Load cleaned dataset
df = pd.read_csv("cleaned_college_data.csv")

# Features for recommendation
features = ["UG_fee", "PG_fee", "Rating", "Academic", "Accommodation", "Faculty", "Infrastructure", "Placement", "Social_Life"]

# Standardize the numerical features
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[features])

# Train the KNN model
knn = NearestNeighbors(n_neighbors=5, metric="euclidean")
knn.fit(df_scaled)

# Save the model & scaler
with open("college_recommendation_model.pkl", "wb") as model_file:
    pickle.dump(knn, model_file)

with open("scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

print("✅ Model training completed! Model saved as 'college_recommendation_model.pkl'")
