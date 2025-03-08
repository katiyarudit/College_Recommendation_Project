import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the dataset and model
df = pd.read_csv("cleaned_college_data.csv")
with open("college_recommendation_model.pkl", "rb") as model_file:
    knn = pickle.load(model_file)
with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Define features
features = ["UG_fee", "PG_fee", "Rating", "Academic", "Accommodation", "Faculty", "Infrastructure", "Placement", "Social_Life"]

# Streamlit UI
st.title("🎓 College Recommendation System")

# User input fields
stream = st.selectbox("Select Stream", df["Stream"].unique())
state = st.selectbox("Select State", df["State"].unique())
budget = st.slider("Maximum UG Fee (per year)", 50000, 500000, 150000)
min_rating = st.slider("Minimum College Rating", 0.0, 10.0, 7.0)

if st.button("Recommend Colleges"):
    # Filter dataset based on user input
    filtered_df = df[(df["Stream"] == stream) & (df["State"] == state) & (df["Rating"] >= min_rating)]
    
    if len(filtered_df) < 5:
        st.warning("⚠️ Not enough colleges found for this selection.")
    else:
        # Scale input
        user_input = scaler.transform([[budget, budget, min_rating, 8.0, 7.5, 8.0, 8.0, 7.5, 8.0]])
        scaled_filtered = scaler.transform(filtered_df[features])

        # Train new KNN for this subset
        knn_filtered = knn.fit(scaled_filtered)
        distances, indices = knn_filtered.kneighbors(user_input)

        # Show recommendations
        recommendations = filtered_df.iloc[indices[0]][["College_Name", "State", "Stream", "Rating", "UG_fee"]]
        st.table(recommendations)
