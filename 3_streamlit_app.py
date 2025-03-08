import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
import plotly.express as px
import plotly.graph_objects as go

# Load the dataset and model
df = pd.read_csv("cleaned_college_data.csv")
with open("college_recommendation_model.pkl", "rb") as model_file:
    knn = pickle.load(model_file)
with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Define features
features = ["UG_fee", "PG_fee", "Rating", "Academic", "Accommodation", "Faculty", "Infrastructure", "Placement", "Social_Life"]

# Streamlit UI Enhancements
st.set_page_config(page_title="🎓 College Recommendation System", page_icon="🎓", layout="wide")

# Title
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🎓 College Recommendation System</h1>", unsafe_allow_html=True)

# Sidebar for user inputs
st.sidebar.header("📌 User Preferences")
stream = st.sidebar.selectbox("Select Stream", df["Stream"].unique())
state = st.sidebar.selectbox("Select State", df["State"].unique())
budget = st.sidebar.slider("Maximum UG Fee (per year)", 50000, 500000, 150000)
min_rating = st.sidebar.slider("Minimum College Rating", 0.0, 10.0, 7.0)
accommodation = st.sidebar.slider("Minimum Accommodation Score", 0.0, 10.0, 7.0)
placement = st.sidebar.slider("Minimum Placement Score", 0.0, 10.0, 7.0)
extra_facilities = st.sidebar.multiselect("Preferred Facilities (For Display Only)", ["WiFi", "Sports Complex", "Library", "Gym", "Medical Facilities", "Hostel", "Cafeteria"], default=["WiFi", "Library"])

# Button for recommendations
if st.sidebar.button("🔍 Recommend Colleges"):
    filtered_df = df[(df["Stream"] == stream) & (df["State"] == state) & (df["Rating"] >= min_rating)]
    
    if len(filtered_df) < 5:
        st.warning("⚠️ Not enough colleges found for this selection. Try adjusting filters.")
    else:
        user_input = scaler.transform([[budget, budget, min_rating, 8.0, accommodation, 8.0, 8.0, placement, 8.0]])
        scaled_filtered = scaler.transform(filtered_df[features])

        knn_filtered = knn.fit(scaled_filtered)
        distances, indices = knn_filtered.kneighbors(user_input)

        recommendations = filtered_df.iloc[indices[0]][["College_Name", "State", "Stream", "Rating", "UG_fee", "Placement", "Accommodation"]]
        st.markdown("### 🎯 Recommended Colleges for You")
        st.dataframe(recommendations.style.set_properties(**{'background-color': '#f0f0f0', 'color': 'black'}))

        # Improved Bar Chart
        st.markdown("### 📊 College Ratings Comparison")
        fig = px.bar(recommendations, x="College_Name", y="Rating", color="Rating", 
                     color_continuous_scale="Blues", labels={"Rating": "College Rating"}, 
                     title="College Ratings Comparison")
        st.plotly_chart(fig, use_container_width=True)

        # Scatter plot for Fees vs Placement
        st.markdown("### 📈 Fees vs Placement Comparison")
        fig2 = px.scatter(recommendations, x="UG_fee", y="Placement", size="Rating", color="Rating", 
                          hover_name="College_Name", labels={"UG_fee": "UG Fee", "Placement": "Placement Score"}, 
                          title="UG Fee vs Placement Score")
        st.plotly_chart(fig2, use_container_width=True)

        # Pie Chart for College Distribution by State
        st.markdown("### 🏛️ College Distribution by State")
        fig3 = px.pie(recommendations, names="State", title="Colleges Distribution by State", color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig3)

        # Radar Chart for College Comparison
        st.markdown("### 📊 College Feature Comparison")
        radar_data = recommendations.melt(id_vars=["College_Name"], value_vars=["Rating", "Placement", "Accommodation"], var_name="Feature", value_name="Score")
        fig4 = px.line_polar(radar_data, r="Score", theta="Feature", color="College_Name", line_close=True, title="College Feature Comparison")
        st.plotly_chart(fig4)

        # Show a map if location data is available
        if "Latitude" in df.columns and "Longitude" in df.columns:
            st.markdown("### 🌍 College Locations on Map")
            map_data = recommendations[["Latitude", "Longitude"]].dropna()
            st.map(map_data)

# About Me Section
st.markdown("### 👨‍💻 About Me")
st.markdown("""
**Udit Katiyar**
- 🎓 Second-year Computer Science Engineering student at Lovely Professional University
- 💻 Passionate about AI, Machine Learning, and Web Development
- 🚀 Skilled in Python, C++, Flask, React, and Data Science
- 🔍 Interested in building innovative solutions and contributing to open-source projects

Find me on [GitHub](https://github.com/katiyarudit) | [LinkedIn](https://linkedin.com/in/katiyarudit)
""")

# Footer
st.markdown("""
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        text-align: center;
        padding: 10px;
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
    }
    </style>
    <div class='footer'>Made with ❤️ by Udit Katiyar</div>
    """, unsafe_allow_html=True)
