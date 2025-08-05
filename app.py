import streamlit as st
import joblib
import numpy as np
import os
import pandas as pd
from sqlalchemy import create_engine

# Load ML model
model_path = 'model/burnout_model.pkl'
if not os.path.exists(model_path):
    st.error("Model file not found! Please train the model first using model_train.py.")
    st.stop()

model = joblib.load(model_path)

# Database configuration
db_url = os.getenv('DATABASE_URL')
if db_url and db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://")

db_url = db_url or 'sqlite:///employee_results.db'
engine = create_engine(db_url)

# Create table if not exists
with engine.connect() as conn:
    conn.execute("""
    CREATE TABLE IF NOT EXISTS results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        working_hours INTEGER NOT NULL,
        tasks_completed INTEGER NOT NULL,
        breaks INTEGER NOT NULL,
        satisfaction INTEGER NOT NULL,
        prediction TEXT NOT NULL
    )
    """)

# Streamlit UI
st.title("AI-Based Workforce Productivity & Burnout Analyzer")

menu = st.sidebar.selectbox("Menu", ["Predict", "History"])

if menu == "Predict":
    st.subheader("Employee Burnout Risk Prediction")

    name = st.text_input("Employee Name")
    hours = st.number_input("Working Hours", min_value=0)
    tasks = st.number_input("Tasks Completed", min_value=0)
    breaks = st.number_input("Breaks Taken", min_value=0)
    satisfaction = st.slider("Satisfaction Level (1-10)", 1, 10, 5)

    if st.button("Predict"):
        features = np.array([[hours, tasks, breaks, satisfaction]])
        prediction = model.predict(features)[0]

        risk_map = {0: 'Low', 1: 'Medium', 2: 'High'}
        risk_level = risk_map.get(prediction, 'Unknown')

        # Save result in DB
        with engine.begin() as conn:
            conn.execute("""
                INSERT INTO results (name, working_hours, tasks_completed, breaks, satisfaction, prediction)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (name, hours, tasks, breaks, satisfaction, risk_level))

        st.success(f"Burnout Risk for {name}: {risk_level}")

elif menu == "History":
    st.subheader("Prediction History")
    df = pd.read_sql("SELECT * FROM results", engine)
    st.dataframe(df)
