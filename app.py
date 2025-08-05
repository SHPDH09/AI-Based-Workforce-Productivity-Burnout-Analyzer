import streamlit as st
import joblib
import numpy as np
import os
import pandas as pd
from sqlalchemy import create_engine, text

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
    conn.execute(text("""
    CREATE TABLE IF NOT EXISTS results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        working_hours INTEGER NOT NULL,
        tasks_completed INTEGER NOT NULL,
        breaks INTEGER NOT NULL,
        satisfaction INTEGER NOT NULL,
        prediction TEXT NOT NULL
    )
    """))

# Custom CSS
st.markdown("""
    <style>
        body {
            font-family: 'Poppins', sans-serif;
            background: linear-gradient(135deg, #d4af37, #c0c0c0);
            color: #000;
        }
        .result-container {
            background: rgba(255, 255, 255, 0.95);
            padding: 35px;
            border-radius: 16px;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.4);
            width: 500px;
            max-width: 95%;
            text-align: center;
            margin: 50px auto;
        }
        .result-risk {
            font-size: 20px;
            font-weight: bold;
            margin-top: 20px;
            padding: 10px;
            border-radius: 8px;
        }
        .low-risk { background: #d4edda; color: #155724; }
        .medium-risk { background: #fff3cd; color: #856404; }
        .high-risk { background: #f8d7da; color: #721c24; }
        .result-container a {
            margin: 10px;
            text-decoration: none;
            font-weight: bold;
            color: #007bff;
        }
        .result-container a:hover { filter: blur(1px); }
    </style>
""", unsafe_allow_html=True)

# Sidebar menu
menu = st.sidebar.selectbox("Menu", ["Predict", "History"])

if menu == "Predict":
    st.markdown("<h1 style='text-align:center;'>AI-Based Workforce Productivity & Burnout Analyzer</h1>", unsafe_allow_html=True)

    name = st.text_input("Employee Name")
    hours = st.number_input("Working Hours (per week)", min_value=0)
    tasks = st.number_input("Tasks Completed", min_value=0)
    breaks = st.number_input("Breaks Taken", min_value=0)
    satisfaction = st.slider("Satisfaction Level (1-10)", 1, 10, 5)

    if st.button("Analyze"):
        if name.strip() == "":
            st.error("Please enter the employee name.")
        else:
            features = np.array([[hours, tasks, breaks, satisfaction]])
            prediction = model.predict(features)[0]

            risk_map = {0: 'Low', 1: 'Medium', 2: 'High'}
            risk_level = risk_map.get(prediction, 'Unknown')

            # Save to DB
            with engine.begin() as conn:
                conn.execute(text("""
                    INSERT INTO results (name, working_hours, tasks_completed, breaks, satisfaction, prediction)
                    VALUES (:name, :hours, :tasks, :breaks, :satisfaction, :prediction)
                """), {
                    "name": name,
                    "hours": hours,
                    "tasks": tasks,
                    "breaks": breaks,
                    "satisfaction": satisfaction,
                    "prediction": risk_level
                })

            # Display result like result.html
            st.markdown(f"""
                <div class="result-container">
                    <h2>Analysis Result for {name}</h2>
                    <p class="result-risk {'low-risk' if risk_level=='Low' else 'medium-risk' if risk_level=='Medium' else 'high-risk'}">
                        Burnout Risk Level: {risk_level}
                    </p>
                    <a href="#">Analyze Another Employee</a> |
                    <a href="#">View Prediction History</a>
                </div>
            """, unsafe_allow_html=True)

elif menu == "History":
    st.subheader("Prediction History")
    df = pd.read_sql("SELECT * FROM results", engine)
    st.dataframe(df)
