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

# Custom CSS for styling
st.markdown("""
    <style>
    .main-title {
        font-size: 40px;
        color: #2E86C1;
        text-align: center;
        margin-bottom: 20px;
    }
    .sub-title {
        font-size: 24px;
        color: #1F618D;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .result-box {
        padding: 20px;
        background-color: #F2F3F4;
        border-radius: 10px;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
        color: #154360;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit UI
st.markdown('<div class="main-title">AI-Based Workforce Productivity & Burnout Analyzer</div>', unsafe_allow_html=True)

menu = st.sidebar.radio("📌 Navigation", ["🏠 Home", "📊 Predict", "📜 History"])

if menu == "🏠 Home":
    st.markdown('<div class="sub-title">Welcome to the Burnout Analyzer</div>', unsafe_allow_html=True)
    st.write("This tool helps predict employee burnout levels based on working hours, tasks completed, breaks taken, and satisfaction levels.")
    st.image("https://cdn-icons-png.flaticon.com/512/2920/2920244.png", width=200)
    st.write("Use the sidebar to navigate to Prediction or History sections.")

elif menu == "📊 Predict":
    st.markdown('<div class="sub-title">Employee Burnout Risk Prediction</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("👤 Employee Name")
        hours = st.number_input("⏱ Working Hours", min_value=0)
    with col2:
        tasks = st.number_input("✅ Tasks Completed", min_value=0)
        breaks = st.number_input("☕ Breaks Taken", min_value=0)

    satisfaction = st.slider("😊 Satisfaction Level (1-10)", 1, 10, 5)

    if st.button("🔍 Predict"):
        if name.strip() == "":
            st.error("Please enter employee name.")
        else:
            features = np.array([[hours, tasks, breaks, satisfaction]])
            prediction = model.predict(features)[0]

            risk_map = {0: 'Low', 1: 'Medium', 2: 'High'}
            risk_level = risk_map.get(prediction, 'Unknown')

            # Save result in DB
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

            st.markdown(f'<div class="result-box">Burnout Risk for {name}: {risk_level}</div>', unsafe_allow_html=True)

elif menu == "📜 History":
    st.markdown('<div class="sub-title">Prediction History</div>', unsafe_allow_html=True)
    df = pd.read_sql("SELECT * FROM results", engine)
    st.dataframe(df, use_container_width=True)
