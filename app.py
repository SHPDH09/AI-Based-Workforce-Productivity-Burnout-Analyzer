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
        .container {
            background: rgba(255, 255, 255, 0.95);
            padding: 35px 40px;
            border-radius: 16px;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.4);
            width: 500px;
            max-width: 95%;
            text-align: center;
            margin: 50px auto;
        }
        h1 {
            font-size: 22px;
            margin-bottom: 8px;
            color: #000;
        }
        .result-risk.low {color: green; font-weight: bold;}
        .result-risk.medium {color: orange; font-weight: bold;}
        .result-risk.high {color: red; font-weight: bold;}
        .stButton button {
            width: 100%;
            padding: 12px;
            border: none;
            border-radius: 10px;
            background: linear-gradient(135deg, #007bff, #0056b3);
            color: white;
            font-size: 16px;
            cursor: pointer;
            transition: 0.3s;
        }
        .stButton button:hover {
            background: linear-gradient(135deg, #0056b3, #007bff);
            box-shadow: 0 0 10px rgba(0, 123, 255, 0.6);
            transform: scale(1.03);
        }
        .footer {
            margin-top: 20px;
            background: rgba(255, 255, 255, 0.9);
            color: #000;
            font-weight: bold;
            border-radius: 8px;
            padding: 8px;
            font-size: 13px;
        }
        .footer a {
            color: #007bff;
            text-decoration: none;
            margin: 0 5px;
            font-weight: bold;
        }
        .footer a:hover {
            filter: blur(1px);
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar menu (Removed Result)
menu = st.sidebar.selectbox("Menu", ["Predict", "History"])

if "show_result" not in st.session_state:
    st.session_state.show_result = False

if menu == "Predict":
    if not st.session_state.show_result:
        st.markdown('<div class="container">', unsafe_allow_html=True)
        st.markdown("<h1>AI-Based Workforce Productivity & Burnout Analyzer</h1>", unsafe_allow_html=True)
        st.markdown("<p>Enter employee details to analyze productivity and burnout risk.</p>", unsafe_allow_html=True)

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

                st.session_state.last_result = {"name": name, "risk": risk_level}
                st.session_state.show_result = True
                st.experimental_rerun()

        st.markdown("""
            <div class="footer">
                <p>© 2025 AI-Based Workforce Productivity & Burnout Analyzer</p>
                <p>
                    Contact: <a href="mailto:rk331159@gmail.com">rk331159@gmail.com</a> |
                    Portfolio: <a href="https://portfolioraunakprasad.netlify.app/" target="_blank">Raunak Prasad</a>
                </p>
            </div>
            </div>
        """, unsafe_allow_html=True)
    else:
        # Show Result directly after prediction
        result = st.session_state.last_result
        risk_class = "low" if result["risk"] == "Low" else "medium" if result["risk"] == "Medium" else "high"
        st.markdown(f"""
            <div class="container">
                <h2>Analysis Result for {result["name"]}</h2>
                <p class="result-risk {risk_class}">Burnout Risk Level: {result["risk"]}</p>
                <p><a href="#" onclick="window.location.reload()">Analyze Another Employee</a></p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Analyze Another Employee"):
            st.session_state.show_result = False
            st.experimental_rerun()

elif menu == "History":
    st.subheader("Prediction History")
    df = pd.read_sql("SELECT * FROM results", engine)
    st.dataframe(df)
