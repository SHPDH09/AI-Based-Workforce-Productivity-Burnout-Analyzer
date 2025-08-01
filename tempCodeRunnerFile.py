from flask import Flask, render_template, request
import joblib
import numpy as np
import sqlite3
import os

app = Flask(__name__)

# Load ML model
model_path = 'model/burnout_model.pkl'
if not os.path.exists(model_path):
    raise FileNotFoundError("Model file not found! Please train the model first using model_train.py.")
model = joblib.load(model_path)

# Database initialization
def init_db():
    conn = sqlite3.connect('employee_results.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT,
                        working_hours INTEGER,
                        tasks_completed INTEGER,
                        breaks INTEGER,
                        satisfaction INTEGER,
                        prediction TEXT
                    )''')
    conn.commit()
    conn.close()

init_db()

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        name = request.form['name']
        hours = int(request.form['working_hours'])
        tasks = int(request.form['tasks_completed'])
        breaks = int(request.form['breaks'])
        satisfaction = int(request.form['satisfaction'])

        features = np.array([[hours, tasks, breaks, satisfaction]])
        prediction = model.predict(features)[0]

        risk_map = {0: 'Low', 1: 'Medium', 2: 'High'}
        risk_level = risk_map.get(prediction, 'Unknown')

        # Save result in DB
        conn = sqlite3.connect('employee_results.db')
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO results (name, working_hours, tasks_completed, breaks, satisfaction, prediction) VALUES (?, ?, ?, ?, ?, ?)",
            (name, hours, tasks, breaks, satisfaction, risk_level)
        )
        conn.commit()
        conn.close()

        return render_template('result.html', name=name, risk=risk_level)

    except Exception as e:
        return f"Error: {str(e)}"

# Route to view previous results
@app.route('/history')
def history():
    conn = sqlite3.connect('employee_results.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM results")
    rows = cursor.fetchall()
    conn.close()
    return render_template('history.html', records=rows)

if __name__ == "__main__":
    app.run(debug=True)
