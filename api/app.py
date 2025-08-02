from flask import Flask, render_template, request
import joblib
import numpy as np
import os
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

# Load ML model
model_path = 'model/burnout_model.pkl'
if not os.path.exists(model_path):
    raise FileNotFoundError("Model file not found! Please train the model first using model_train.py.")
model = joblib.load(model_path)

# Database configuration
db_url = os.getenv('DATABASE_URL')
if db_url and db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://")

app.config['SQLALCHEMY_DATABASE_URI'] = db_url or 'sqlite:///employee_results.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Database model
class Result(db.Model):
    __tablename__ = 'results'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(100), nullable=False)
    working_hours = db.Column(db.Integer, nullable=False)
    tasks_completed = db.Column(db.Integer, nullable=False)
    breaks = db.Column(db.Integer, nullable=False)
    satisfaction = db.Column(db.Integer, nullable=False)
    prediction = db.Column(db.String(50), nullable=False)

# Create tables
with app.app_context():
    db.create_all()

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
        result = Result(
            name=name,
            working_hours=hours,
            tasks_completed=tasks,
            breaks=breaks,
            satisfaction=satisfaction,
            prediction=risk_level
        )
        db.session.add(result)
        db.session.commit()

        return render_template('result.html', name=name, risk=risk_level)

    except Exception as e:
        return f"Error: {str(e)}"

# Route to view previous results
@app.route('/history')
def history():
    records = Result.query.all()
    return render_template('history.html', records=records)

if __name__ == "__main__":
    app.run(debug=True)
