# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os

MODEL_PATH = "sentiment_model.joblib"

app = Flask(__name__)
CORS(app)

# Load model (ensure you ran train_model.py first)
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model not found. Run train_model.py to create sentiment_model.joblib")

model = joblib.load(MODEL_PATH)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    if not text.strip():
        return jsonify({"error": "Empty text"}), 400

    pred = model.predict([text])[0]
    prob = None
    try:
        probs = model.predict_proba([text])[0]
        prob = float(max(probs))
    except Exception:
        prob = None

    label = "Positive" if pred == 1 else "Negative"
    return jsonify({"label": label, "score": prob})

@app.route("/")
def home():
    return "Sentiment API running"

if __name__ == "__main__":
    app.run(debug=True)
