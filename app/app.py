from flask import Flask, render_template, request, jsonify
from pathlib import Path
import sys, os

# allow importing from src/
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from inference import SentimentService

app = Flask(__name__)

# === use your trained model ===
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "distilbert_best"
svc = SentimentService(model_dir=str(MODEL_PATH))

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "No text provided."}), 400
    out = svc.predict(text)
    return jsonify(out)

if __name__ == "__main__":
    print("Starting Flask on http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=8080, debug=True)
