# app/app.py
from flask import Flask, render_template, request
from pathlib import Path
import sys

# Make src/ importable
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from inference import SentimentService

app = Flask(__name__)

# OPTION A (instant demo): use strong pretrained model
svc = SentimentService(model_dir=None)

# OPTION B (after you train): point to your saved model dir
# svc = SentimentService(model_dir=str(Path(__file__).resolve().parents[1] / "models" / "distilbert_best"))

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", result=None)

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form.get("review_text", "").strip()
    if not text:
        return render_template("index.html", result={"error": "Please enter a review."})
    out = svc.predict(text)
    return render_template("index.html", result=out, original=text)

if __name__ == "__main__":
    print("Starting Flask dev server on http://127.0.0.1:5000 ...")
    app.run(host="127.0.0.1", port=5000, debug=True)
