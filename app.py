from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("HDI.pkl", "rb"))


def classify_hdi(score):
    """Classify an HDI score into the official UNDP tiers."""
    if score >= 0.800:
        return "Very High", "very-high"
    if score >= 0.700:
        return "High", "high"
    if score >= 0.550:
        return "Medium", "medium"
    return "Low", "low"


def run_prediction(payload):
    life = float(payload["life"])
    expected = float(payload["expected"])
    mean = float(payload["mean"])
    income = float(payload["income"])

    features = np.array([[life, expected, mean, income]])
    raw_prediction = float(model.predict(features)[0])

    # HDI is bounded between 0 and 1
    score = max(0.0, min(1.0, raw_prediction))
    tier_label, tier_class = classify_hdi(score)

    return {
        "score": round(score, 3),
        "tier_label": tier_label,
        "tier_class": tier_class,
        "inputs": {
            "life": life,
            "expected": expected,
            "mean": mean,
            "income": income,
        },
    }


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(silent=True) or request.form
        result = run_prediction(payload)
    except (KeyError, ValueError, TypeError) as exc:
        error_message = "Please provide valid numeric values for every field."
        if request.is_json or request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify({"error": error_message}), 400
        return render_template("index.html", prediction_text=None, error_text=error_message)

    if request.is_json or request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return jsonify(result)

    prediction_text = f"Predicted HDI: {result['score']:.3f} ({result['tier_label']})"
    return render_template("index.html", prediction_text=prediction_text, error_text=None)


if __name__ == "__main__":
    app.run(debug=True)
