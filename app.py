# app.py
import os
import warnings
import pickle
import numpy as np
from flask import Flask, request, render_template, abort
from feature import FeatureExtraction

warnings.filterwarnings("ignore")

# --- Load model once ---
MODEL_PATH = os.path.join("pickle", "model.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

def get_safe_probability(model, xrow: np.ndarray) -> float:
    """
    Returns P(class==1) robustly, regardless of class order in model.classes_.
    """
    proba = model.predict_proba(xrow)[0]
    # classes_ could be [-1, 1] or [0, 1]; find index of 'safe' class (1)
    if hasattr(model, "classes_"):
        classes = list(model.classes_)
        if 1 in classes:
            idx = classes.index(1)
            return float(proba[idx])
    # Fallback: assume last column is positive class
    return float(proba[-1])

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        url = (request.form.get("url") or "").strip()
        if not url:
            return render_template("index.html", xx=-1, url="", error="Please enter a URL.")

        try:
            # Extract features (should return exactly 30 features in the same order as training)
            feats = FeatureExtraction(url).getFeaturesList()
            x = np.array(feats, dtype=float).reshape(1, 30)

            # Predict
            y_pred = model.predict(x)[0]  # -1 = phishing, +1 = safe (as trained)
            p_safe = get_safe_probability(model, x)  # in [0,1]

            # Pass p_safe to the template as `xx`
            return render_template("index.html", xx=round(p_safe, 2), url=url, y_pred=int(y_pred))
        except Exception as e:
            # Don’t expose internals to users; show a friendly error
            return render_template(
                "index.html",
                xx=-1,
                url=url,
                error="Could not analyze the URL right now. Please try a different URL or try again."
            )

    # GET
    return render_template("index.html", xx=-1)

@app.get("/healthz")
def healthz():
    return {"status": "ok"}, 200

if __name__ == "__main__":
    # Use env PORT for cloud platforms; default 5000 locally
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
