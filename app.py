from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

print(">>> Starting app.py")  # debug print so we know file is running

# Load model + feature columns
artifact = joblib.load("house_price_rf_model.pkl")
model = artifact["model"]
feature_cols = artifact["features"]  # list of feature names

print(">>> Loaded model with features:", feature_cols)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction_text = None

    if request.method == "POST":
        # Collect form values in correct order
        input_data = {}
        for feat in feature_cols:
            raw_val = request.form.get(feat, "0")
            try:
                val = float(raw_val)
            except ValueError:
                val = 0.0
            input_data[feat] = val

        # Create DataFrame with one row
        X_new = pd.DataFrame([input_data], columns=feature_cols)

        # Predict
        pred_price = model.predict(X_new)[0]
        prediction_text = f"Estimated House Price: ${pred_price:,.0f}"

    return render_template("index.html",
                           prediction_text=prediction_text,
                           feature_cols=feature_cols)


if __name__ == "__main__":
    print(">>> Running Flask app...")
    app.run(debug=True)
