from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

model = joblib.load("models/model.pkl")
pipeline = joblib.load("models/pipeline.pkl")


columns_order = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    try:

        data = request.get_json()

        time = float(data['time'])
        amount = float(data['amount'])

        features = data['features']

        if len(features) != 28:
            return jsonify({
                "error": "Exactly 28 feature values required."
            })

        features = [float(x) for x in features]

        input_data = [time] + features + [amount]

        input_df = pd.DataFrame(
            [input_data],
            columns=columns_order
        )

        transformed_data = pipeline.transform(input_df)

        fraud_probability = float(
            model.predict_proba(transformed_data)[0][1]
        )

        threshold = 0.20

        prediction = int(fraud_probability >= threshold)

        label = "Fraud" if prediction == 1 else "Not Fraud"

        return jsonify({
            "prediction": prediction,
            "label": label,
            "fraud_probability": round(fraud_probability * 100, 2)
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        })


# -----------------------------
# RUN APP
# -----------------------------
if __name__ == "__main__":

    app.run(host="0.0.0.0", port=5000)

