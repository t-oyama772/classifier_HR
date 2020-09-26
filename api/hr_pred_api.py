from flask import Flask, request, jsonify, abort
import pandas as pd
import joblib
from datetime import datetime
import sys
app = Flask(__name__)

# load model
model = joblib.load('../model/CLASSIFIER_001_rf_over.pkl')


@app.route('/hr_pred_api/predict', methods=["POST"])
def predict():
    """Function that returns a predicted value when POST"""
    try:
        # feature
        X = pd.DataFrame(request.json, index=[0])
        print(X)
        # predict
        y_pred = model.predict_proba(X)[:,1]
        print(y_pred)
        response = {"status": "OK", "predicted": y_pred[0]}
        return jsonify(response), 200
    except Exception as e:
        print(e)
        abort(400)


@app.errorhandler(400)
def error_handler(error):
    """response when abort(400)"""
    response = {"status": "Error", "message": "Invalid Parameters"}
    return jsonify(response), error.code


if __name__ == "__main__":
    app.run(debug=True)  # server run
