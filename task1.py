from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
model_path = 'LinearModel.pkl'
with open(model_path, 'rb') as f:
    pipeline = pickle.load(f)

# R² Score calculated during training
r2_score_train = 0.8131052864906375  # Replace with your actual R² score after training

@app.route('/')
def home():
    return render_template('index.html', r2_score=r2_score_train)

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve input data from the form
    try:
        input_data = {
            'GrLivArea': float(request.form['GrLivArea']),
            'BedroomAbvGr': float(request.form['BedroomAbvGr']),
            'FullBath': float(request.form['FullBath']),
            'TotRmsAbvGrd': float(request.form['TotRmsAbvGrd']),
            'LotArea': float(request.form['LotArea']),
            'OverallQual': float(request.form['OverallQual']),
            'OverallCond': float(request.form['OverallCond']),
            'TotalBsmtSF': float(request.form['TotalBsmtSF']),
            '1stFlrSF': float(request.form['1stFlrSF']),
        }
        input_df = pd.DataFrame([input_data])

        # Log transformation of inputs (ensure they are consistent with training transformations)
        for column in input_df.columns:
            input_df[column] = np.log1p(input_df[column])

        # Make prediction
        predicted_log_price = pipeline.predict(input_df)[0]
        predicted_price = np.expm1(predicted_log_price)  # Reverse log transformation

        # Ensure non-negative prediction
        predicted_price = max(predicted_price, 0)  # Set minimum price to 0

        return render_template('index.html', prediction_text=f"Predicted sales price for given house: {predicted_price:.2f}", r2_score=r2_score_train)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}", r2_score=r2_score_train)

if __name__ == "__main__":
    app.run(debug=True, port=5001)
