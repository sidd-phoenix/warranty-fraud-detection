from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained models
model1 = joblib.load('decision_tree_model.pkl')
model2 = joblib.load('logistic_regression_model.pkl')
model3 = joblib.load('random_forest_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        selected_model = data.get('model')

        if not selected_model:
            return jsonify({'error': 'No model selected'}), 400

        # Prepare input data for the model
        input_data = pd.DataFrame([{
            'Region': data.get('region', ''),
            'State': data.get('state', ''),
            'Area': data.get('area', ''),
            'City': data.get('city', ''),
            'Consumer_profile': data.get('consumer_profile', ''),
            'Product_category': data.get('product_category', ''),
            'Product_type': data.get('product_type', ''),
            'AC_1001_Issue': data.get('AC_1001_Issue', 0),
            'AC_1002_Issue': data.get('AC_1002_Issue', 0),
            'AC_1003_Issue': data.get('AC_1003_Issue', 0),
            'TV_2001_Issue': data.get('TV_2001_Issue', 0),
            'TV_2002_Issue': data.get('TV_2002_Issue', 0),
            'TV_2003_Issue': data.get('TV_2003_Issue', 0),
            'Claim_Value': data.get('claim_value', 0),
            'Service_Centre': data.get('service_centre', 0),
            'Product_Age': data.get('product_age', 0),
            'Purchased_from': data.get('purchased_from', ''),
            'Call_details': data.get('call_details', ''),
            'Purpose': data.get('purpose', '')
        }])

        # Select the appropriate model
        model = {'model1': model1, 'model2': model2, 'model3': model3}.get(selected_model)
        if not model:
            return jsonify({'error': 'Invalid model selected'}), 400

        # Predict
        prediction = model.predict(input_data)[0]
        return jsonify({'prediction': int(prediction)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
