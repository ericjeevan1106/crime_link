from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)  # ✅ Corrected from _name_ to __name__

# ✅ Load the lightweight model
model = joblib.load("chennai_crime_predictor_light.joblib")

# Define input fields
input_features = ['Area_Name', 'Pincode', 'Latitude', 'Longitude', 'Zone_Name']

# Define output fields (same order used during training)
output_features = [
    'Crime_Type', 'Crime_Subtype', 'Crime_Severity', 'Victim_Age_Group',
    'Victim_Gender', 'Suspect_Count', 'Weapon_Used', 'Gang_Involvement',
    'Vehicle_Used', 'CCTV_Captured', 'Reported_By', 'Response_Time_Minutes',
    'Arrest_Made', 'Crime_History_Count', 'Crimes_Same_Type_Count', 'Risk_Level'
]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON input
        data = request.get_json()
        
        # Validate input
        if not all(feature in data for feature in input_features):
            return jsonify({'error': 'Missing one or more input features'}), 400

        # Prepare input for prediction
        input_df = pd.DataFrame([data])
        prediction = model.predict(input_df)

        # Map predictions to output features
        result = {feature: prediction[0][i] for i, feature in enumerate(output_features)}
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ✅ Entry point
if __name__ == '__main__':
    app.run(debug=True)
