from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load model
model = joblib.load("chennai_crime_predictor.joblib")

# Define input fields
input_features = ['Area_Name', 'Pincode', 'Latitude', 'Longitude', 'Zone_Name']

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_df = pd.DataFrame([data])
    prediction = model.predict(input_df)
    
    # List of output features (same order as training)
    output_features = [
        'Crime_Type', 'Crime_Subtype', 'Crime_Severity', 'Victim_Age_Group',
        'Victim_Gender', 'Suspect_Count', 'Weapon_Used', 'Gang_Involvement',
        'Vehicle_Used', 'CCTV_Captured', 'Reported_By', 'Response_Time_Minutes',
        'Arrest_Made', 'Crime_History_Count', 'Crimes_Same_Type_Count', 'Risk_Level'
    ]

    result = {feature: prediction[0][i] for i, feature in enumerate(output_features)}
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
