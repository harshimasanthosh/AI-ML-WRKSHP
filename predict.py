import pandas as pd
import joblib

model = joblib.load('nn_model.pkl')
x_scaler = joblib.load('x_scaler.pkl')
thal_encoder = joblib.load('thal_encoder.pkl')
age = int(input("Enter Age: "))
sex = int(input("Enter Sex (1=Male, 0=Female): "))
cp = int(input("Enter Chest Pain Type (0-4): "))
trestbps = int(input("Enter Resting Blood Pressure (mmHg): "))
chol = int(input("Enter Serum Cholesterol (mg/dl): "))
fbs = int(input("Enter Fasting Blood Sugar > 120mg/dl (1/0): "))
restecg = int(input("Enter Resting ECG Results (0-2): "))
thalach = int(input("Enter Maximum Heart Rate Achieved: "))
exang = int(input("Enter Exercise Induced Angina (1/0): "))
oldpeak = float(input("Enter ST Depression (oldpeak): "))
slope = int(input("Enter Slope of Peak ST Segment (1-3): "))
ca = int(input("Enter Number of Major Vessels (0-3): "))
thal = input("Enter Thal (normal/fixed/reversible): ")

input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal_encoder.transform([thal])[0]]],
    columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal_encoded'])

input_data_scaled = x_scaler.transform(input_data)
prediction_prob = model.predict(input_data_scaled)[0][0]
if prediction_prob >= 0.5:
    prediction = 1
else:
    prediction = 0

print("Prediction Probability:", prediction_prob)
if prediction == 1:
    print("Result: Heart Disease Detected")
else:
    print("Result: No Heart Disease")