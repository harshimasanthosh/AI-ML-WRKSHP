import pandas as pd
import joblib

model = joblib.load('obesity_model.pkl')
x_scaler = joblib.load('x_scaler.pkl')
gender_encoder = joblib.load('gender_encoder.pkl')
favc_encoder = joblib.load('favc_encoder.pkl')
caec_encoder = joblib.load('caec_encoder.pkl')
smoke_encoder = joblib.load('smoke_encoder.pkl')
scc_encoder = joblib.load('scc_encoder.pkl')
calc_encoder = joblib.load('calc_encoder.pkl')
mtrans_encoder = joblib.load('mtrans_encoder.pkl')
family_encoder = joblib.load('family_encoder.pkl')
target_encoder = joblib.load('target_encoder.pkl')

age = float(input("Enter Age: "))
gender = input("Enter Gender (Male/Female): ")
height = float(input("Enter Height (m): "))
weight = float(input("Enter Weight (kg): "))
favc = input("Frequent High Caloric Food (yes/no): ")
fcvc = float(input("Vegetable Consumption Frequency (1-3): "))
ncp = float(input("Number of Main Meals (1-4): "))
caec = input("Eating Between Meals (no/Sometimes/Frequently/Always): ")
smoke = input("Do You Smoke (yes/no): ")
ch2o = float(input("Daily Water Intake (1-3): "))
scc = input("Monitor Calorie Consumption (yes/no): ")
faf = float(input("Physical Activity Frequency (0-3): "))
tue = float(input("Time Using Technology Devices (0-2): "))
calc = input("Alcohol Consumption (no/Sometimes/Frequently/Always): ")
mtrans = input("Transportation Used (Public_Transportation/Walking/Automobile/Motorbike/Bike): ")
family_history = input("Family History With Overweight (yes/no): ")

input_data = pd.DataFrame([[
    age, height, weight, fcvc, ncp, ch2o, faf, tue,
    gender_encoder.transform([gender])[0],
    favc_encoder.transform([favc])[0],
    caec_encoder.transform([caec])[0],
    smoke_encoder.transform([smoke])[0],
    scc_encoder.transform([scc])[0],
    calc_encoder.transform([calc])[0],
    mtrans_encoder.transform([mtrans])[0],
    family_encoder.transform([family_history])[0]
]], columns=['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE',
             'Gender_encoded', 'FAVC_encoded', 'CAEC_encoded', 'SMOKE_encoded',
             'SCC_encoded', 'CALC_encoded', 'MTRANS_encoded', 'family_history_encoded'])

input_data_scaled = x_scaler.transform(input_data)
prediction_prob = model.predict(input_data_scaled)[0]
prediction = prediction_prob.argmax()
result = target_encoder.inverse_transform([prediction])[0]

print("Prediction Probabilities:", prediction_prob)
print("Predicted Obesity Level:", result)
