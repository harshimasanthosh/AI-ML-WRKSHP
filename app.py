from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load model and encoders
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

# Home page route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        age = float(data['age'])
        gender = data['gender']
        height = float(data['height'])
        weight = float(data['weight'])
        favc = data['favc']
        fcvc = float(data['fcvc'])
        ncp = float(data['ncp'])
        caec = data['caec']
        smoke = data['smoke']
        ch2o = float(data['ch2o'])
        scc = data['scc']
        faf = float(data['faf'])
        tue = float(data['tue'])
        calc = data['calc']
        mtrans = data['mtrans']
        family_history = data['family_history']

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
        ]], columns=[
            'Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE',
            'Gender_encoded', 'FAVC_encoded', 'CAEC_encoded', 'SMOKE_encoded',
            'SCC_encoded', 'CALC_encoded', 'MTRANS_encoded', 'family_history_encoded'
        ])

        input_data_scaled = x_scaler.transform(input_data)
        prediction_prob = model.predict(input_data_scaled)[0]
        prediction = prediction_prob.argmax()
        result = target_encoder.inverse_transform([prediction])[0]

        return jsonify({
            "prediction": result,
            "confidence": float(prediction_prob[prediction])
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print("Flask app is starting...")
    app.run(host='127.0.0.1', port=5000, debug=True)

