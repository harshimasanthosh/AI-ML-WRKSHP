import joblib
model = joblib.load("model.pkl")
input_data = [[1,323,424,546,646,646,78,979,800,900,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000]]
prediction = model.predict(input_data)
print("Prediction for the input data:", prediction)