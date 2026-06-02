import joblib
model = joblib.load('iris_model.pkl')
input_data = [[5.0,3.6,1.4,0.2] ,[6.1,2.8,4.0,1.3], [6.9,3.1,5.4,2.1]]
for sample in input_data:
    prediction = model.predict([sample])[0]
    print(f"Input: {sample} => Predicted Species: {prediction}")