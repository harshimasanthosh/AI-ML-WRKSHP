import pandas as pd
import joblib

model = joblib.load('model.pkl')
x_scaler = joblib.load('x_scaler.pkl')
make_encoder = joblib.load('make.pkl')
class_encoder = joblib.load('class.pkl')
trans_encoder = joblib.load('trans.pkl')
fuel_encoder = joblib.load('fuel.pkl')

make = input("Enter Make: ")
vehicle_class = input("Enter Vehicle Class: ")
engine_size = float(input("Enter Engine Size (L): "))
cylinders = int(input("Enter Cylinders: "))
transmission = input("Enter Transmission: ")
fuel_type = input("Enter Fuel Type: ")
fuel_cons_city = float(input("Enter Fuel Consumption City (L/100km): "))
fuel_cons_hwy = float(input("Enter Fuel Consumption Hwy (L/100km): "))
fuel_cons_comb = float(input("Enter Fuel Consumption Combined (L/100km): "))

input_data = pd.DataFrame([[make, vehicle_class, engine_size, cylinders, transmission, fuel_type, fuel_cons_city, fuel_cons_hwy, fuel_cons_comb]],
                          columns=['Make', 'VehicleClass', 'EngineSize', 'Cylinders', 'Transmission', 'FuelType', 'FuelConsumptionCity', 'FuelConsumptionHwy', 'FuelConsumptionComb'])

input_data['Make'] = make_encoder.transform(input_data['Make'].str.title())
input_data['VehicleClass'] = class_encoder.transform(input_data['VehicleClass'].str.upper())
input_data['Transmission'] = trans_encoder.transform(input_data['Transmission'].str.upper())
input_data['FuelType'] = fuel_encoder.transform(input_data['FuelType'].str.upper())

input_scaled = x_scaler.transform(input_data)

prediction = model.predict(input_scaled)

print("Predicted CO2 Emissions:",prediction[0], "g/km")
if prediction[0] <= 200:
    print("Low emission vehicle")
elif prediction[0] <= 300:
    print("Moderate emission vehicle")
else:
    print("High emission vehicle")
