import joblib
model = joblib.load("wind_turbine_power_model.pkl")
inp_wind_speed = int(input("Enter Wind Speed: "))
inp_blade_angle = int(input("Enter Blade Angle: "))
inp_rotor_speed = int(input("Enter Rotor Speed: "))
prediction = model.predict([[inp_wind_speed, inp_blade_angle, inp_rotor_speed]])
print("Predicted Power Output:", prediction[0])