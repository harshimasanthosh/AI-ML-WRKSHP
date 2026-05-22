import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

data = pd.read_csv("C:\\Users\\HP\\Downloads\\python prgm\\updated_power_data.csv")
x = data[["Wind_Speed","Blade_Angle","Rotor_Speed"]]
y = data["Power_Output"]

model = LinearRegression()
model.fit(x, y)
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

joblib.dump(model, 'wind_turbine_power_model.pkl')