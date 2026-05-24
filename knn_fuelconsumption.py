import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
import math
data = pd.read_csv('KNN_Dataset.csv')
x = data[["Temperature"]]
y = data[["Fuel_Consumption"]]


model=KNeighborsRegressor(n_neighbors=3)
model.fit(x,y)

temp_1 = 58
y_pred = model.predict([[temp_1]])
print(f"Predicted fuel consumption for temperature {temp_1} is: {y_pred[0][0]}")


temp = int(input("Enter a temperature to predict fuel consumption: "))
y_pred = model.predict([[temp]])
print(f"Predicted fuel consumption for temperature {temp} is: {y_pred[0][0]}")