

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv("data.csv")

x = data[["Load (N)"]]
y = data[["Extension (mm)"]]

plt.scatter(x,y)
plt.show()
model = LinearRegression()
model.fit(x,y)
print("Coeff: ",model.coef_)
print ("Intercept: ",model.intercept_)

inp_user = int(input("Enter a Load (N) ? "))
new_extension_mm = model.predict([[inp_user]])
print(new_extension_mm)