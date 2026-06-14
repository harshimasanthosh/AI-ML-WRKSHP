import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

data = pd.read_csv("co2_data.csv")

print("Dataset Shape:", data.shape)
print(data.head())

data = data.drop_duplicates()
print("Shape after dropping duplicates:", data.shape)

make_encoder = LabelEncoder()
data['Make'] = make_encoder.fit_transform(data['Make'])
class_encoder = LabelEncoder()
data['VehicleClass'] = class_encoder.fit_transform(data['VehicleClass'])
trans_encoder = LabelEncoder()
data['Transmission'] = trans_encoder.fit_transform(data['Transmission'])
fuel_encoder = LabelEncoder()
data['FuelType'] = fuel_encoder.fit_transform(data['FuelType'])

X = data.drop('CO2Emissions', axis=1)
y = data['CO2Emissions']

x_scaler = StandardScaler()
X_scaled = x_scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R-squared Score:", r2)

joblib.dump(model, 'model.pkl')
joblib.dump(x_scaler, 'x_scaler.pkl')
joblib.dump(make_encoder, 'make.pkl')
joblib.dump(class_encoder, 'class.pkl')
joblib.dump(trans_encoder, 'trans.pkl')
joblib.dump(fuel_encoder, 'fuel.pkl')
print("Model, scaler and label encoders saved.")
