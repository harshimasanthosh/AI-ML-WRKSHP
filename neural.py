import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib

data = pd.read_csv('heart.csv')

thal_encoder = LabelEncoder()
data['thal_encoded'] = thal_encoder.fit_transform(data['thal'])
x = data.drop(['target', 'thal'], axis=1)
y = data['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_scaler = StandardScaler()
x_train = x_scaler.fit_transform(x_train)
x_test = x_scaler.transform(x_test)

model = Sequential([
    Dense(16, input_shape=(x_train.shape[1],), activation='relu'),
    Dense(8, activation='relu'),
    Dense(4, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=20)
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy:", accuracy)
print("Loss:", loss)

y_prob = (model.predict(x_test))
y_pred = (y_prob >= 0.5).astype(int)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

joblib.dump(model, 'nn_model.pkl')
joblib.dump(x_scaler, 'x_scaler.pkl')
joblib.dump(thal_encoder, 'thal_encoder.pkl')
print("Model, scaler, and encoder saved successfully.")