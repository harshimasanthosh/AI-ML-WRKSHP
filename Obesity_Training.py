import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import joblib

data = pd.read_csv('ObesityDataSet.csv')

print(data.shape)
print(data.isnull().sum())
print(data['NObeyesdad'].value_counts())

data['NObeyesdad'].value_counts().plot(kind='bar')
plt.title("Obesity Level Distribution")
plt.tight_layout()
plt.savefig("class_distribution.png")
plt.close()

gender_encoder = LabelEncoder()
favc_encoder = LabelEncoder()
caec_encoder = LabelEncoder()
smoke_encoder = LabelEncoder()
scc_encoder = LabelEncoder()
calc_encoder = LabelEncoder()
mtrans_encoder = LabelEncoder()
family_encoder = LabelEncoder()
target_encoder = LabelEncoder()

data['Gender_encoded'] = gender_encoder.fit_transform(data['Gender'])
data['FAVC_encoded'] = favc_encoder.fit_transform(data['FAVC'])
data['CAEC_encoded'] = caec_encoder.fit_transform(data['CAEC'])
data['SMOKE_encoded'] = smoke_encoder.fit_transform(data['SMOKE'])
data['SCC_encoded'] = scc_encoder.fit_transform(data['SCC'])
data['CALC_encoded'] = calc_encoder.fit_transform(data['CALC'])
data['MTRANS_encoded'] = mtrans_encoder.fit_transform(data['MTRANS'])
data['family_history_encoded'] = family_encoder.fit_transform(data['family_history_with_overweight'])
data['NObeyesdad_encoded'] = target_encoder.fit_transform(data['NObeyesdad'])

x = data.drop(['Gender', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS',
               'family_history_with_overweight', 'NObeyesdad', 'NObeyesdad_encoded'], axis=1)
y = data['NObeyesdad_encoded']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_scaler = StandardScaler()
x_train = x_scaler.fit_transform(x_train)
x_test = x_scaler.transform(x_test)

num_classes = y.nunique()
y_train_cat = to_categorical(y_train, num_classes=num_classes)
y_test_cat = to_categorical(y_test, num_classes=num_classes)

model = Sequential([
    Dense(32, input_shape=(x_train.shape[1],), activation='relu'),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train_cat, epochs=50)
loss, accuracy = model.evaluate(x_test, y_test_cat, verbose=0)
print("Accuracy:", accuracy)
print("Loss:", loss)

y_prob = model.predict(x_test)
y_pred = y_prob.argmax(axis=1)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=target_encoder.classes_))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

joblib.dump(model, 'obesity_model.pkl')
joblib.dump(x_scaler, 'x_scaler.pkl')
joblib.dump(gender_encoder, 'gender_encoder.pkl')
joblib.dump(favc_encoder, 'favc_encoder.pkl')
joblib.dump(caec_encoder, 'caec_encoder.pkl')
joblib.dump(smoke_encoder, 'smoke_encoder.pkl')
joblib.dump(scc_encoder, 'scc_encoder.pkl')
joblib.dump(calc_encoder, 'calc_encoder.pkl')
joblib.dump(mtrans_encoder, 'mtrans_encoder.pkl')
joblib.dump(family_encoder, 'family_encoder.pkl')
joblib.dump(target_encoder, 'target_encoder.pkl')
print("Model, scaler, and encoders saved successfully.")
