import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib
data = pd.read_csv("data.csv")
labelencoder = LabelEncoder()
data['diagnosis_encoded'] = labelencoder.fit_transform(data['diagnosis'])
x= data[["radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean","radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se","radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst"]]
y= data['diagnosis_encoded']
model = LogisticRegression()
model.fit(x,y)  
print(" coef =",model.coef_)
print("intercept =",model.intercept_)   




joblib.dump(model, 'model.pkl')
print("Model saved as model.pkl ")