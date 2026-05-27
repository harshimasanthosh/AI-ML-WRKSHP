import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

data=pd.read_csv("machine_failure.csv")
label_encoder=LabelEncoder()
df=data.drop(["UDI","Product ID"], axis=1 )
data["type_encoded"]=label_encoder.fit_transform(data["Type"])
x=data[["type_encoded","Air temperature [K]","Process temperature [K]","Rotational speed [rpm]","Torque [Nm]","Tool wear [min]","TWF","HDF","PWF","OSF","RNF"]]
y=data[["Machine failure"]]

model=LogisticRegression()
model.fit(x,y)
type_new_enc=label_encoder.transform(["M"])[0]
output=model.predict([[type_new_enc,301,311,1450,52,28,1,0,0,0,0]])
print("Predicted result for the sample input=",output)
if output==1:
    print("The machine will fail")
else:
    print("The machine will not fail")