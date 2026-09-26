import joblib
import pandas as pd
model = joblib.load("fraud_model.pkl")
df = pd.read_csv("data/creditcard.csv")

X = df.drop("Class", axis=1)

sample = X.iloc[0:1]

prediction = model.predict(sample)

if prediction[0] == 1:
    print("Prediction: FRAUD")
else:
    print("Prediction: NORMAL")