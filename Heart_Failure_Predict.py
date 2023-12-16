import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from ucimlrepo import fetch_ucirepo

# fetch dataset
heart_disease = fetch_ucirepo(id=45)

X = heart_disease.data.features
y = heart_disease.data.targets

imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

model = LogisticRegression()
model.fit(X_scaled, y)

def predict(new_data):
    new_data_with_columns = pd.DataFrame(new_data, columns=X.columns)
    new_data_imputed = imputer.transform(new_data_with_columns)
    new_data_scaled = scaler.transform(new_data_imputed)
    predictions = model.predict(new_data_scaled)
    for i, prediction in enumerate(predictions):
        result = 'You have a potential Heart Failure. Please seek medical help immediately.' if prediction == 1 else 'Yay. You have no potential for Heart Failure'
    return result
