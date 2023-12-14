import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# # Sample data for testing
# data = {
#     'Age': [30, 45, 55, 65, 25, 40, 50, 60, 35, 42],
#     'Sex': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],  # 1 for male, 0 for female
#     'BMI': [22, 26, 28, 31, 24, 28, 25, 30, 27, 29],
#     'BPM': [70, 80, 75, 85, 72, 72, 78, 68, 85, 76],
#     'HeartFailure': [0, 1, 1, 0, 0, 0, 1, 0, 1, 0]  # 1 for potential heart failure, 0 for no potential heart failure
# }

# df = pd.DataFrame(data)

# # Separate features and target variable
# X = df[['Age', 'Sex', 'BMI', 'BPM']]
# y = df['HeartFailure']

from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
heart_disease = fetch_ucirepo(id=45) 
  
# data (as pandas dataframes) 
X = heart_disease.data.features 
y = heart_disease.data.targets 


# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train a logistic regression model
model = LogisticRegression()
#model.fit(X, y)


def predict(new_data):
# Standardize the new data
    new_data_scaled = scaler.transform(new_data)

    # Make predictions
    predictions = model.predict(new_data_scaled)
    # Display predictions
    for i, prediction in enumerate(predictions):
        result = 'You have a potential Heart Failure. Please seek medical help immediately.' if prediction == 1 else 'Yay. You have no potential Heart Failure'
        print(f"Test case {i + 1}: {result}")

def main():
    print("Enter: <age> <sex> <Ca> <Cholesterol> <Resting BPS> <Max Heart Rate Achieved> :")
    new_data = pd.DataFrame({
        'age': [int(input())],
        'sex': [int(input())],
        'ca': [float(input())],
        'chol': [float(input())],
        'trestbps': [float(input())],
        'thalach': [int(input())]
    })
    predict(new_data)

if __name__ == "__main__":
    main()

    