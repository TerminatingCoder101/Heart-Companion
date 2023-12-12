import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Sample data for testing
data = {
    'Age': [30, 45, 55, 65, 25],
    'Sex': [1, 0, 1, 0, 1],  # 1 for male, 0 for female
    'BMI': [22, 26, 28, 31, 24],
    'BPM': [70, 80, 75, 85, 72],
    'HeartFailure': [0, 1, 1, 0, 0]  # 1 for potential heart failure, 0 for no potential heart failure
}

df = pd.DataFrame(data)

# Separate features and target variable
X = df[['Age', 'Sex', 'BMI', 'BPM']]
y = df['HeartFailure']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_scaled, y)

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
    print("Enter: <age> <sex> <bmi> <bpm>:")
    new_data = pd.DataFrame({
        'Age': [int(input())],
        'Sex': [int(input())],
        'BMI': [float(input())],
        'BPM': [float(input())]
    })
    predict(new_data)

    

