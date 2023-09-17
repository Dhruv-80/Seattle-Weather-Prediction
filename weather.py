import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load your weather dataset (replace 'your_dataset.csv' with the actual file path)
data = pd.read_csv('seattle-weather.csv')

# Select the relevant columns for prediction and the target variable
selected_features = ['precipitation', 'temp_max', 'temp_min', 'wind']
target_variable = 'weather'

# Prepare the feature matrix (X) and target variable (y)
X = data[selected_features]
y = data[target_variable]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)

# Calculate and print the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Generate a classification report for more detailed evaluation
report = classification_report(y_test, y_pred)
print('Classification Report:\n', report)

# Save the trained model to a .pkl file
joblib.dump(clf, 'weather.pkl')
