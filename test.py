import pandas as pd
import joblib

# Load your trained RandomForestClassifier model from the .pkl file
model = joblib.load('weather.pkl')
a=input("Enter Precipitation:")
b=input("Enter Max Temp:")
c=input("Enter Min Temp:")
d=input("Enter wind speed:")

# Input features (precipitation, temp_max, temp_min, wind)
input_features = {
    'precipitation': a,
    'temp_max': b,
    'temp_min': c,
    'wind': d
}

# Create a DataFrame from the input data
input_data = pd.DataFrame([input_features])

# Make predictions on the input data
predicted_weather = model.predict(input_data)

# Map the predicted label to the corresponding weather condition
weather_conditions = {
    'drizzle': 'Drizzle',
    'fog': 'Fog',
    'rain': 'Rain',
    'snow': 'Snow',
    'sun': 'Sunny'
}

predicted_condition = weather_conditions.get(predicted_weather[0])

print(f'Predicted Weather Condition: {predicted_condition}')
