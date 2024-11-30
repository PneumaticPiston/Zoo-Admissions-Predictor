import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Sample data generation function
def generate_zoo_visitor_data(num_samples=1000):
    """
    Generate synthetic data for zoo visitor prediction
    
    Columns:
    - temperature: float (in Celsius)
    - weather: categorical (Sunny, Cloudy, Rainy)
    - day_of_week: categorical (Monday, Tuesday, etc.)
    - event: boolean 
    - visitors: target variable (number of visitors)
    """
    np.random.seed(42)
    
    # Generate features
    temperature = np.random.normal(22, 5, num_samples)  # Mean 22°C, std dev 5°C
    weather_options = ['Sunny', 'Cloudy', 'Rainy']
    weather = np.random.choice(weather_options, num_samples)
    
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_of_week = np.random.choice(days, num_samples)
    
    event = np.random.choice([True, False], num_samples, p=[0.2, 0.8])
    
    # Create base visitor calculation with some randomness
    base_visitors = 500  # base daily visitors
    
    # Add impact of different features
    visitors = (
        base_visitors + 
        (temperature * 10) +  # More visitors on warmer days
        (weather == 'Sunny') * 200 +  # More visitors on sunny days
        (weather == 'Rainy') * -100 +  # Fewer visitors on rainy days
        (day_of_week.isin(['Saturday', 'Sunday'])) * 150 +  # More weekend visitors
        (event * 300)  # Significant boost for events
    )
    
    # Add some noise
    visitors += np.random.normal(0, 50, num_samples)
    visitors = np.maximum(visitors, 0).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'temperature': temperature,
        'weather': weather,
        'day_of_week': day_of_week,
        'event': event,
        'visitors': visitors
    })
    
    return df

# Load or generate data
data = generate_zoo_visitor_data()

# Preprocessing
# Separate features and target
X = data.drop('visitors', axis=1)
y = data['visitors']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['temperature']),
        ('cat', OneHotEncoder(drop='first', sparse=False), ['weather', 'day_of_week', 'event'])
    ])

# Create a pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Model Performance:")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R-squared Score: {r2:.2f}")

# Feature importance visualization
def plot_feature_importance(model, X):
    # Get feature names after preprocessing
    feature_names = (
        ['temperature'] + 
        [f'weather_{cat}' for cat in model.named_steps['preprocessor'].named_transformers_['cat'].categories_[0][1:]] +
        [f'day_of_week_{cat}' for cat in model.named_steps['preprocessor'].named_transformers_['cat'].categories_[1][1:]] +
        ['event_True']
    )
    
    # Get feature importances
    importances = model.named_steps['regressor'].feature_importances_
    
    # Create DataFrame for visualization
    feature_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_imp)
    plt.title('Feature Importances for Zoo Visitor Prediction')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.show()

# Plot feature importances
plot_feature_importance(model, X)

# Example prediction function
def predict_zoo_visitors(temperature, weather, day_of_week, event):
    """
    Make a prediction for zoo visitors
    
    :param temperature: Temperature in Celsius
    :param weather: 'Sunny', 'Cloudy', or 'Rainy'
    :param day_of_week: Day of the week
    :param event: Whether there's an event (True/False)
    :return: Predicted number of visitors
    """
    input_data = pd.DataFrame({
        'temperature': [temperature],
        'weather': [weather],
        'day_of_week': [day_of_week],
        'event': [event]
    })
    
    prediction = model.predict(input_data)
    return int(prediction[0])

# Example usage
print("\nExample Predictions:")
print("Sunny Saturday with event:", 
      predict_zoo_visitors(25, 'Sunny', 'Saturday', True))
print("Rainy Monday without event:", 
      predict_zoo_visitors(15, 'Rainy', 'Monday', False))
