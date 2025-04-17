import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pickle

# Model and scaler file paths
MODEL_FILE = 'house_price_model.pkl'
SCALER_FILE = 'scaler.pkl'
FEATURES_FILE = 'feature_columns.pkl'

def train_and_save_model():
    """Train and save the model with all required components"""
    # Load your dataset
    df = pd.read_csv("housing_data.csv") 

    # Feature Engineering
    df['bathrooms_per_bedroom'] = df['number of bathrooms'] / df['number of bedrooms']
    df['living_area_per_floor'] = df['living area'] / df['number of floors']
    df['lot_to_living_ratio'] = df['lot area'] / df['living area']

    # One-hot encoding
    df = pd.get_dummies(df, columns=['condition of the house', 'grade of the house'], drop_first=True)

    # Select features and target
    features = [
        'number of bedrooms', 'number of bathrooms', 'living area', 'lot area',
        'number of floors', 'waterfront present', 'number of views',
        'Area of the house(excluding basement)', 'Area of the basement',
        'Built Year', 'Renovation Year', 'living_area_renov', 'lot_area_renov',
        'Number of schools nearby', 'Distance from the airport', 'Lattitude', 'Longitude',
        'bathrooms_per_bedroom', 'living_area_per_floor', 'lot_to_living_ratio'
    ] + [col for col in df.columns if col.startswith(('condition of the house_', 'grade of the house_'))]

    X = df[features]
    y = df['Price']

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)

    # Save all components
    pickle.dump(model, open(MODEL_FILE, 'wb'))
    pickle.dump(scaler, open(SCALER_FILE, 'wb'))
    pickle.dump(features, open(FEATURES_FILE, 'wb'))

    return model, scaler, features

train_and_save_model()