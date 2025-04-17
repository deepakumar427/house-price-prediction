# app.py
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pickle
import os
from io import BytesIO
import base64
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import datetime

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# Initialize geocoder
geolocator = Nominatim(user_agent="indian_house_price_prediction")

# Enhanced Indian locations data with popular cities
indian_locations = {
    "Maharashtra": ["Mumbai", "Pune", "Nagpur", "Nashik", "Thane"],
    "Karnataka": ["Bangalore", "Mysore", "Hubli", "Mangalore"],
    "Tamil Nadu": ["Chennai", "Coimbatore", "Madurai", "Tiruchirappalli"],
    "Delhi": ["New Delhi"],
    "West Bengal": ["Kolkata", "Howrah", "Durgapur"],
    "Gujarat": ["Ahmedabad", "Surat", "Vadodara", "Rajkot"],
    "Rajasthan": ["Jaipur", "Udaipur", "Jodhpur", "Kota"],
    "Telangana": ["Hyderabad", "Warangal"],
    "Uttar Pradesh": ["Lucknow", "Kanpur", "Varanasi", "Agra"]
}

# Model files
MODEL_DIR = 'model_data'
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_FILE = os.path.join(MODEL_DIR, 'house_price_model.pkl')
SCALER_FILE = os.path.join(MODEL_DIR, 'scaler.pkl')
FEATURES_FILE = os.path.join(MODEL_DIR, 'feature_columns.pkl')

def train_and_save_model():
    """Train and save the model with all required components"""
    try:
        # Load your dataset
        df = pd.read_csv("housing_data.csv")  # Replace with your actual file path

        # Feature Engineering
        df['bathrooms_per_bedroom'] = df['number of bathrooms'] / df['number of bedrooms']
        df['living_area_per_floor'] = df['living area'] / df['number of floors']
        df['lot_to_living_ratio'] = df['lot area'] / df['living area']

        # One-hot encoding
        df = pd.get_dummies(df, columns=['condition of the house', 'grade of the house'], drop_first=True)

        # Select features and target
        base_features = [
            'number of bedrooms', 'number of bathrooms', 'living area', 'lot area',
            'number of floors', 'waterfront present', 'number of views',
            'Area of the house(excluding basement)', 'Area of the basement',
            'Built Year', 'Renovation Year', 'living_area_renov', 'lot_area_renov',
            'Number of schools nearby', 'Distance from the airport', 'Lattitude', 'Longitude',
            'bathrooms_per_bedroom', 'living_area_per_floor', 'lot_to_living_ratio'
        ]
        encoded_features = [col for col in df.columns if col.startswith(('condition of the house_', 'grade of the house_'))]
        feature_columns = base_features + encoded_features

        X = df[feature_columns]
        y = df['Price']

        # Feature scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train model
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_scaled, y)

        # Save all components
        pickle.dump(model, open(MODEL_FILE, 'wb'))
        pickle.dump(scaler, open(SCALER_FILE, 'wb'))
        pickle.dump(feature_columns, open(FEATURES_FILE, 'wb'))

        return model, scaler, feature_columns

    except Exception as e:
        print(f"Error during model training: {str(e)}")
        raise

# Load or train model
try:
    if not os.path.exists(MODEL_FILE):
        print("Training model...")
        model, scaler, feature_columns = train_and_save_model()
        print("Model training completed successfully!")
    else:
        model = pickle.load(open(MODEL_FILE, 'rb'))
        scaler = pickle.load(open(SCALER_FILE, 'rb'))
        feature_columns = pickle.load(open(FEATURES_FILE, 'rb'))
        print("Model loaded successfully!")
except Exception as e:
    print(f"Failed to load/train model: {str(e)}")
    exit(1)

def get_coordinates(city, state):
    """Get coordinates for Indian city with state context"""
    try:
        location = geolocator.geocode(f"{city}, {state}, India", timeout=10)
        if location:
            return location.latitude, location.longitude
        return None, None
    except (GeocoderTimedOut, GeocoderUnavailable) as e:
        print(f"Geocoding error for {city}, {state}: {str(e)}")
        return None, None

def generate_price_trend_plot(city):
    """Generate a realistic price trend plot with around 100 monthly data points"""
    
    np.random.seed(42)  # for reproducibility
    
    months = pd.date_range(start='2015-01', periods=100, freq='M')
    
    base_prices = {
        'Mumbai': 15000,
        'Bangalore': 8000,
        'Chennai': 6000,
        'Pune': 7000,
        'Kolkata': 5000,
        'New Delhi': 9000
    }
    
    start_price = base_prices.get(city, 5000)
    
    # Simulate monthly percentage changes: small fluctuations with slight upward trend
    monthly_growth = np.random.normal(loc=0.0025, scale=0.01, size=100)
    price_series = [start_price]
    
    for growth in monthly_growth:
        new_price = price_series[-1] * (1 + growth)
        price_series.append(round(new_price, 2))
    
    price_series = price_series[1:]  # drop the initial base price
    
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(months, price_series, marker='.', linestyle='-', linewidth=1)
    plt.title(f'Price per sqft Trend in {city} (Monthly)')
    plt.xlabel('Month')
    plt.ylabel('Price per sqft (₹)')
    plt.grid(True)
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()

    return plot_data, list(zip(months.strftime('%Y-%m'), price_series))

@app.route('/')
def home():
    return render_template('index.html', 
                         states=list(indian_locations.keys()),
                         current_year=datetime.datetime.now().year)

@app.route('/get_cities/<state>')
def get_cities(state):
    return jsonify(indian_locations.get(state, []))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        state = request.form['state']
        city = request.form['city']
        
        # Get coordinates
        lat, lon = get_coordinates(city, state)
        if lat is None or lon is None:
            return render_template('index.html', 
                                states=list(indian_locations.keys()),
                                error="Could not find coordinates for this location. Please try another city.")
        
        # Create feature dictionary from form data
        features = {
            'number of bedrooms': float(request.form['bedrooms']),
            'number of bathrooms': float(request.form['bathrooms']),
            'living area': float(request.form['living_area']),
            'lot area': float(request.form['lot_area']),
            'number of floors': float(request.form['floors']),
            'waterfront present': int(request.form.get('waterfront', 0)),
            'number of views': int(request.form['views']),
            'Area of the house(excluding basement)': float(request.form['house_area']),
            'Area of the basement': float(request.form['basement_area']),
            'Built Year': int(request.form['built_year']),
            'Renovation Year': int(request.form.get('renovation_year', 0)),
            'living_area_renov': float(request.form.get('living_area_renov', request.form['living_area'])),
            'lot_area_renov': float(request.form.get('lot_area_renov', request.form['lot_area'])),
            'Number of schools nearby': int(request.form['schools']),
            'Distance from the airport': float(request.form['airport_distance']),
            'condition of the house': int(request.form['condition']),
            'grade of the house': int(request.form['grade']),
            'Lattitude': lat,
            'Longitude': lon
        }
        
        # Feature engineering
        features['bathrooms_per_bedroom'] = features['number of bathrooms'] / features['number of bedrooms']
        features['living_area_per_floor'] = features['living area'] / features['number of floors']
        features['lot_to_living_ratio'] = features['lot area'] / features['living area']
        
        # Create input DataFrame
        input_data = {**features}
        input_df = pd.DataFrame([input_data]).reindex(columns=feature_columns, fill_value=0)
        
        # Set one-hot encoded columns
        condition_col = f'condition of the house_{features["condition of the house"]}'
        grade_col = f'grade of the house_{features["grade of the house"]}'
        
        if condition_col in input_df.columns:
            input_df[condition_col] = 1
        if grade_col in input_df.columns:
            input_df[grade_col] = 1
        
        # Scale and predict
        input_scaled = scaler.transform(input_df)
        predicted_price = model.predict(input_scaled)[0]
        
        # Generate price trend plot
        plot_url, price_data = generate_price_trend_plot(city)
        
        return render_template('result.html',
                             city=city,
                             state=state,
                             predicted_price=f"₹{predicted_price:,.0f}",
                             plot_url=plot_url,
                             price_data=price_data,
                             coordinates=f"{lat:.4f}, {lon:.4f}",
                             current_year=datetime.datetime.now().year)

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return render_template('index.html', 
                            states=list(indian_locations.keys()),
                            error="An error occurred during prediction. Please check your inputs and try again.")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)