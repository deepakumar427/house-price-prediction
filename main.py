# House Price Prediction with Geospatial Data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# Try importing sklearn
try:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error
    from sklearn.preprocessing import StandardScaler
except ModuleNotFoundError as e:
    raise ImportError(
        "scikit-learn is required to run this code. Please install it using:\n\npip install scikit-learn"
    ) from e

# Try importing folium for map
try:
    import folium
    has_folium = True
except ImportError:
    print("folium not installed. Skipping map generation.")
    has_folium = False

# Load dataset
df = pd.read_csv("https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv")

# Data Cleaning
df['total_bedrooms'].fillna(df['total_bedrooms'].median(), inplace=True)
df = pd.get_dummies(df, columns=['ocean_proximity'], drop_first=True)

# Feature Engineering
df['rooms_per_household'] = df['total_rooms'] / df['households']
df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']
df['population_per_household'] = df['population'] / df['households']

# Define features and target
X = df.drop('median_house_value', axis=1)
y = df['median_house_value']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("✅ Model Evaluation:")
print("RMSE:", rmse)

# Optional: Geospatial Visualization
if has_folium:
    california_map = folium.Map(location=[36.7783, -119.4179], zoom_start=6)
    for i in range(min(100, len(df))):
        folium.CircleMarker(
            location=[df.iloc[i]['latitude'], df.iloc[i]['longitude']],
            radius=5,
            popup=f"Price: ${df.iloc[i]['median_house_value']}",
            color='blue',
            fill=True,
            fill_opacity=0.6
        ).add_to(california_map)

    california_map.save("house_price_map.html")
    print("🗺️ Map saved to house_price_map.html")

# 🧮 Predict House Price from User Input
def predict_house_price():
    print("\n📍 Enter house details to predict price:")
    latitude = float(input("Latitude: "))
    longitude = float(input("Longitude: "))
    housing_median_age = float(input("Housing median age: "))
    total_rooms = float(input("Total rooms: "))
    total_bedrooms = float(input("Total bedrooms: "))
    population = float(input("Population: "))
    households = float(input("Households: "))
    median_income = float(input("Median income: "))
    ocean = input("Ocean proximity (options: NEAR BAY, INLAND, NEAR OCEAN, ISLAND): ").strip().upper()

    # Handle one-hot encoding (use the same column names as during training)
    ocean_proximity_INLAND = 1 if ocean == "INLAND" else 0
    ocean_proximity_ISLAND = 1 if ocean == "ISLAND" else 0
    ocean_proximity_NEAR_OCEAN = 1 if ocean == "NEAR OCEAN" else 0
    ocean_proximity_NEAR_BAY = 1 if ocean == "NEAR BAY" else 0

    # Derived features
    rooms_per_household = total_rooms / households
    bedrooms_per_room = total_bedrooms / total_rooms
    population_per_household = population / households

    # Create a DataFrame with the new input
    input_dict = {
        'longitude': longitude,
        'latitude': latitude,
        'housing_median_age': housing_median_age,
        'total_rooms': total_rooms,
        'total_bedrooms': total_bedrooms,
        'population': population,
        'households': households,
        'median_income': median_income,
        'ocean_proximity_INLAND': ocean_proximity_INLAND,
        'ocean_proximity_ISLAND': ocean_proximity_ISLAND,
        'ocean_proximity_NEAR_OCEAN': ocean_proximity_NEAR_OCEAN,
        'ocean_proximity_NEAR_BAY': ocean_proximity_NEAR_BAY,
        'rooms_per_household': rooms_per_household,
        'bedrooms_per_room': bedrooms_per_room,
        'population_per_household': population_per_household
    }

    input_df = pd.DataFrame([input_dict])

    # Reorder columns to match the training data columns
    input_df = input_df.reindex(columns=X.columns, fill_value=0)

    # Scale input using the same scaler used during training
    input_scaled = scaler.transform(input_df)

    # Predict the house price
    predicted_price = model.predict(input_scaled)[0]

    print(f"\n💰 Predicted House Price: ${predicted_price:,.2f}")

# Run the prediction function
predict_house_price()