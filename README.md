# 🏠 Indian House Price Prediction Web App

A Flask web application that predicts housing prices across major Indian cities using machine learning, featuring interactive visualizations and location-based analysis.

## ✨ Key Features

### 🎯 Core Functionality
- **Accurate Price Predictions** using Random Forest Regressor
- **Indian City Focus** with state-city dropdown selection
- **Comprehensive Features** including:
  - Property specifications (bedrooms, area, etc.)
  - Location attributes (schools, airport distance)
  - Quality metrics (condition, grade)

### 📊 Interactive Visualizations
- **Interactive Price Trend Charts** with seaborn
- **Historical Price Data** in sortable tables

### 🖥️ Modern UI/UX
- **Beautiful Dashboard** with cards and clean layout
- **Responsive Design** works on desktop and mobile
- **Intuitive Forms** with validation and helpful prompts
- **Professional Styling** with custom color scheme

## 🧠 Machine Learning Model

### 🔧 Model Specifications
- **Algorithm**: Random Forest Regressor
- **Key Parameters**:
  - `n_estimators=100`
  - `max_depth=20`
  - `random_state=42`
- **Feature Engineering**:
  - Bathrooms per bedroom ratio
  - Living area per floor
  - Lot to living area ratio
- **Evaluation Metric**: RMSE (Root Mean Squared Error)

### 📈 Data Processing Pipeline
1. Data loading and cleaning
2. Feature engineering
3. One-hot encoding for categorical features
4. Standard scaling of numerical features
5. Model training/persistence

## 🚀 Deployment Guide

### 📥 Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/deepakumar427/house-price-prediction.git
   cd house-price-prediction

2. Create and activate virtual environments:
   ```bash
   python -m venv env
   env\Scripts\activate   # For Windows
   source env\bin\activate   # For Mac

3. Install dependencies
   ```bash
   pip install -r requirements.txt
