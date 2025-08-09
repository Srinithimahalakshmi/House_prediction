# House Price Prediction Model 🏠💰

![Real Estate Prediction](https://img.shields.io/badge/domain-real_estate-blue) ![Machine Learning](https://img.shields.io/badge/ML-Regression-green) ![Python](https://img.shields.io/badge/Python-3.8%2B-yellow)

Machine learning system for predicting residential property prices using features like location, size, amenities, and market trends. Includes EDA, feature engineering, and multiple regression models.

## Features ✨
- Comprehensive data preprocessing pipeline
- Advanced feature engineering for real estate data
- Multiple regression models (XGBoost, Random Forest, Gradient Boosting)
- Hyperparameter tuning with Optuna
- Interactive price prediction interface
- SHAP value interpretation for predictions

## Installation 💻

### Prerequisites
- Python 3.8+
- pip package manager

### Setup
```bash
# Clone repository
git clone https://github.com/Srinithimahalakshmi/House_prediction.git
cd House_prediction

# Create virtual environment
python -m venv house_env
source house_env/bin/activate  # Linux/Mac
house_env\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
Dataset 📊
Real Estate Dataset - Contains property features and historical prices

Located in data/ directory

Features:

Location (latitude, longitude, neighborhood)

Property size (sqft, rooms, bathrooms)

Amenities (pool, garage, garden)

Year built and renovation status

Market conditions at time of sale

Usage 🚀
1. Data Preparation
bash
python src/data_preprocessing.py
2. Model Training
bash
# Train XGBoost model
python src/models/train_xgb.py

# Train Random Forest model
python src/models/train_rf.py
3. Make Predictions
python
from src.predict import HousePricePredictor

# Initialize predictor
predictor = HousePricePredictor('models/xgboost_model.pkl')

# Sample property features
property_data = {
    'bedrooms': 3,
    'bathrooms': 2.5,
    'sqft_living': 2150,
    'location': 'Downtown',
    'year_built': 2010,
    'waterfront': 0
}

# Get price prediction
prediction = predictor.predict(property_data)
print(f"Predicted price: ${prediction:,.2f}")
4. Start Web Interface (Flask)
bash
python app.py  # Access at http://localhost:5000
Model Performance 📈
Model	MAE	RMSE	R² Score
XGBoost	$42,150	$68,900	0.912
Random Forest	$45,780	$72,500	0.902
Gradient Boost	$43,950	$70,100	0.908
https://results/price_comparison.png <!-- Add actual path -->

Key Features Impacting Price
Location (geographic coordinates)

Living area square footage

Number of bathrooms

Proximity to city center

Property age and condition

Repository Structure 📂
text
├── data/                   # Raw and processed datasets
│   ├── raw/                # Original datasets
│   └── processed/          # Cleaned data
│
├── models/                 # Trained model files
│   ├── xgboost_model.pkl
│   └── random_forest.pkl
│
├── src/                    # Source code
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── models/             # Model training scripts
│   │   ├── train_xgb.py
│   │   ├── train_rf.py
│   │   └── model_evaluation.py
│   └── predict.py          # Prediction functions
│
├── notebooks/              # Jupyter notebooks
│   ├── 01_EDA.ipynb
│   ├── 02_Feature_Importance.ipynb
│   └── 03_Model_Comparison.ipynb
│
├── web_app/                # Flask application
│   ├── templates/          
│   │   └── index.html
│   ├── static/
│   └── app.py
│
├── results/                # Evaluation metrics and plots
├── requirements.txt        # Python dependencies
└── LICENSE
How It Works 🧠
Data Ingestion: Load property data from CSV

Feature Engineering:

Location clustering

Age-based value depreciation

Amenity score calculation

Model Training:

Hyperparameter tuning with cross-validation

Ensemble model creation

Prediction:

Generate price estimates

Explain predictions with SHAP values

Try the Web Interface 🌐
https://web_app/static/screenshot.png <!-- Add actual path -->
Access the prediction form at http://localhost:5000 after starting the Flask app

Business Applications 💼
Real estate valuation for buyers/sellers

Investment property analysis

Automated property appraisal

Market trend analysis

Mortgage risk assessment

Contributors 👥
Srinithi Mahalakshmi
https://img.shields.io/badge/LinkedIn-Connect-blue
https://img.shields.io/badge/GitHub-Follow-lightgrey
