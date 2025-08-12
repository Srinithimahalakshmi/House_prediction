
#  House Price Prediction

##  Overview
A machine learning system for estimating residential property prices using features like **location**, **size**, **number of rooms**, and **amenities**. Includes data preprocessing, feature engineering, model training (e.g., XGBoost, Random Forest, Gradient Boosting), hyperparameter tuning, and an interactive Flask-based interface with SHAP explainability.

---

##  Table of Contents
- [⚙️ Installation](#-installation)  
- [🚀 Usage](#-usage)  
- [📁 Project Structure](#-project-structure)  
- [📊 Results](#-results)  
- [🤝 Contributing](#-contributing)  
- [📬 Contact](#-contact)  

---

##  Installation

Prerequisites:
- Python 3.8+
- pip

```bash
git clone https://github.com/Srinithimahalakshmi/House_prediction.git
cd House_prediction

# Set up virtual environment
python3 -m venv house_env
source house_env/bin/activate   # Windows: house_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
````

---

## Usage

### 1. Data Preparation

```bash
python src/data_preprocessing.py
```

### 2. Train Models

```bash
python src/models/train_xgb.py       # XGBoost
python src/models/train_rf.py        # Random Forest
```

### 3. Predictions in Code

```python
from src.predict import HousePricePredictor

predictor = HousePricePredictor('models/xgboost_model.pkl')
property_data = {
    'bedrooms': 3,
    'bathrooms': 2.5,
    'sqft_living': 2150,
    'location': 'Downtown',
    'year_built': 2010,
    'waterfront': 0
}
prediction = predictor.predict(property_data)
print(f"Predicted price: ${prediction:,.2f}")
```

### 4. Launch Web Interface

```bash
python app.py
```

Access the app at **[http://localhost:5000](http://localhost:5000)** to interactively get price estimates and explanations.

---

## Project Structure

```
House_prediction/
├── data/
│   ├── raw/                  # Original datasets
│   └── processed/            # Cleaned data for training
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── models/
│   │   ├── train_xgb.py
│   │   ├── train_rf.py
│   │   └── model_evaluation.py
│   └── predict.py
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Feature_Importance.ipynb
│   └── 03_Model_Comparison.ipynb
├── web_app/
│   ├── app.py
│   ├── templates/
│   │   └── index.html
│   └── static/
├── models/
│   ├── xgboost_model.pkl
│   └── random_forest.pkl
├── results/
│   └── price_comparison.png
├── requirements.txt
└── README.md
```

---

## Results

| Model          | MAE       | RMSE      | R² Score |
| -------------- | --------- | --------- | -------- |
| XGBoost        | \~\$4,150 | \~\$6,900 | 0.912    |
| Random Forest  | \~\$4,580 | \~\$7,250 | 0.902    |
| Gradient Boost | \~\$4,395 | \~\$7,010 | 0.908    |

Key prediction drivers include **location**, **living area**, **bathrooms**, proximity to the city center, and property age. SHAP-based explanations help interpret feature impact.

---

## Contributing

Contributions are welcome! You may:

* Refine feature engineering or introduce new features
* Experiment with alternative models or ensembles
* Enhance model interpretability or UI visuals
* Improve web app interactivity or experience
* Add model comparison dashboards or CLI options

To contribute: fork → branch (`feature/YourFeature`) → commit → push → PR

---

## Contact

👤 **Maintainer**: Srinithi Mahalakshmi
📧 **Email**: [srinithiarumugam2003@gmail.com](mailto:srinithiarumugam2003@gmail.com)
🔗 **GitHub**: [Srinithimahalakshmi](https://github.com/Srinithimahalakshmi)

---

⭐ *If this project has helped you, your star would be greatly appreciated!*

```
