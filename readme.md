## Repository Structure

```text
car-price-prediction-benchmark/
│
├── data/
│   └── car_data.csv                    # Dataset placeholder
│
├── src/
│   ├── predict_pricing_01_linear_regression.py         # Baseline Statistical Model
│   ├── predict_pricing_02_random_forest.py             # Ensemble Bagging Model
│   ├── predict_pricing_03_gradient_boosting.py         # Sequential Boosting Model
│   ├── predict_pricing_04_xgboost.py                   # Extreme Gradient Boosting Model
│   └── predict_pricing_05_svr.py                       # Support Vector Regressor Model
│
├── requirements.txt
└── README.md
