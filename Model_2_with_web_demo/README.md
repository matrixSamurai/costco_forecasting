# Model 2: Revenue Loss Prediction

Predicts how much money Costco loses when weather delays cause fresh lettuce to spoil during transit.

## What it does

- **XGBoost model** (10 features, ~50K training samples) predicts spoilage loss rate
- **Pipeline**: Model 1 delay output -> Model 2 loss prediction -> dollar amount
- **Supply Risk**: Assesses source region risk using USDA production + FEMA disaster data
- **Recommendations**: Rule engine generates scheduling, sourcing, and routing advice
- **Harvest Pattern**: Auto-fills source region and transit parameters based on month

## Quick Start

```bash
pip install -r requirements.txt
python3 app.py
# Open http://localhost:5001
```

## Project Structure

```
pipeline_s/
├── app.py                  # Flask web server (port 5001)
├── predict_loss.py         # Model 2 prediction interface
├── recommendations.py      # Supply Risk + Recommendations engine
├── pipeline.py             # Model 1 -> Model 2 end-to-end pipeline
├── build_loss_model.py     # Training data generation + model training
├── test_pipeline.py        # 23 test scenarios
├── loss_model_xgb.joblib   # Trained XGBoost model
├── training_data.csv       # 49,920 training samples
├── requirements.txt        # Python dependencies
├── data/
│   ├── supply_risk_lookup.csv          # Supply risk lookup (5 types x 12 months)
│   ├── lettuce_shelf_life_by_temp.csv  # Shelf life by temperature
│   ├── seasonal_price_indices.csv      # Seasonal price index
│   ├── sigmoid_parameters.csv          # Decay curve parameters
│   ├── usda_lettuce_production.csv     # USDA annual production (68 records)
│   ├── fema_disasters.csv              # FEMA disaster declarations (599 records)
│   ├── build_supply_index.py           # Build supply_risk_lookup.csv
│   ├── download_usda_data.py           # Download USDA data
│   └── download_fema_disasters.py      # Download FEMA data
└── frontend/
    ├── index.html
    └── static/
        ├── css/style.css
        └── js/loss.js
```

## Retrain the Model

```bash
python3 build_loss_model.py    # Generate data + train XGBoost
python3 test_pipeline.py       # Run 23 tests (all should pass)
```

## Rebuild Supply Risk Data

```bash
python3 data/download_usda_data.py
python3 data/download_fema_disasters.py
python3 data/build_supply_index.py
```

## Key Concepts

- **loss_rate**: Predicted spoilage rate (0-1) from XGBoost
- **revenue_loss**: quantity x price x loss_rate
- **consumption_pct**: Percentage of shelf life consumed. >100% means past shelf life
- **Supply Risk**: LOW / MEDIUM / HIGH based on regional supply + disaster frequency
- **Harvest Pattern**: Apr-Oct ships from Salinas, CA (200km); Nov-Mar from Yuma, AZ (800km)
