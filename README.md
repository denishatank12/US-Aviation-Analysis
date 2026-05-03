# US Aviation Analysis

Machine learning and visualization project on U.S. domestic flight delays and disruptions.

## Project structure

- `Code/` - cleaning, feature engineering, dataset building, model training, dashboard dataset generation
- `Data/raw/` - raw input files
- `Data/processed/` - cleaned and model-ready datasets
- `Models/` - trained model artifacts
- `Figures/` - ML result visualizations
- `Reports/` - summary reports and CSV outputs

## Pipeline

1. `python Code/cleaning.py`
2. `python Code/build_datasets.py`
3. `python Code/train_models.py`
4. `python Code/build_dashboard_data.py`

## Main project focus

- 3-class flight delay severity prediction
- Logistic Regression and Random Forest baselines
- ML-result visualizations
- dashboard-ready prediction outputs