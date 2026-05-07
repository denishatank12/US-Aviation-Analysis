✈️ US Aviation Analysis

A machine learning and data visualization project focused on understanding and predicting U.S. domestic flight delays and disruptions.

This project explores patterns in flight performance and builds predictive models to classify delay severity, with an emphasis on producing insights that are both analytical and dashboard-ready.

📊 Dataset

The data used in this project comes from Kaggle:

👉 https://www.kaggle.com/datasets/patrickzel/flight-delay-and-cancellation-dataset-2019-2023?select=flights_sample_3m.csv

It includes U.S. domestic flight records from 2019–2023, covering delays, cancellations, and operational details.

📁 Project Structure
Code/                # Data cleaning, feature engineering, model training, and pipeline scripts  
Data/raw/            # Original raw datasets  
Data/processed/      # Cleaned and model-ready datasets  
Models/              # Saved trained model artifacts  
Figures/             # Visualizations of ML results and insights  
Reports/             # Summary outputs, reports, and CSV exports  


⚙️ Pipeline

Run the project step-by-step using the following scripts:
python Code/cleaning.py
python Code/build_datasets.py
python Code/train_models.py
python Code/build_dashboard_data.py

🎯 Main Focus
Flight Delay Severity Prediction
3-class classification problem (e.g., on-time, moderate delay, severe delay)
Baseline Models
Logistic Regression
Random Forest
Data Visualization
Model performance plots
Delay distribution insights
Feature importance analysis
Dashboard-Ready Outputs

🚀 What This Project Demonstrates
End-to-end ML pipeline (data → features → models → outputs)
Real-world dataset handling (messy aviation data)
Model comparison and evaluation
Translating ML results into usable analytics