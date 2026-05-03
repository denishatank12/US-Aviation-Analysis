import pandas as pd

from project_config import CLEANED_DATA, MODEL_READY_3CLASS
from feature_engineering import build_model_ready_3class

def main():
    df = pd.read_csv(CLEANED_DATA, low_memory=False)
    model_df = build_model_ready_3class(df)
    model_df.to_csv(MODEL_READY_3CLASS, index=False)

    print("Saved model-ready dataset to:", MODEL_READY_3CLASS)
    print("Final shape:", model_df.shape)

if __name__ == "__main__":
    main()