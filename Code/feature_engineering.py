import pandas as pd

SAFE_BASE_COLS = [
    "FL_DATE",
    "AIRLINE",
    "AIRLINE_CODE",
    "ORIGIN",
    "DEST",
    "ORIGIN_STATE",
    "DEST_STATE",
    "CRS_DEP_TIME_MIN",
    "CRS_ARR_TIME_MIN",
    "CRS_ELAPSED_TIME",
    "DISTANCE",
    "SCHED_BLOCK_MINS",
]

def make_delay_class_4(dep_delay):
    if pd.isna(dep_delay):
        return pd.NA
    if dep_delay <= 15:
        return 0
    if dep_delay <= 60:
        return 1
    if dep_delay <= 180:
        return 2
    return 3

def make_delay_class_3(dep_delay):
    if pd.isna(dep_delay):
        return pd.NA
    if dep_delay <= 15:
        return 0
    if dep_delay <= 60:
        return 1
    return 2

def make_dep_time_bucket(minutes):
    if pd.isna(minutes):
        return pd.NA
    hour = int(minutes) // 60
    if 5 <= hour < 9:
        return "Early Morning"
    if 9 <= hour < 12:
        return "Morning"
    if 12 <= hour < 17:
        return "Afternoon"
    if 17 <= hour < 21:
        return "Evening"
    return "Night"

def make_season(month):
    if pd.isna(month):
        return pd.NA
    month = int(month)
    if month in [12, 1, 2]:
        return "Winter"
    if month in [3, 4, 5]:
        return "Spring"
    if month in [6, 7, 8]:
        return "Summer"
    return "Fall"

def filter_valid_multiclass_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "CANCELLED" in out.columns:
        out = out[out["CANCELLED"] == 0].copy()
    if "DIVERTED" in out.columns:
        out = out[out["DIVERTED"] == 0].copy()
    out = out[out["DEP_DELAY"].notna()].copy()
    return out

def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["delay_class_4"] = out["DEP_DELAY"].apply(make_delay_class_4).astype("Int64")
    out["delay_class_3"] = out["DEP_DELAY"].apply(make_delay_class_3).astype("Int64")
    return out

def build_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["FL_DATE"] = pd.to_datetime(out["FL_DATE"], errors="coerce")

    out["year"] = out["FL_DATE"].dt.year
    out["month"] = out["FL_DATE"].dt.month
    out["day"] = out["FL_DATE"].dt.day
    out["day_of_week"] = out["FL_DATE"].dt.dayofweek
    out["is_weekend"] = out["day_of_week"].isin([5, 6]).astype(int)

    out["dep_hour"] = (out["CRS_DEP_TIME_MIN"] // 60).astype("Int64")
    out["arr_hour"] = (out["CRS_ARR_TIME_MIN"] // 60).astype("Int64")

    out["dep_time_bucket"] = out["CRS_DEP_TIME_MIN"].apply(make_dep_time_bucket)
    out["season"] = out["month"].apply(make_season)
    out["route"] = out["ORIGIN"].astype(str) + "_" + out["DEST"].astype(str)

    return out

def build_model_ready_3class(df: pd.DataFrame) -> pd.DataFrame:
    out = filter_valid_multiclass_rows(df)
    out = add_targets(out)

    keep_cols = SAFE_BASE_COLS + ["delay_class_3"]
    out = out[keep_cols].copy()
    out = build_calendar_features(out)

    required = [
        "FL_DATE",
        "AIRLINE",
        "ORIGIN",
        "DEST",
        "CRS_DEP_TIME_MIN",
        "CRS_ARR_TIME_MIN",
        "CRS_ELAPSED_TIME",
        "DISTANCE",
        "SCHED_BLOCK_MINS",
        "delay_class_3",
    ]
    out = out.dropna(subset=required).copy()
    return out