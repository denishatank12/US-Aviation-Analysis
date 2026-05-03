from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from project_config import RAW_INPUT, CLEANED_DATA, CLEANING_REPORT


INPUT_CSV = RAW_INPUT
OUTPUT_CSV = CLEANED_DATA
REPORT_JSON = CLEANING_REPORT

CHUNKSIZE = 200_000

# For EDA keep them False; for ML-only cleaned output set to True
DROP_CANCELLED = False
DROP_DIVERTED = False

DROP_COLS = ["AIRLINE_DOT", "ORIGIN_CITY", "DEST_CITY"]

TIME_COLS = [
    "CRS_DEP_TIME",
    "DEP_TIME",
    "WHEELS_OFF",
    "WHEELS_ON",
    "CRS_ARR_TIME",
    "ARR_TIME",
]

NUMERIC_COLS = [
    "DOT_CODE",
    "FL_NUMBER",
    "CRS_DEP_TIME",
    "DEP_TIME",
    "DEP_DELAY",
    "TAXI_OUT",
    "WHEELS_OFF",
    "WHEELS_ON",
    "TAXI_IN",
    "CRS_ARR_TIME",
    "ARR_TIME",
    "ARR_DELAY",
    "CANCELLED",
    "DIVERTED",
    "CRS_ELAPSED_TIME",
    "ELAPSED_TIME",
    "AIR_TIME",
    "DISTANCE",
    "DELAY_DUE_CARRIER",
    "DELAY_DUE_WEATHER",
    "DELAY_DUE_NAS",
    "DELAY_DUE_SECURITY",
    "DELAY_DUE_LATE_AIRCRAFT",
]

DELAY_COMPONENT_COLS = [
    "DELAY_DUE_CARRIER",
    "DELAY_DUE_WEATHER",
    "DELAY_DUE_NAS",
    "DELAY_DUE_SECURITY",
    "DELAY_DUE_LATE_AIRCRAFT",
]

WINSOR_COLS = ["DEP_DELAY", "ARR_DELAY", "TAXI_OUT", "TAXI_IN", "AIR_TIME", "ELAPSED_TIME"]
NONNEGATIVE_COLS = ["TAXI_OUT", "TAXI_IN", "AIR_TIME", "ELAPSED_TIME"]

PLACEHOLDER_STRINGS = {
    "", " ", "NA", "N/A", "NULL", "NONE", "UNKNOWN", "nan", "NaN", "null"
}

CANCELLED_RAW_COLS = [
    "DEP_TIME", "DEP_DELAY", "TAXI_OUT", "WHEELS_OFF",
    "WHEELS_ON", "TAXI_IN", "ARR_TIME", "ARR_DELAY",
    "ELAPSED_TIME", "AIR_TIME",
    "DELAY_DUE_CARRIER", "DELAY_DUE_WEATHER", "DELAY_DUE_NAS",
    "DELAY_DUE_SECURITY", "DELAY_DUE_LATE_AIRCRAFT",
]

CANCELLED_DERIVED_COLS = [
    "DEP_TIME_MIN", "WHEELS_OFF_MIN", "WHEELS_ON_MIN", "ARR_TIME_MIN",
    "DEP_DT", "ARR_DT", "ACTUAL_BLOCK_MINS",
    "DEP_DELAY_CLEAN", "ARR_DELAY_CLEAN", "TAXI_OUT_CLEAN", "TAXI_IN_CLEAN",
    "AIR_TIME_CLEAN", "ELAPSED_TIME_CLEAN",
]

DIVERTED_ARRIVAL_COLS = [
    "WHEELS_ON", "TAXI_IN", "ARR_TIME", "ARR_DELAY",
    "ELAPSED_TIME", "AIR_TIME",
    "DELAY_DUE_CARRIER", "DELAY_DUE_WEATHER", "DELAY_DUE_NAS",
    "DELAY_DUE_SECURITY", "DELAY_DUE_LATE_AIRCRAFT",
    "WHEELS_ON_MIN", "ARR_TIME_MIN", "ARR_DT", "ACTUAL_BLOCK_MINS",
    "ARR_DELAY_CLEAN", "TAXI_IN_CLEAN", "AIR_TIME_CLEAN", "ELAPSED_TIME_CLEAN",
]


@dataclass
class CleaningReport:
    input_rows: int
    output_rows: int
    dropped_cancelled_rows: int
    dropped_diverted_rows: int
    invalid_sched_block_rows_removed: int
    invalid_actual_block_rows_removed: int
    missing_before: dict[str, int]
    missing_after: dict[str, int]
    winsor_bounds: dict[str, dict[str, float]]
    cancelled_rows: int
    diverted_rows: int


def normalize_string_placeholders(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip()
    s = s.replace(list(PLACEHOLDER_STRINGS), pd.NA)
    return s


def hhmm_to_minutes(value: Any) -> Optional[int]:
    if pd.isna(value):
        return None
    try:
        iv = int(round(float(value)))
    except Exception:
        return None

    if iv < 0:
        return None
    if iv == 2400:
        return 0

    s = str(iv).zfill(4)
    hh = int(s[:-2])
    mm = int(s[-2:])

    if hh > 24 or mm > 59:
        return None
    if hh == 24 and mm != 0:
        return None
    if hh == 24:
        hh = 0

    return hh * 60 + mm


def build_event_datetime(
    flight_date: pd.Series,
    minutes: pd.Series,
    ref_minutes: pd.Series | None = None
) -> pd.Series:
    dt = pd.to_datetime(flight_date, errors="coerce")
    mins = minutes.astype("Int64")

    out = dt + pd.to_timedelta(mins.fillna(0), unit="m")
    out = out.where(~mins.isna(), pd.NaT)

    if ref_minutes is not None:
        ref = ref_minutes.astype("Int64")
        rollover = (~mins.isna()) & (~ref.isna()) & (mins < ref)
        out = out + pd.to_timedelta(rollover.astype(int) * 1440, unit="m")

    return out


def clean_city_name(series: pd.Series) -> pd.Series:
    s = normalize_string_placeholders(series)
    return s.str.title()


def split_city_state(s: pd.Series) -> tuple[pd.Series, pd.Series]:
    s = normalize_string_placeholders(s)
    parts = s.str.split(",", n=1, expand=True)
    city = parts[0].str.strip().str.title().astype("string")
    if parts.shape[1] > 1:
        state = parts[1].str.strip().str.upper().astype("string")
    else:
        state = pd.Series(pd.NA, index=s.index, dtype="string")
    return city, state


def compute_winsor_bounds(
    input_path: Path,
    chunksize: int = CHUNKSIZE,
    sample_target: int = 200_000
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(7)
    samples = {col: [] for col in WINSOR_COLS}
    per_col_target = max(20_000, sample_target // max(1, len(WINSOR_COLS)))

    for chunk in pd.read_csv(input_path, chunksize=chunksize):
        for col in WINSOR_COLS:
            if col not in chunk.columns:
                continue

            x = pd.to_numeric(chunk[col], errors="coerce").dropna().to_numpy(dtype=float)
            if x.size == 0:
                continue

            current_n = sum(arr.size for arr in samples[col])
            remaining = per_col_target - current_n
            if remaining <= 0:
                continue

            take = min(remaining, max(1000, int(0.02 * x.size)))
            if take >= x.size:
                chosen = x
            else:
                idx = rng.choice(x.size, size=take, replace=False)
                chosen = x[idx]

            samples[col].append(chosen)

    bounds = {}
    for col, arrays in samples.items():
        if not arrays:
            continue

        x = np.concatenate(arrays)
        q1 = float(np.percentile(x, 25))
        q3 = float(np.percentile(x, 75))
        iqr = q3 - q1

        if iqr == 0:
            lower, upper = q1, q3
        else:
            lower = q1 - 3.0 * iqr
            upper = q3 + 3.0 * iqr

        if col in NONNEGATIVE_COLS:
            lower = max(0.0, lower)

        bounds[col] = {"lower": lower, "upper": upper}

    return bounds


def null_operational_fields_for_cancelled(df: pd.DataFrame) -> pd.DataFrame:
    if "CANCELLED" not in df.columns:
        return df

    mask = df["CANCELLED"] == 1
    for col in CANCELLED_RAW_COLS + CANCELLED_DERIVED_COLS:
        if col in df.columns:
            df.loc[mask, col] = np.nan
    return df


def null_arrival_fields_for_diverted(df: pd.DataFrame) -> pd.DataFrame:
    if "DIVERTED" not in df.columns:
        return df

    cancelled = df["CANCELLED"] if "CANCELLED" in df.columns else 0
    mask = (df["DIVERTED"] == 1) & (cancelled == 0)
    for col in DIVERTED_ARRIVAL_COLS:
        if col in df.columns:
            df.loc[mask, col] = np.nan
    return df


def process_chunk(
    df: pd.DataFrame,
    winsor_bounds: dict[str, dict[str, float]]
) -> tuple[pd.DataFrame, int, int]:
    df = df.copy()

    for col in df.columns:
        if df[col].dtype == object or str(df[col].dtype).startswith("string"):
            df[col] = normalize_string_placeholders(df[col])

    if "FL_DATE" in df.columns:
        df["FL_DATE"] = pd.to_datetime(df["FL_DATE"], errors="coerce")

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["CANCELLED", "DIVERTED"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    if "CANCELLATION_CODE" in df.columns:
        df["CANCELLATION_CODE"] = normalize_string_placeholders(df["CANCELLATION_CODE"])
        if "CANCELLED" in df.columns:
            df.loc[df["CANCELLED"] == 0, "CANCELLATION_CODE"] = pd.NA

    for col in DELAY_COMPONENT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if "CANCELLED" in df.columns and "DIVERTED" in df.columns:
                mask = (df["CANCELLED"] == 0) & (df["DIVERTED"] == 0) & (df[col].isna())
                df.loc[mask, col] = 0.0

    if "ORIGIN_CITY" in df.columns:
        df["ORIGIN_CITY_NAME"], df["ORIGIN_STATE"] = split_city_state(df["ORIGIN_CITY"])
    elif "ORIGIN_CITY_NAME" in df.columns:
        df["ORIGIN_CITY_NAME"] = clean_city_name(df["ORIGIN_CITY_NAME"])

    if "DEST_CITY" in df.columns:
        df["DEST_CITY_NAME"], df["DEST_STATE"] = split_city_state(df["DEST_CITY"])
    elif "DEST_CITY_NAME" in df.columns:
        df["DEST_CITY_NAME"] = clean_city_name(df["DEST_CITY_NAME"])

    for col in TIME_COLS:
        if col in df.columns:
            df[f"{col}_MIN"] = df[col].apply(hhmm_to_minutes).astype("Int64")

    if "FL_DATE" in df.columns and "CRS_DEP_TIME_MIN" in df.columns:
        df["CRS_DEP_DT"] = build_event_datetime(df["FL_DATE"], df["CRS_DEP_TIME_MIN"])
    if "FL_DATE" in df.columns and "CRS_ARR_TIME_MIN" in df.columns and "CRS_DEP_TIME_MIN" in df.columns:
        df["CRS_ARR_DT"] = build_event_datetime(df["FL_DATE"], df["CRS_ARR_TIME_MIN"], df["CRS_DEP_TIME_MIN"])
    if "FL_DATE" in df.columns and "DEP_TIME_MIN" in df.columns:
        df["DEP_DT"] = build_event_datetime(df["FL_DATE"], df["DEP_TIME_MIN"])
    if "FL_DATE" in df.columns and "ARR_TIME_MIN" in df.columns and "DEP_TIME_MIN" in df.columns:
        df["ARR_DT"] = build_event_datetime(df["FL_DATE"], df["ARR_TIME_MIN"], df["DEP_TIME_MIN"])

    if {"CRS_DEP_DT", "CRS_ARR_DT"}.issubset(df.columns):
        df["SCHED_BLOCK_MINS"] = (df["CRS_ARR_DT"] - df["CRS_DEP_DT"]).dt.total_seconds() / 60

    if {"DEP_DT", "ARR_DT"}.issubset(df.columns):
        df["ACTUAL_BLOCK_MINS"] = (df["ARR_DT"] - df["DEP_DT"]).dt.total_seconds() / 60

    for col, bounds in winsor_bounds.items():
        if col in df.columns:
            cleaned = pd.to_numeric(df[col], errors="coerce").clip(
                lower=bounds["lower"],
                upper=bounds["upper"]
            )
            if col in NONNEGATIVE_COLS:
                cleaned = cleaned.clip(lower=0)
            df[f"{col}_CLEAN"] = cleaned

    df = null_operational_fields_for_cancelled(df)
    df = null_arrival_fields_for_diverted(df)

    invalid_sched_removed = 0
    invalid_actual_removed = 0

    if "SCHED_BLOCK_MINS" in df.columns:
        bad_sched = df["SCHED_BLOCK_MINS"].notna() & (df["SCHED_BLOCK_MINS"] <= 0)
        invalid_sched_removed = int(bad_sched.sum())
        df = df.loc[~bad_sched].copy()

    if "ACTUAL_BLOCK_MINS" in df.columns:
        operational = pd.Series(True, index=df.index)
        if "CANCELLED" in df.columns:
            operational &= df["CANCELLED"] == 0
        if "DIVERTED" in df.columns:
            operational &= df["DIVERTED"] == 0

        bad_actual = operational & df["ACTUAL_BLOCK_MINS"].notna() & (df["ACTUAL_BLOCK_MINS"] <= 0)
        invalid_actual_removed = int(bad_actual.sum())
        df = df.loc[~bad_actual].copy()

    existing_drop = [col for col in DROP_COLS if col in df.columns]
    if existing_drop:
        df = df.drop(columns=existing_drop)

    if DROP_CANCELLED and "CANCELLED" in df.columns:
        df = df[df["CANCELLED"] == 0].copy()

    if DROP_DIVERTED and "DIVERTED" in df.columns:
        df = df[df["DIVERTED"] == 0].copy()

    return df, invalid_sched_removed, invalid_actual_removed


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_CSV}")

    winsor_bounds = compute_winsor_bounds(INPUT_CSV)

    missing_before: dict[str, int] = {}
    missing_after: dict[str, int] = {}
    input_rows = 0
    output_rows = 0
    cancelled_rows = 0
    diverted_rows = 0
    dropped_cancelled_rows = 0
    dropped_diverted_rows = 0
    invalid_sched_block_rows_removed = 0
    invalid_actual_block_rows_removed = 0

    for chunk in pd.read_csv(INPUT_CSV, chunksize=CHUNKSIZE):
        input_rows += len(chunk)

        na_counts = chunk.isna().sum().to_dict()
        for k, v in na_counts.items():
            missing_before[k] = missing_before.get(k, 0) + int(v)

        if "CANCELLED" in chunk.columns:
            c = pd.to_numeric(chunk["CANCELLED"], errors="coerce").fillna(0).astype(int)
            cancelled_rows += int(c.sum())
            if DROP_CANCELLED:
                dropped_cancelled_rows += int(c.sum())

        if "DIVERTED" in chunk.columns:
            d = pd.to_numeric(chunk["DIVERTED"], errors="coerce").fillna(0).astype(int)
            diverted_rows += int(d.sum())
            if DROP_DIVERTED:
                dropped_diverted_rows += int(d.sum())

    if OUTPUT_CSV.exists():
        OUTPUT_CSV.unlink()

    wrote_header = False

    for chunk in pd.read_csv(INPUT_CSV, chunksize=CHUNKSIZE):
        cleaned, bad_sched_n, bad_actual_n = process_chunk(chunk, winsor_bounds)

        invalid_sched_block_rows_removed += bad_sched_n
        invalid_actual_block_rows_removed += bad_actual_n
        output_rows += len(cleaned)

        na_counts = cleaned.isna().sum().to_dict()
        for k, v in na_counts.items():
            missing_after[k] = missing_after.get(k, 0) + int(v)

        cleaned.to_csv(
            OUTPUT_CSV,
            index=False,
            mode="a",
            header=not wrote_header,
        )
        wrote_header = True

    report = CleaningReport(
        input_rows=int(input_rows),
        output_rows=int(output_rows),
        dropped_cancelled_rows=int(dropped_cancelled_rows),
        dropped_diverted_rows=int(dropped_diverted_rows),
        invalid_sched_block_rows_removed=int(invalid_sched_block_rows_removed),
        invalid_actual_block_rows_removed=int(invalid_actual_block_rows_removed),
        missing_before={k: int(v) for k, v in missing_before.items()},
        missing_after={k: int(v) for k, v in missing_after.items()},
        winsor_bounds=winsor_bounds,
        cancelled_rows=int(cancelled_rows),
        diverted_rows=int(diverted_rows),
    )

    REPORT_JSON.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")

    print(f"Wrote: {OUTPUT_CSV}")
    print(f"Wrote: {REPORT_JSON}")


if __name__ == "__main__":
    main()