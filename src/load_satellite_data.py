from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


# ============================================================
# EXPECTED FILES
# ============================================================

SATELLITE_FILES = {
    "ndvi": RAW_DATA_DIR / "ndvi.csv",
    "ndwi": RAW_DATA_DIR / "ndwi.csv",
    "vv": RAW_DATA_DIR / "vv.csv",
    "vh": RAW_DATA_DIR / "vh.csv",
}


# ============================================================
# COLUMN NORMALIZATION
# ============================================================

def normalize_column_name(column):
    """
    Convert different column naming styles into a standard format.
    """

    column = str(column).strip().lower()
    column = column.replace("\ufeff", "")
    column = column.replace(" ", "_")

    return column


def normalize_columns(df):
    """
    Normalize all column names.
    """

    df = df.copy()

    df.columns = [
        normalize_column_name(col)
        for col in df.columns
    ]

    return df


# ============================================================
# FIND DATE COLUMN
# ============================================================

def find_date_column(df):
    """
    Find the date column even if the company uses names such as:
        date
        c0/date
        timestamp
        acquisition_date
    """

    possible_columns = [
        "date",
        "timestamp",
        "acquisition_date",
        "datetime"
    ]

    for column in possible_columns:
        if column in df.columns:
            return column

    # Support current files such as c0/date
    for column in df.columns:
        if column.endswith("/date"):
            return column

    raise ValueError(
        "Could not find a date column in satellite CSV."
    )


# ============================================================
# FIND FARM ID
# ============================================================

def find_farm_id_column(df):
    """
    Find farm ID if supplied by the company.

    IMPORTANT:
    We DO NOT create a farm ID if it is missing.
    """

    possible_columns = [
        "farm_id",
        "farmid",
        "farm",
        "field_id",
        "fieldid"
    ]

    for column in possible_columns:
        if column in df.columns:
            return column

    return None


# ============================================================
# LOAD ONE SATELLITE FILE
# ============================================================

def load_satellite_file(file_path, metric_name):
    """
    Load and standardize one satellite CSV.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Satellite file not found: {file_path}"
        )

    print(f"\nLoading {metric_name.upper()} data:")
    print(file_path)

    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError(
            f"{metric_name} CSV is empty."
        )

    df = normalize_columns(df)

    # --------------------------------------------------------
    # DATE
    # --------------------------------------------------------

    date_column = find_date_column(df)

    df["date"] = pd.to_datetime(
        df[date_column],
        errors="coerce"
    )

    invalid_dates = df["date"].isna().sum()

    if invalid_dates > 0:
        print(
            f"WARNING: {invalid_dates} rows have invalid dates "
            f"in {metric_name}."
        )

    # --------------------------------------------------------
    # FARM ID
    # --------------------------------------------------------

    farm_id_column = find_farm_id_column(df)

    if farm_id_column is not None:
        df["farm_id"] = (
            df[farm_id_column]
            .astype(str)
            .str.strip()
        )

        print(
            f"Farm ID detected: {farm_id_column}"
        )

    else:
        print(
            f"WARNING: No farm_id found in {metric_name}. "
            f"This dataset cannot currently be treated as "
            f"multi-farm data."
        )

    # --------------------------------------------------------
    # FIND MEAN VALUE
    # --------------------------------------------------------

    mean_column = None

    possible_mean_columns = [
        "mean",
        "c0/mean"
    ]

    for column in possible_mean_columns:
        if column in df.columns:
            mean_column = column
            break

    if mean_column is None:

        for column in df.columns:

            if column.endswith("/mean"):
                mean_column = column
                break

    if mean_column is not None:

        df[metric_name] = pd.to_numeric(
            df[mean_column],
            errors="coerce"
        )

    else:

        # Future company CSV may directly use the metric name
        if metric_name in df.columns:

            df[metric_name] = pd.to_numeric(
                df[metric_name],
                errors="coerce"
            )

        else:

            raise ValueError(
                f"Could not find mean/{metric_name} value "
                f"in {file_path}"
            )

    # --------------------------------------------------------
    # PRESERVE OTHER STATISTICS
    # --------------------------------------------------------

    statistic_mapping = {
        "min": f"{metric_name}_min",
        "max": f"{metric_name}_max",
        "stddev": f"{metric_name}_std",
        "std_dev": f"{metric_name}_std",
        "median": f"{metric_name}_median",
        "p10": f"{metric_name}_p10",
        "p90": f"{metric_name}_p90",
        "samplecount": f"{metric_name}_sample_count",
        "sample_count": f"{metric_name}_sample_count",
        "nodata_count": f"{metric_name}_no_data_count",
        "cloudcoveragepercent": f"{metric_name}_cloud_coverage"
    }

    for original_column, new_column in statistic_mapping.items():

        if original_column in df.columns:

            df[new_column] = pd.to_numeric(
                df[original_column],
                errors="coerce"
            )

    # --------------------------------------------------------
    # SELECT USEFUL COLUMNS
    # --------------------------------------------------------

    keep_columns = [
        "date",
        metric_name
    ]

    if "farm_id" in df.columns:
        keep_columns.insert(0, "farm_id")

    for column in df.columns:

        if column.startswith(metric_name + "_"):
            if column not in keep_columns:
                keep_columns.append(column)

    df = df[keep_columns]

    # --------------------------------------------------------
    # REMOVE INVALID ROWS
    # --------------------------------------------------------

    df = df.dropna(
        subset=["date", metric_name]
    )

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    sort_columns = ["date"]

    if "farm_id" in df.columns:
        sort_columns = ["farm_id", "date"]

    df = df.sort_values(
        sort_columns
    ).reset_index(drop=True)

    print(
        f"Loaded {len(df)} valid {metric_name.upper()} observations."
    )

    return df


# ============================================================
# LOAD ALL SATELLITE DATA
# ============================================================

def load_all_satellite_data():
    """
    Load NDVI, NDWI, VV and VH.
    """

    data = {}

    for metric_name, file_path in SATELLITE_FILES.items():

        data[metric_name] = load_satellite_file(
            file_path,
            metric_name
        )

    return data


# ============================================================
# MAIN TEST
# ============================================================

if __name__ == "__main__":

    satellite_data = load_all_satellite_data()

    print("\n")
    print("=" * 60)
    print("SATELLITE DATA SUMMARY")
    print("=" * 60)

    for metric_name, df in satellite_data.items():

        print(
            f"\n{metric_name.upper()}"
        )

        print(
            f"Rows: {len(df)}"
        )

        print(
            f"Columns: {list(df.columns)}"
        )

        print(
            f"Date range: "
            f"{df['date'].min().date()} → "
            f"{df['date'].max().date()}"
        )

        if "farm_id" in df.columns:

            print(
                f"Farms: "
                f"{df['farm_id'].nunique()}"
            )

        else:

            print(
                "Farms: NOT PROVIDED"
            )