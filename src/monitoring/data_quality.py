import pandas as pd

def generate_data_quality_report(df):
    report = {}

    report["total_rows"] = len(df)
    report["missing_values"] = df.isnull().sum().to_dict()
    report["duplicate_rows"] = df.duplicated().sum()

    numeric_cols = df.select_dtypes(include=['number']).columns

    report["stats"] = {}
    for col in numeric_cols:
        report["stats"][col] = {
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
            "min": float(df[col].min()),
            "max": float(df[col].max())
        }

    return report