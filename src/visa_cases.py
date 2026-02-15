import os
import pandas as pd
import numpy as np

FILE_LOCATION = r"A:\Infosys_SpringBoard\AI_Enabled_VISA_Status_Prediction_and_Processing_Time_Estimator\src\data\VisaFile.csv"


# STEP - 1
def read_dataset(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError("Visa dataset not found at given path.")

    data = pd.read_csv(file_path, encoding="latin1", low_memory=False)
    return data


# STEP - 2
def normalize_headers(dataframe):
    cleaned_cols = []

    for col in dataframe.columns:
        col = col.strip().lower().replace(" ", "_")
        cleaned_cols.append(col)

    dataframe.columns = cleaned_cols
    return dataframe


# STEP - 3
def treat_missing_data(dataframe):

    # remove duplicate rows first
    dataframe = dataframe.drop_duplicates().copy()

    # drop columns with excessive missing values >60%
    min_non_null = int(0.4 * len(dataframe))
    dataframe = dataframe.dropna(axis=1, thresh=min_non_null)

    # fill remaining missing values
    for column in dataframe.columns:

        if dataframe[column].dtype == "O":
            dataframe[column].fillna("unknown", inplace=True)

        else:
            median_val = dataframe[column].median()
            dataframe[column].fillna(median_val, inplace=True)

    return dataframe


# STEP - 4
def generate_processing_time(dataframe):

    dataframe["case_received_date"] = pd.to_datetime(
        dataframe["case_received_date"], errors="coerce"
    )

    dataframe["decision_date"] = pd.to_datetime(
        dataframe["decision_date"], errors="coerce"
    )

    # remove invalid date rows
    dataframe = dataframe.dropna(
        subset=["case_received_date", "decision_date"]
    )

    # calculate processing time
    dataframe["processing_time_days"] = (
        dataframe["decision_date"] - dataframe["case_received_date"]
    ).dt.days

    # remove negative values
    dataframe = dataframe[dataframe["processing_time_days"] >= 0]

    return dataframe


# MAIN
def run_preprocessing():

    df = read_dataset(FILE_LOCATION)

    df = normalize_headers(df)

    df = treat_missing_data(df)

    df = generate_processing_time(df)

    print("\n✅ Preprocessing Completed")
    print("Final Dataset Shape:", df.shape)
    print(df.head())

    return df


if __name__ == "__main__":
    final_df = run_preprocessing()
