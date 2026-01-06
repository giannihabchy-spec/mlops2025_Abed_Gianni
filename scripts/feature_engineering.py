import pandas as pd

from mlproject.features.basic_features import (
    convert_datetime,
    add_distance_km,
    add_pickup_month,
    add_pickup_hour,
    add_pickup_weekday,
    drop_large_distance,
)

TRAIN_INPUT = "src/mlproject/data/train_clean.csv"
TEST_INPUT = "src/mlproject/data/test_clean.csv"

TRAIN_OUTPUT = "src/mlproject/data/train_features.csv"
TEST_OUTPUT = "src/mlproject/data/test_features.csv"

TRAIN_COLUMNS_TO_DROP = [
    "pickup_datetime",
    "dropoff_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
]

TEST_COLUMNS_TO_DROP = [
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
]

def engineer_features(df, is_train=True):
    df = convert_datetime(df, ["pickup_datetime"])
    df["store_and_fwd_flag"] = (df["store_and_fwd_flag"] == "Y").astype(int)
    df = add_distance_km(df)
    df = add_pickup_month(df)
    df = add_pickup_hour(df)
    df = add_pickup_weekday(df)

    if is_train:
        df = drop_large_distance(df, max_km=200)
        df = df.drop(columns=TRAIN_COLUMNS_TO_DROP, errors="ignore")
    else:
        df = df.drop(columns=TEST_COLUMNS_TO_DROP, errors="ignore")

    return df


def main():
    train = pd.read_csv(TRAIN_INPUT)
    test = pd.read_csv(TEST_INPUT)

    train = engineer_features(train, is_train=True)
    test = engineer_features(test, is_train=False)

    train.to_csv(TRAIN_OUTPUT, index=False)
    test.to_csv(TEST_OUTPUT, index=False)

    print("Saved feature datasets")


if __name__ == "__main__":
    main()
