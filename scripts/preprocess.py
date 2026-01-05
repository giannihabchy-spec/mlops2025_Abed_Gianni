import pandas as pd

from mlproject.preprocess.filters import (
    drop_missing_rows,
    drop_duplicate_rows,
    filter_trip_duration,
    drop_same_location,
)

TRAIN_INPUT_PATH = "src/mlproject/data/train.csv"
TEST_INPUT_PATH = "src/mlproject/data/test.csv"

TRAIN_OUTPUT_PATH = "src/mlproject/data/train_clean.csv"
TEST_OUTPUT_PATH = "src/mlproject/data/test_clean.csv"


def clean_train(data: pd.DataFrame) -> pd.DataFrame:
    data = drop_missing_rows(data)
    data = drop_duplicate_rows(data)
    data = filter_trip_duration(data)   # train only
    data = drop_same_location(data)
    return data


def clean_test(data: pd.DataFrame) -> pd.DataFrame:
    data = drop_missing_rows(data)
    data = drop_duplicate_rows(data)
    # no trip_duration in test
    data = drop_same_location(data)
    return data


def main():
    train = pd.read_csv(TRAIN_INPUT_PATH)
    test = pd.read_csv(TEST_INPUT_PATH)

    train_clean = clean_train(train)
    test_clean = clean_test(test)

    train_clean.to_csv(TRAIN_OUTPUT_PATH, index=False)
    test_clean.to_csv(TEST_OUTPUT_PATH, index=False)

    print(f"Saved cleaned train to {TRAIN_OUTPUT_PATH}")
    print(f"Saved cleaned test  to {TEST_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
