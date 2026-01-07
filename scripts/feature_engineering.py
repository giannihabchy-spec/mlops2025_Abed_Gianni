import argparse
import pandas as pd

from mlproject.preprocess.filters import (
    drop_missing_rows,
    drop_duplicate_rows,
    filter_trip_duration,
    drop_same_location,
)

DEFAULT_TRAIN_INPUT_PATH = "src/mlproject/data/train.csv"
DEFAULT_TEST_INPUT_PATH = "src/mlproject/data/test.csv"

DEFAULT_TRAIN_OUTPUT_PATH = "src/mlproject/data/train_clean.csv"
DEFAULT_TEST_OUTPUT_PATH = "src/mlproject/data/test_clean.csv"


def clean_train(data: pd.DataFrame) -> pd.DataFrame:
    data = drop_missing_rows(data)
    data = drop_duplicate_rows(data)
    data = filter_trip_duration(data)   # train only
    data = drop_same_location(data)
    return data


def clean_test(data: pd.DataFrame) -> pd.DataFrame:
    data = drop_missing_rows(data)
    data = drop_duplicate_rows(data)
    data = drop_same_location(data)
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-in", default=DEFAULT_TRAIN_INPUT_PATH)
    parser.add_argument("--test-in", default=DEFAULT_TEST_INPUT_PATH)
    parser.add_argument("--train-out", default=DEFAULT_TRAIN_OUTPUT_PATH)
    parser.add_argument("--test-out", default=DEFAULT_TEST_OUTPUT_PATH)
    args = parser.parse_args()

    train = pd.read_csv(args.train_in)
    test = pd.read_csv(args.test_in)

    train_clean = clean_train(train)
    test_clean = clean_test(test)

    train_clean.to_csv(args.train_out, index=False)
    test_clean.to_csv(args.test_out, index=False)

    print(f"Saved cleaned train to {args.train_out}")
    print(f"Saved cleaned test  to {args.test_out}")


if __name__ == "__main__":
    main()

