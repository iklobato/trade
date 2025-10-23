"""
Time-series cross-validation splits for ML model training.
"""

import pandas as pd
from typing import List, Tuple, Generator


def create_time_series_splits(
    data: pd.DataFrame, n_folds: int = 6, train_frac: float = 0.75, test_len: str = "7D"
) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
    """
    Create time-series cross-validation splits.

    Args:
        data: DataFrame with DatetimeIndex
        n_folds: Number of CV folds
        train_frac: Fraction of data for training
        test_len: Test set length (pandas frequency string)

    Yields:
        Tuple of (train_data, test_data) for each fold
    """
    total_len = len(data)
    train_len = int(total_len * train_frac)

    # Calculate test length in periods
    test_periods = pd.Timedelta(test_len) // (data.index[1] - data.index[0])

    for i in range(n_folds):
        # Calculate start and end indices for this fold
        fold_start = i * (total_len - train_len - test_periods) // (n_folds - 1)
        train_end = fold_start + train_len
        test_start = train_end
        test_end = test_start + test_periods

        # Ensure we don't exceed data bounds
        if test_end > total_len:
            test_end = total_len
            test_start = test_end - test_periods
            train_end = test_start
            train_start = train_end - train_len

            if train_start < 0:
                train_start = 0
                train_end = min(train_len, total_len - test_periods)
                test_start = train_end
        else:
            train_start = fold_start

        # Extract train and test sets
        train_data = data.iloc[train_start:train_end].copy()
        test_data = data.iloc[test_start:test_end].copy()

        # Skip if either set is too small
        if len(train_data) < 100 or len(test_data) < 10:
            continue

        yield train_data, test_data


def create_walk_forward_splits(
    data: pd.DataFrame,
    initial_train_len: int = 1000,
    step_size: int = 100,
    test_len: int = 100,
) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
    """
    Create walk-forward analysis splits.

    Args:
        data: DataFrame with DatetimeIndex
        initial_train_len: Initial training set length
        step_size: Steps to move forward each iteration
        test_len: Test set length

    Yields:
        Tuple of (train_data, test_data) for each step
    """
    total_len = len(data)

    for start_idx in range(initial_train_len, total_len - test_len, step_size):
        train_end = start_idx
        test_start = train_end
        test_end = test_start + test_len

        if test_end > total_len:
            break

        train_data = data.iloc[:train_end].copy()
        test_data = data.iloc[test_start:test_end].copy()

        yield train_data, test_data


def validate_split_causality(
    train_data: pd.DataFrame, test_data: pd.DataFrame, feature_cols: List[str]
) -> bool:
    """
    Validate that no future information leaks into training data.

    Args:
        train_data: Training DataFrame
        test_data: Test DataFrame
        feature_cols: List of feature column names

    Returns:
        True if split is causal, False otherwise
    """
    if len(train_data) == 0 or len(test_data) == 0:
        return False

    # Check that training data ends before test data starts
    train_end = train_data.index[-1]
    test_start = test_data.index[0]

    if train_end >= test_start:
        return False

    # Check for any NaN values in features
    train_features = train_data[feature_cols]
    test_features = test_data[feature_cols]

    if train_features.isnull().any().any() or test_features.isnull().any().any():
        return False

    return True
