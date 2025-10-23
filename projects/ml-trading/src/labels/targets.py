"""
Label generation for ML trading model.
"""

import pandas as pd
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class LabelGenerator:
    """
    Generate classification labels based on forward returns.
    """

    def __init__(
        self,
        horizon_minutes: int = 10,
        up_threshold: float = 0.002,
        down_threshold: float = -0.002,
    ):
        self.horizon_minutes = horizon_minutes
        self.up_threshold = up_threshold
        self.down_threshold = down_threshold

    def generate_labels(
        self, data: pd.DataFrame, price_col: str = "close"
    ) -> pd.DataFrame:
        """
        Generate classification labels based on forward returns.

        Args:
            data: DataFrame with OHLCV data and DatetimeIndex
            price_col: Column name for price data

        Returns:
            DataFrame with labels added
        """
        df = data.copy()

        if price_col not in df.columns:
            raise ValueError(f"Price column '{price_col}' not found in data")

        # Calculate forward returns
        df["forward_return"] = self._calculate_forward_returns(df[price_col])

        # Generate classification labels
        df["label"] = self._classify_returns(df["forward_return"])

        # Drop tail rows equal to horizon length to prevent lookahead bias
        df = df.iloc[: -self.horizon_minutes].copy()

        # Remove rows with NaN labels
        initial_len = len(df)
        df = df.dropna(subset=["label"])
        final_len = len(df)

        if initial_len != final_len:
            logger.info(f"Removed {initial_len - final_len} rows with NaN labels")

        logger.info(f"Generated labels for {len(df)} samples")
        logger.info(f"Label distribution: {df['label'].value_counts().to_dict()}")

        return df

    def _calculate_forward_returns(self, prices: pd.Series) -> pd.Series:
        """
        Calculate forward returns over the specified horizon.

        Args:
            prices: Price series

        Returns:
            Series of forward returns
        """
        # Calculate price at horizon
        future_prices = prices.shift(-self.horizon_minutes)

        # Calculate forward returns
        forward_returns = (future_prices - prices) / prices

        return forward_returns

    def _classify_returns(self, returns: pd.Series) -> pd.Series:
        """
        Classify returns into up/down/flat categories.

        Args:
            returns: Series of forward returns

        Returns:
            Series of classification labels (1: up, -1: down, 0: flat)
        """
        labels = pd.Series(index=returns.index, dtype=int)

        # Up trend
        labels[returns > self.up_threshold] = 1

        # Down trend
        labels[returns < self.down_threshold] = -1

        # Flat trend
        labels[(returns >= self.down_threshold) & (returns <= self.up_threshold)] = 0

        return labels

    def get_label_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get statistics about the generated labels.

        Args:
            data: DataFrame with labels

        Returns:
            Dictionary with label statistics
        """
        if "label" not in data.columns:
            raise ValueError("No 'label' column found in data")

        label_counts = data["label"].value_counts().sort_index()
        total_samples = len(data)

        stats = {
            "total_samples": total_samples,
            "label_counts": label_counts.to_dict(),
            "label_percentages": (label_counts / total_samples * 100).to_dict(),
            "up_samples": label_counts.get(1, 0),
            "down_samples": label_counts.get(-1, 0),
            "flat_samples": label_counts.get(0, 0),
            "up_percentage": label_counts.get(1, 0) / total_samples * 100,
            "down_percentage": label_counts.get(-1, 0) / total_samples * 100,
            "flat_percentage": label_counts.get(0, 0) / total_samples * 100,
            "class_balance_ratio": (
                min(label_counts) / max(label_counts) if len(label_counts) > 0 else 0
            ),
        }

        return stats

    def validate_labels(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate label quality and detect potential issues.

        Args:
            data: DataFrame with labels

        Returns:
            Dictionary with validation results
        """
        if "label" not in data.columns:
            return {"valid": False, "error": "No label column found"}

        if "forward_return" not in data.columns:
            return {"valid": False, "error": "No forward_return column found"}

        validation = {"valid": True, "issues": [], "warnings": []}

        # Check for NaN labels
        nan_labels = data["label"].isnull().sum()
        if nan_labels > 0:
            validation["issues"].append(f"{nan_labels} NaN labels found")
            validation["valid"] = False

        # Check label distribution
        label_counts = data["label"].value_counts()

        # Check for extreme class imbalance
        if len(label_counts) > 0:
            min_class = label_counts.min()
            max_class = label_counts.max()
            imbalance_ratio = min_class / max_class

            if imbalance_ratio < 0.1:
                validation["warnings"].append(
                    f"Extreme class imbalance: {imbalance_ratio:.3f}"
                )
            elif imbalance_ratio < 0.3:
                validation["warnings"].append(
                    f"Moderate class imbalance: {imbalance_ratio:.3f}"
                )

        # Check for missing classes
        expected_classes = [-1, 0, 1]
        missing_classes = [
            cls for cls in expected_classes if cls not in label_counts.index
        ]
        if missing_classes:
            validation["warnings"].append(f"Missing classes: {missing_classes}")

        # Check alignment between labels and returns
        if "forward_return" in data.columns:
            # Verify that labels match return thresholds
            up_mask = data["forward_return"] > self.up_threshold
            down_mask = data["forward_return"] < self.down_threshold
            flat_mask = (data["forward_return"] >= self.down_threshold) & (
                data["forward_return"] <= self.up_threshold
            )

            up_label_mismatch = (data[up_mask]["label"] != 1).sum()
            down_label_mismatch = (data[down_mask]["label"] != -1).sum()
            flat_label_mismatch = (data[flat_mask]["label"] != 0).sum()

            total_mismatches = (
                up_label_mismatch + down_label_mismatch + flat_label_mismatch
            )

            if total_mismatches > 0:
                validation["issues"].append(
                    f"{total_mismatches} label-return mismatches found"
                )
                validation["valid"] = False

        return validation

    def create_binary_labels(
        self, data: pd.DataFrame, target_class: int = 1
    ) -> pd.DataFrame:
        """
        Create binary labels for specific target class.

        Args:
            data: DataFrame with multi-class labels
            target_class: Target class to predict (1: up, -1: down, 0: flat)

        Returns:
            DataFrame with binary labels
        """
        df = data.copy()

        if "label" not in df.columns:
            raise ValueError("No 'label' column found in data")

        # Create binary labels
        df["binary_label"] = (df["label"] == target_class).astype(int)

        logger.info(f"Created binary labels for target class {target_class}")
        logger.info(
            f"Binary label distribution: {df['binary_label'].value_counts().to_dict()}"
        )

        return df

    def get_feature_target_split(
        self, data: pd.DataFrame, feature_cols: list, target_col: str = "label"
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Split data into features and targets.

        Args:
            data: DataFrame with features and labels
            feature_cols: List of feature column names
            target_col: Target column name

        Returns:
            Tuple of (features_df, target_series)
        """
        # Check that all feature columns exist
        missing_features = [col for col in feature_cols if col not in data.columns]
        if missing_features:
            raise ValueError(f"Missing feature columns: {missing_features}")

        if target_col not in data.columns:
            raise ValueError(f"Target column '{target_col}' not found")

        # Extract features and target
        features = data[feature_cols].copy()
        target = data[target_col].copy()

        # Remove rows with NaN values
        valid_mask = ~(features.isnull().any(axis=1) | target.isnull())
        features = features[valid_mask]
        target = target[valid_mask]

        logger.info(
            f"Created feature-target split: {len(features)} samples, {len(feature_cols)} features"
        )

        return features, target
