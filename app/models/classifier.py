"""
XGBoost classifier for crypto trend prediction.
"""

import pandas as pd
import numpy as np

try:
    import xgboost as xgb
except ImportError:
    xgb = None
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class XGBoostClassifier:
    """
    XGBoost classifier for crypto trend prediction.
    """

    def __init__(
        self, model_params: Dict[str, Any], artifacts_dir: str = "./app/artifacts"
    ):
        if xgb is None:
            raise ImportError(
                "XGBoost is required but not installed. Install with: uv add xgboost"
            )

        self.model_params = model_params
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.feature_names = []
        self.training_metadata = {}
        self.is_trained = False

    def fit(
        self, X: pd.DataFrame, y: pd.Series, validation_split: float = 0.2
    ) -> Dict[str, Any]:
        """
        Train the XGBoost classifier.

        Args:
            X: Feature matrix
            y: Target labels
            validation_split: Fraction of data for validation

        Returns:
            Dictionary with training metrics
        """
        logger.info(
            f"Training XGBoost classifier with {len(X)} samples and {len(X.columns)} features"
        )

        # Store feature names
        self.feature_names = list(X.columns)

        # Split data for validation
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        # Convert labels to 0, 1, 2 for XGBoost multi-class
        y_train_encoded = y_train + 1  # Convert -1,0,1 to 0,1,2
        y_val_encoded = y_val + 1

        # Prepare XGBoost data
        dtrain = xgb.DMatrix(
            X_train, label=y_train_encoded, feature_names=self.feature_names
        )
        dval = xgb.DMatrix(X_val, label=y_val_encoded, feature_names=self.feature_names)

        # Set up parameters
        params = {
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "random_state": 42,
            **self.model_params,
        }

        # Train model
        evals = [(dtrain, "train"), (dval, "val")]
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.model_params.get("n_rounds", 400),
            evals=evals,
            early_stopping_rounds=50,
            verbose_eval=False,
        )

        # Calculate training metrics
        train_pred = self.model.predict(dtrain)
        train_pred_labels = np.argmax(train_pred, axis=1) - 1  # Convert 0,1,2 to -1,0,1

        val_pred = self.model.predict(dval)
        val_pred_labels = np.argmax(val_pred, axis=1) - 1  # Convert 0,1,2 to -1,0,1

        # Calculate metrics
        train_accuracy = accuracy_score(y_train, train_pred_labels)
        val_accuracy = accuracy_score(y_val, val_pred_labels)

        metrics = {
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "n_features": len(self.feature_names),
            "n_boost_rounds": self.model.num_boosted_rounds(),
        }

        # Store training metadata
        self.training_metadata = {
            "timestamp": datetime.now().isoformat(),
            "model_params": params,
            "metrics": metrics,
            "feature_names": self.feature_names,
            "train_samples": len(X_train),
            "val_samples": len(X_val),
        }

        self.is_trained = True

        logger.info(
            f"Training completed - Train accuracy: {train_accuracy:.4f}, Val accuracy: {val_accuracy:.4f}"
        )

        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            X: Feature matrix

        Returns:
            Array of predicted labels (-1, 0, 1)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Ensure feature order matches training
        X_ordered = X[self.feature_names]

        # Create DMatrix
        dtest = xgb.DMatrix(X_ordered, feature_names=self.feature_names)

        # Make predictions
        pred_proba = self.model.predict(dtest)
        pred_labels = np.argmax(pred_proba, axis=1) - 1  # Convert to -1, 0, 1

        return pred_labels

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Feature matrix

        Returns:
            Array of class probabilities [P(down), P(flat), P(up)]
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Ensure feature order matches training
        X_ordered = X[self.feature_names]

        # Create DMatrix
        dtest = xgb.DMatrix(X_ordered, feature_names=self.feature_names)

        # Make predictions
        pred_proba = self.model.predict(dtest)

        return pred_proba

    def get_feature_importance(self, importance_type: str = "weight") -> pd.Series:
        """
        Get feature importance scores.

        Args:
            importance_type: Type of importance ('weight', 'gain', 'cover')

        Returns:
            Series with feature importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")

        importance_dict = self.model.get_score(importance_type=importance_type)

        # Create series with all features
        importance_series = pd.Series(index=self.feature_names, dtype=float)
        for feature, score in importance_dict.items():
            importance_series[feature] = score

        # Fill missing values with 0
        importance_series = importance_series.fillna(0)

        # Sort by importance
        importance_series = importance_series.sort_values(ascending=False)

        return importance_series

    def cross_validate(
        self, X: pd.DataFrame, y: pd.Series, cv_folds: int = 5
    ) -> Dict[str, Any]:
        """
        Perform cross-validation.

        Args:
            X: Feature matrix
            y: Target labels
            cv_folds: Number of CV folds

        Returns:
            Dictionary with CV results
        """
        logger.info(f"Performing {cv_folds}-fold cross-validation")

        # Convert labels to 0, 1, 2 for XGBoost multi-class
        y_encoded = y + 1  # Convert -1,0,1 to 0,1,2

        # Prepare data
        dtrain = xgb.DMatrix(X, label=y_encoded)

        # Set up parameters
        params = {
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "random_state": 42,
            **self.model_params,
        }

        # Perform cross-validation
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=self.model_params.get("n_rounds", 400),
            nfold=cv_folds,
            early_stopping_rounds=50,
            verbose_eval=False,
        )

        # Extract metrics
        cv_metrics = {
            "cv_mean_accuracy": cv_results["test-mlogloss-mean"].iloc[-1],
            "cv_std_accuracy": cv_results["test-mlogloss-std"].iloc[-1],
            "best_iteration": len(cv_results),
            "cv_folds": cv_folds,
        }

        logger.info(
            f"CV completed - Mean accuracy: {cv_metrics['cv_mean_accuracy']:.4f} ± {cv_metrics['cv_std_accuracy']:.4f}"
        )

        return cv_metrics

    def save(self, model_name: str = None) -> str:
        """
        Save the trained model and metadata.

        Args:
            model_name: Name for the model (default: timestamp)

        Returns:
            Path to saved model directory
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")

        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"xgboost_model_{timestamp}"

        model_dir = self.artifacts_dir / model_name
        model_dir.mkdir(exist_ok=True)

        # Save model
        model_path = model_dir / "model.json"
        self.model.save_model(str(model_path))

        # Save metadata
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(self.training_metadata, f, indent=2)

        # Save feature names
        features_path = model_dir / "features.json"
        with open(features_path, "w") as f:
            json.dump(self.feature_names, f, indent=2)

        logger.info(f"Model saved to {model_dir}")

        return str(model_dir)

    def load(self, model_path: str) -> None:
        """
        Load a trained model.

        Args:
            model_path: Path to model directory
        """
        model_dir = Path(model_path)

        # Load model
        model_file = model_dir / "model.json"
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")

        self.model = xgb.Booster()
        self.model.load_model(str(model_file))

        # Load metadata
        metadata_file = model_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                self.training_metadata = json.load(f)

        # Load feature names
        features_file = model_dir / "features.json"
        if features_file.exists():
            with open(features_file, "r") as f:
                self.feature_names = json.load(f)

        self.is_trained = True

        logger.info(f"Model loaded from {model_dir}")

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Evaluate model performance on test data.

        Args:
            X: Feature matrix
            y: True labels

        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")

        # Make predictions
        y_pred = self.predict(X)

        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)

        # Classification report
        class_report = classification_report(y, y_pred, output_dict=True)

        # Confusion matrix
        cm = confusion_matrix(y, y_pred)

        # Additional metrics
        metrics = {
            "accuracy": accuracy,
            "classification_report": class_report,
            "confusion_matrix": cm.tolist(),
            "n_samples": len(X),
            "n_features": len(self.feature_names),
        }

        logger.info(f"Model evaluation completed - Accuracy: {accuracy:.4f}")

        return metrics

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.

        Returns:
            Dictionary with model information
        """
        info = {
            "is_trained": self.is_trained,
            "n_features": len(self.feature_names),
            "feature_names": self.feature_names,
            "model_params": self.model_params,
            "training_metadata": self.training_metadata,
        }

        if self.is_trained and self.model is not None:
            info["n_boost_rounds"] = self.model.num_boosted_rounds()

        return info
