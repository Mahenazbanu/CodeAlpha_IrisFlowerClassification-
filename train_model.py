"""
Iris Flower Classification - Model Training Module

This module provides comprehensive functionality for:
- Exploratory Data Analysis (EDA)
- Data preprocessing
- Multiple ML model training and evaluation
- Model persistence with joblib

Author: ML Engineer
Date: 2026-06-14
"""

import os
import warnings
from typing import Tuple, Dict, Any, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump, load

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# Machine Learning Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set visualization styles
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class IrisClassifier:
    """
    A comprehensive classifier for Iris flower species prediction.

    This class handles the complete ML pipeline from data exploration
    to model persistence, supporting multiple classification algorithms.
    """

    def __init__(self, data_path: str = "Iris.csv", random_state: int = 42) -> None:
        """
        Initialize the IrisClassifier with dataset path and configuration.

        Args:
            data_path: Path to the Iris CSV dataset file.
            random_state: Random seed for reproducibility.
        """
        self.data_path = data_path
        self.random_state = random_state
        self.df: Optional[pd.DataFrame] = None
        self.X_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None
        self.scaler: Optional[StandardScaler] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.models: Dict[str, Any] = {}
        self.results: Dict[str, Dict[str, Any]] = {}
        self.best_model: Optional[Any] = None
        self.best_model_name: Optional[str] = None
        self.feature_names: List[str] = [
            'SepalLengthCm', 'SepalWidthCm',
            'PetalLengthCm', 'PetalWidthCm'
        ]
        self.target_name: str = 'Species'

    # =====================================================================
    # 1. DATA LOADING AND EXPLORATION
    # =====================================================================

    def load_data(self) -> pd.DataFrame:
        """
        Load the Iris dataset from the CSV file.

        Returns:
            DataFrame containing the Iris dataset.

        Raises:
            FileNotFoundError: If the dataset file is not found.
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"Dataset not found at: {self.data_path}. "
                "Please ensure Iris.csv is in the working directory."
            )

        self.df = pd.read_csv(self.data_path)
        print(f"Dataset loaded successfully: {self.df.shape[0]} rows, "
              f"{self.df.shape[1]} columns")
        return self.df

    def explore_data(self) -> None:
        """
        Perform comprehensive exploratory data analysis.

        Displays dataset overview, structure, missing values,
        duplicates, and statistical summary.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        print("\n" + "=" * 60)
        print("EXPLORATORY DATA ANALYSIS")
        print("=" * 60)

        # Dataset Overview
        print("\n--- Dataset Overview ---")
        print(f"Shape: {self.df.shape}")
        print(f"\nFirst 5 rows:\n{self.df.head()}")
        print(f"\nColumn Names: {list(self.df.columns)}")

        # Data Types and Structure
        print("\n--- Data Types and Structure ---")
        print(self.df.dtypes)
        print(f"\nMemory Usage: {self.df.memory_usage(deep=True).sum()} bytes")

        # Missing Value Detection
        print("\n--- Missing Value Detection ---")
        missing = self.df.isnull().sum()
        print(f"Missing values per column:\n{missing}")
        print(f"Total missing values: {missing.sum()}")

        # Duplicate Record Detection
        print("\n--- Duplicate Record Detection ---")
        duplicates = self.df.duplicated().sum()
        print(f"Duplicate rows: {duplicates}")

        # Statistical Summary
        print("\n--- Statistical Summary ---")
        print(self.df.describe())

        # Class Distribution
        print("\n--- Class Distribution ---")
        class_dist = self.df[self.target_name].value_counts()
        print(class_dist)
        print(f"\nClass Proportions:")
        print(self.df[self.target_name].value_counts(normalize=True))

    def visualize_data(self, output_dir: str = ".") -> None:
        """
        Create comprehensive visualizations for EDA.

        Generates and saves:
        - Class distribution bar plot
        - Feature distributions (histograms)
        - Pair plot with species hue
        - Correlation heatmap
        - Feature importance visualization
        - Box plots for each feature by species

        Args:
            output_dir: Directory to save visualization plots.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        print("\n--- Generating Visualizations ---")
        os.makedirs(output_dir, exist_ok=True)

        # 1. Class Distribution
        fig, ax = plt.subplots(figsize=(8, 5))
        class_counts = self.df[self.target_name].value_counts()
        colors = ['#E8F5E9', '#C8E6C9', '#A5D6A7']
        bars = ax.bar(class_counts.index, class_counts.values, color=colors, edgecolor='#2E7D32', linewidth=1.2)
        ax.set_title('Class Distribution of Iris Species', fontsize=14, fontweight='bold')
        ax.set_xlabel('Species', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'class_distribution.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - Saved: class_distribution.png")

        # 2. Feature Distributions
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        for idx, feature in enumerate(self.feature_names):
            self.df[feature].hist(bins=20, ax=axes[idx], color='#A5D6A7',
                                   edgecolor='#2E7D32', alpha=0.8)
            axes[idx].set_title(f'Distribution of {feature}', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel(feature, fontsize=10)
            axes[idx].set_ylabel('Frequency', fontsize=10)
        plt.suptitle('Feature Distributions', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_distributions.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - Saved: feature_distributions.png")

        # 3. Pair Plot
        pair_df = self.df[self.feature_names + [self.target_name]]
        g = sns.pairplot(pair_df, hue=self.target_name, height=2.5,
                         plot_kws={'alpha': 0.7, 's': 50},
                         diag_kws={'alpha': 0.6})
        g.fig.suptitle('Pair Plot of Features by Species', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'pair_plot.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - Saved: pair_plot.png")

        # 4. Correlation Heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        corr_matrix = self.df[self.feature_names].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='YlGnBu', fmt='.2f',
                    linewidths=0.5, ax=ax, square=True)
        ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - Saved: correlation_heatmap.png")

        # 5. Box Plots by Species
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        for idx, feature in enumerate(self.feature_names):
            sns.boxplot(data=self.df, x=self.target_name, y=feature,
                        palette=['#E8F5E9', '#C8E6C9', '#A5D6A7'], ax=axes[idx])
            axes[idx].set_title(f'{feature} by Species', fontsize=11, fontweight='bold')
            axes[idx].set_xlabel('')
        plt.suptitle('Feature Box Plots by Species', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'box_plots.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - Saved: box_plots.png")

        print("All visualizations saved successfully!")

    # =====================================================================
    # 2. DATA PREPROCESSING
    # =====================================================================

    def preprocess_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess the Iris dataset for machine learning.

        Steps:
        1. Drop unnecessary columns (Id)
        2. Handle missing values (drop or impute)
        3. Remove duplicate records
        4. Encode target labels
        5. Split into train/test sets (80/20)
        6. Scale features using StandardScaler

        Returns:
            Tuple of (X_train, X_test, y_train, y_test) as numpy arrays.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        print("\n" + "=" * 60)
        print("DATA PREPROCESSING")
        print("=" * 60)

        # Work on a copy to preserve original data
        df_processed = self.df.copy()

        # Step 1: Drop Id column if present
        if 'Id' in df_processed.columns:
            df_processed = df_processed.drop(columns=['Id'])
            print("- Dropped 'Id' column")

        # Step 2: Handle missing values
        if df_processed.isnull().sum().sum() > 0:
            print(f"- Found {df_processed.isnull().sum().sum()} missing values")
            # For Iris dataset, drop rows with missing values
            initial_rows = len(df_processed)
            df_processed = df_processed.dropna()
            print(f"- Dropped {initial_rows - len(df_processed)} rows with missing values")
        else:
            print("- No missing values found")

        # Step 3: Remove duplicate records
        initial_rows = len(df_processed)
        df_processed = df_processed.drop_duplicates()
        duplicates_removed = initial_rows - len(df_processed)
        if duplicates_removed > 0:
            print(f"- Removed {duplicates_removed} duplicate records")
        else:
            print("- No duplicate records found")

        # Step 4: Separate features and target
        X = df_processed[self.feature_names]
        y = df_processed[self.target_name]
        print(f"- Feature matrix shape: {X.shape}")
        print(f"- Target vector shape: {y.shape}")

        # Step 5: Encode target labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        print(f"- Label encoding completed:")
        for i, class_name in enumerate(self.label_encoder.classes_):
            print(f"    {i}: {class_name}")

        # Step 6: Train-test split (80% train, 20% test)
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y_encoded,
            test_size=0.2,
            random_state=self.random_state,
            stratify=y_encoded
        )
        print(f"- Train set: {X_train_raw.shape[0]} samples")
        print(f"- Test set: {X_test_raw.shape[0]} samples")

        # Step 7: Feature scaling (StandardScaler)
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(X_train_raw)
        self.X_test = self.scaler.transform(X_test_raw)
        self.y_train = y_train
        self.y_test = y_test
        print("- Feature scaling completed (StandardScaler)")

        return self.X_train, self.X_test, self.y_train, self.y_test

    # =====================================================================
    # 3. MACHINE LEARNING MODEL DEVELOPMENT
    # =====================================================================

    def initialize_models(self) -> Dict[str, Any]:
        """
        Initialize all classification models with default parameters.

        Returns:
            Dictionary mapping model names to initialized estimators.
        """
        self.models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000, random_state=self.random_state, multi_class='ovr'
            ),
            'Decision Tree': DecisionTreeClassifier(
                random_state=self.random_state, max_depth=5, min_samples_split=5
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100, random_state=self.random_state, max_depth=5
            ),
            'K-Nearest Neighbors': KNeighborsClassifier(
                n_neighbors=5, weights='uniform', metric='minkowski'
            ),
            'Support Vector Machine': SVC(
                kernel='rbf', probability=True, random_state=self.random_state, C=1.0
            ),
        }
        return self.models

    def train_models(self) -> Dict[str, Any]:
        """
        Train all initialized models on the training data.

        Returns:
            Dictionary mapping model names to trained estimators.
        """
        if self.X_train is None:
            raise ValueError("Data not preprocessed. Call preprocess_data() first.")

        print("\n" + "=" * 60)
        print("MODEL TRAINING")
        print("=" * 60)

        self.initialize_models()

        for name, model in self.models.items():
            print(f"\n- Training {name}...")
            model.fit(self.X_train, self.y_train)
            print(f"  {name} trained successfully")

        return self.models

    # =====================================================================
    # 4. MODEL EVALUATION
    # =====================================================================

    def evaluate_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate all trained models using comprehensive metrics.

        Metrics computed:
        - Accuracy, Precision, Recall, F1-Score
        - Confusion Matrix
        - Classification Report

        Returns:
            Dictionary mapping model names to evaluation results.
        """
        if not self.models:
            raise ValueError("Models not trained. Call train_models() first.")

        print("\n" + "=" * 60)
        print("MODEL EVALUATION")
        print("=" * 60)

        self.results = {}

        for name, model in self.models.items():
            print(f"\n--- {name} ---")

            # Generate predictions
            y_pred = model.predict(self.X_test)

            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
            conf_matrix = confusion_matrix(self.y_test, y_pred)
            class_report = classification_report(
                self.y_test, y_pred,
                target_names=self.label_encoder.classes_,
                zero_division=0
            )

            # Store results
            self.results[name] = {
                'model': model,
                'predictions': y_pred,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': conf_matrix,
                'classification_report': class_report,
            }

            # Print metrics
            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1-Score:  {f1:.4f}")
            print(f"  Confusion Matrix:\n{conf_matrix}")
            print(f"  Classification Report:\n{class_report}")

        return self.results

    def compare_models(self) -> pd.DataFrame:
        """
        Create a comparison DataFrame of all model performances.

        Returns:
            DataFrame with models as rows and metrics as columns.
        """
        if not self.results:
            raise ValueError("Models not evaluated. Call evaluate_models() first.")

        print("\n" + "=" * 60)
        print("MODEL COMPARISON SUMMARY")
        print("=" * 60)

        comparison_data = []
        for name, metrics in self.results.items():
            comparison_data.append({
                'Model': name,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1_score']:.4f}",
            })

        comparison_df = pd.DataFrame(comparison_data)
        print("\n", comparison_df.to_string(index=False))

        # Identify best model based on F1-Score
        best_f1 = 0.0
        for name, metrics in self.results.items():
            if metrics['f1_score'] > best_f1:
                best_f1 = metrics['f1_score']
                self.best_model_name = name
                self.best_model = metrics['model']

        print(f"\n>>> Best Model: {self.best_model_name} "
              f"(F1-Score: {best_f1:.4f}) <<<")

        return comparison_df

    def plot_model_comparison(self, output_dir: str = ".") -> None:
        """
        Create a visualization comparing model performances.

        Args:
            output_dir: Directory to save the comparison plot.
        """
        if not self.results:
            raise ValueError("Models not evaluated. Call evaluate_models() first.")

        fig, ax = plt.subplots(figsize=(12, 7))

        models = list(self.results.keys())
        accuracy_scores = [self.results[m]['accuracy'] for m in models]
        precision_scores = [self.results[m]['precision'] for m in models]
        recall_scores = [self.results[m]['recall'] for m in models]
        f1_scores = [self.results[m]['f1_score'] for m in models]

        x = np.arange(len(models))
        width = 0.2

        bars1 = ax.bar(x - 1.5 * width, accuracy_scores, width, label='Accuracy', color='#C8E6C9', edgecolor='#2E7D32')
        bars2 = ax.bar(x - 0.5 * width, precision_scores, width, label='Precision', color='#A5D6A7', edgecolor='#2E7D32')
        bars3 = ax.bar(x + 0.5 * width, recall_scores, width, label='Recall', color='#81C784', edgecolor='#2E7D32')
        bars4 = ax.bar(x + 1.5 * width, f1_scores, width, label='F1-Score', color='#66BB6A', edgecolor='#2E7D32')

        ax.set_xlabel('Models', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha='right', fontsize=10)
        ax.legend(loc='lower right', fontsize=10)
        ax.set_ylim(0, 1.1)

        # Add value labels on bars
        for bars in [bars1, bars2, bars3, bars4]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - Saved: model_comparison.png")

    # =====================================================================
    # 5. MODEL PERSISTENCE
    # =====================================================================

    def save_model(self, filepath: str = "iris_classifier.pkl") -> str:
        """
        Save the best trained model, scaler, and label encoder.

        Args:
            filepath: Path to save the model pickle file.

        Returns:
            Absolute path to the saved model file.
        """
        if self.best_model is None:
            raise ValueError("No best model found. Run compare_models() first.")

        model_data = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'random_state': self.random_state,
        }

        abs_path = os.path.abspath(filepath)
        dump(model_data, abs_path)
        print(f"\nModel saved to: {abs_path}")
        print(f"Model type: {self.best_model_name}")
        return abs_path

    @staticmethod
    def load_model(filepath: str = "iris_classifier.pkl") -> Dict[str, Any]:
        """
        Load a previously saved model from disk.

        Args:
            filepath: Path to the saved model pickle file.

        Returns:
            Dictionary containing model, scaler, and label encoder.

        Raises:
            FileNotFoundError: If the model file is not found.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Model file not found at: {filepath}. "
                "Please train and save the model first."
            )

        model_data = load(filepath)
        print(f"Model loaded successfully from: {filepath}")
        print(f"Model type: {model_data['model_name']}")
        return model_data

    # =====================================================================
    # 6. PREDICTION INTERFACE
    # =====================================================================

    def predict(
        self,
        sepal_length: float,
        sepal_width: float,
        petal_length: float,
        petal_width: float,
        model_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Predict the Iris species from flower measurements.

        Args:
            sepal_length: Sepal length in cm.
            sepal_width: Sepal width in cm.
            petal_length: Petal length in cm.
            petal_width: Petal width in cm.
            model_data: Optional pre-loaded model data (for loaded models).

        Returns:
            Dictionary with predicted species, confidence, and model info.
        """
        # Use provided model data or instance's best model
        if model_data is not None:
            model = model_data['model']
            scaler = model_data['scaler']
            label_encoder = model_data['label_encoder']
            model_name = model_data['model_name']
        elif self.best_model is not None:
            model = self.best_model
            scaler = self.scaler
            label_encoder = self.label_encoder
            model_name = self.best_model_name
        else:
            raise ValueError("No model available. Train a model or load one.")

        # Create feature array
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        # Scale features
        features_scaled = scaler.transform(features)

        # Predict
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]

        # Decode label
        species = label_encoder.inverse_transform([prediction])[0]
        confidence = probabilities[prediction] * 100

        return {
            'predicted_species': species,
            'confidence': round(confidence, 2),
            'model_used': model_name,
            'all_probabilities': {
                label_encoder.inverse_transform([i])[0]: round(prob * 100, 2)
                for i, prob in enumerate(probabilities)
            }
        }

    # =====================================================================
    # 7. COMPLETE PIPELINE EXECUTION
    # =====================================================================

    def run_full_pipeline(self, save_model_path: str = "iris_classifier.pkl") -> Dict[str, Any]:
        """
        Execute the complete ML pipeline from data loading to model saving.

        Args:
            save_model_path: Path to save the final trained model.

        Returns:
            Dictionary with all results and the best model information.
        """
        print("\n" + "=" * 60)
        print("IRIS FLOWER CLASSIFICATION - FULL PIPELINE")
        print("=" * 60)

        # Step 1: Load data
        self.load_data()

        # Step 2: EDA
        self.explore_data()
        self.visualize_data()

        # Step 3: Preprocessing
        self.preprocess_data()

        # Step 4: Train models
        self.train_models()

        # Step 5: Evaluate models
        self.evaluate_models()

        # Step 6: Compare and select best
        self.compare_models()
        self.plot_model_comparison()

        # Step 7: Save best model
        saved_path = self.save_model(save_model_path)

        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"Best Model: {self.best_model_name}")
        print(f"Model saved to: {saved_path}")

        return {
            'best_model': self.best_model,
            'best_model_name': self.best_model_name,
            'results': self.results,
            'saved_model_path': saved_path,
        }


def main() -> None:
    """Main entry point for the Iris Classification training pipeline."""
    classifier = IrisClassifier(data_path="Iris.csv", random_state=42)
    classifier.run_full_pipeline(save_model_path="iris_classifier.pkl")


if __name__ == "__main__":
    main()
