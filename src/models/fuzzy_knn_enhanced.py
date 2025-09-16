"""
Enhanced Fuzzy K-Nearest Neighbors Classifier for Student Achievement Classification

This module implements an advanced Fuzzy K-NN classifier with adaptive parameters,
multi-criteria distance calculation, and comprehensive evaluation framework.

Author: AI Assistant
Date: September 13, 2025
Purpose: Student Achievement Classification Research
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, 
                           classification_report, roc_curve)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
from scipy.spatial.distance import euclidean
import warnings
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Any
import logging

warnings.filterwarnings('ignore')

class EnhancedFuzzyKNN:
    """
    Advanced Fuzzy K-Nearest Neighbors classifier with adaptive parameters
    and multi-criteria distance calculation for student achievement classification.
    """
    
    def __init__(self, k=None, m=2.0, feature_weights=None, adaptive_params=True):
        """
        Initialize Enhanced Fuzzy K-NN classifier.
        
        Parameters:
        -----------
        k : int, optional
            Number of neighbors. If None, will be determined adaptively.
        m : float, default=2.0
            Fuzzy parameter for membership calculation.
        feature_weights : dict, optional
            Weights for different feature groups.
        adaptive_params : bool, default=True
            Whether to use adaptive parameter selection.
        """
        self.k = k
        self.m = m
        self.feature_weights = feature_weights or {
            'academic': 0.4,
            'achievement': 0.35,
            'organizational': 0.25
        }
        self.adaptive_params = adaptive_params
        self.X_train = None
        self.y_train = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.feature_groups = None
        self.is_fitted = False
        
        # Setup logging
        self._setup_logging()
        
    def get_params(self, deep=True):
        """Get parameters for sklearn compatibility."""
        return {
            'k': self.k,
            'm': self.m,
            'feature_weights': self.feature_weights,
            'adaptive_params': self.adaptive_params
        }
        
    def set_params(self, **params):
        """Set parameters for sklearn compatibility."""
        for param, value in params.items():
            setattr(self, param, value)
        return self
        
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('EnhancedFuzzyKNN')
        
    def _identify_feature_groups(self, feature_names):
        """
        Identify feature groups based on feature names.
        
        Parameters:
        -----------
        feature_names : list
            List of feature names.
            
        Returns:
        --------
        dict : Feature group assignments
        """
        academic_features = [
            'final_ipk', 'final_sks', 'avg_ips', 'stability_score',
            'semester_count', 'academic_trend', 'ipk_progression', 'academic_score'
        ]
        
        achievement_features = [
            'total_prestasi', 'prestasi_akademik', 'prestasi_non_akademik',
            'prestasi_individu', 'international_achievements', 'national_achievements',
            'regional_achievements', 'achievement_level_score', 'achievement_diversity',
            'achievement_score'
        ]
        
        organizational_features = [
            'total_organizations', 'leadership_positions', 'leadership_duration_months',
            'org_type_diversity', 'avg_involvement_duration', 'current_active_orgs',
            'academic_orgs', 'social_orgs', 'organizational_score', 'academic_weighted',
            'achievement_weighted', 'organizational_weighted'
        ]
        
        groups = {}
        for i, feature in enumerate(feature_names):
            if any(af in feature for af in academic_features):
                groups[i] = 'academic'
            elif any(achf in feature for achf in achievement_features):
                groups[i] = 'achievement'
            elif any(of in feature for of in organizational_features):
                groups[i] = 'organizational'
            else:
                groups[i] = 'other'
                
        return groups
        
    def _determine_optimal_k(self, X, y):
        """
        Determine optimal K value based on dataset characteristics.
        
        Parameters:
        -----------
        X : array-like
            Training features
        y : array-like
            Training labels
            
        Returns:
        --------
        int : Optimal K value
        """
        n_samples = X.shape[0]
        n_classes = len(np.unique(y))
        
        # Rule of thumb: K should be sqrt(n) but adjusted for class balance
        base_k = int(np.sqrt(n_samples))
        
        # Adjust for class imbalance
        class_counts = np.bincount(y)
        min_class_size = np.min(class_counts)
        
        # Ensure K doesn't exceed smallest class size
        optimal_k = min(base_k, min_class_size // 2)
        
        # Ensure K is odd for tie-breaking
        if optimal_k % 2 == 0:
            optimal_k += 1
            
        # Minimum K should be 3, maximum 15 for efficiency
        optimal_k = max(3, min(optimal_k, 15))
        
        self.logger.info(f"Determined optimal K: {optimal_k} (dataset size: {n_samples}, classes: {n_classes})")
        return optimal_k
        
    def _determine_adaptive_m(self, X):
        """
        Determine adaptive fuzzy parameter m based on feature complexity.
        
        Parameters:
        -----------
        X : array-like
            Training features
            
        Returns:
        --------
        float : Adaptive m value
        """
        n_features = X.shape[1]
        feature_variance = np.var(X, axis=0).mean()
        
        # Adjust m based on feature complexity
        if n_features > 30:  # High dimensional
            m = 2.5 + (feature_variance * 0.5)
        elif n_features > 15:  # Medium dimensional
            m = 2.0 + (feature_variance * 0.3)
        else:  # Low dimensional
            m = 1.8 + (feature_variance * 0.2)
            
        # Constrain m to reasonable bounds
        m = max(1.5, min(m, 3.5))
        
        self.logger.info(f"Determined adaptive m: {m:.2f} (features: {n_features}, variance: {feature_variance:.4f})")
        return m
        
    def _weighted_distance(self, x1, x2):
        """
        Calculate weighted distance between two samples.
        
        Parameters:
        -----------
        x1, x2 : array-like
            Two samples to compare
            
        Returns:
        --------
        float : Weighted distance
        """
        distances_by_group = {}
        
        for feature_idx in range(len(x1)):
            group = self.feature_groups.get(feature_idx, 'other')
            if group not in distances_by_group:
                distances_by_group[group] = []
            
            diff = (x1[feature_idx] - x2[feature_idx]) ** 2
            distances_by_group[group].append(diff)
        
        # Calculate weighted distance
        total_distance = 0
        for group, diffs in distances_by_group.items():
            group_distance = np.sqrt(np.sum(diffs))
            weight = self.feature_weights.get(group, 0.1)
            total_distance += weight * group_distance
            
        return total_distance
        
    def _calculate_membership(self, distances, labels):
        """
        Calculate fuzzy membership values.
        
        Parameters:
        -----------
        distances : array-like
            Distances to neighbors
        labels : array-like
            Labels of neighbors
            
        Returns:
        --------
        dict : Class membership values
        """
        unique_classes = np.unique(labels)
        memberships = {}
        
        # Avoid division by zero
        distances = np.array(distances)
        distances[distances == 0] = 1e-8
        
        for class_label in unique_classes:
            class_mask = labels == class_label
            class_distances = distances[class_mask]
            
            if len(class_distances) == 0:
                memberships[class_label] = 0.0
                continue
            
            # Calculate fuzzy membership using weighted distance
            weights = 1.0 / (class_distances ** (2.0 / (self.m - 1)))
            membership = np.sum(weights) / np.sum(1.0 / (distances ** (2.0 / (self.m - 1))))
            memberships[class_label] = membership
            
        return memberships
        
    def fit(self, X, y, feature_names=None):
        """
        Fit the Enhanced Fuzzy K-NN classifier.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
        feature_names : list, optional
            Names of features
        """
        self.logger.info("Fitting Enhanced Fuzzy K-NN classifier...")
        
        # Store training data
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X.shape[1])]
        
        # Identify feature groups
        self.feature_groups = self._identify_feature_groups(self.feature_names)
        
        # Normalize features
        self.X_train = self.scaler.fit_transform(self.X_train)
        
        # Determine optimal parameters if adaptive
        if self.adaptive_params:
            if self.k is None:
                self.k = self._determine_optimal_k(self.X_train, self.y_train)
            self.m = self._determine_adaptive_m(self.X_train)
        
        self.is_fitted = True
        self.logger.info(f"Model fitted with K={self.k}, m={self.m:.2f}")
        
    def predict(self, X, return_probabilities=False):
        """
        Predict class labels for samples.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test samples
        return_probabilities : bool, default=False
            Whether to return class probabilities
            
        Returns:
        --------
        array : Predicted class labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction.")
            
        X_test = self.scaler.transform(X)
        predictions = []
        probabilities = []
        
        for test_sample in X_test:
            # Calculate distances to all training samples
            distances = []
            for train_sample in self.X_train:
                distance = self._weighted_distance(test_sample, train_sample)
                distances.append(distance)
            
            # Get K nearest neighbors
            neighbor_indices = np.argsort(distances)[:self.k]
            neighbor_distances = np.array(distances)[neighbor_indices]
            neighbor_labels = self.y_train[neighbor_indices]
            
            # Calculate fuzzy memberships
            memberships = self._calculate_membership(neighbor_distances, neighbor_labels)
            
            # Predict class with highest membership
            predicted_class = max(memberships.keys(), key=lambda x: memberships[x])
            predictions.append(predicted_class)
            probabilities.append(memberships)
            
        if return_probabilities:
            return np.array(predictions), probabilities
        return np.array(predictions)
        
    def predict_with_confidence(self, X):
        """
        Predict with confidence scores and uncertainty quantification.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test samples
            
        Returns:
        --------
        dict : Predictions with confidence and uncertainty measures
        """
        predictions, probabilities = self.predict(X, return_probabilities=True)
        
        results = {
            'predictions': predictions,
            'confidence_scores': [],
            'uncertainty_scores': [],
            'decision_factors': []
        }
        
        for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
            # Confidence: highest membership value
            confidence = max(probs.values())
            
            # Uncertainty: entropy of membership distribution
            prob_values = list(probs.values())
            prob_values = np.array(prob_values) / sum(prob_values)  # Normalize
            uncertainty = -np.sum(prob_values * np.log2(prob_values + 1e-8))
            
            # Decision factors: feature contributions
            decision_factors = self._analyze_decision_factors(X[i], pred)
            
            results['confidence_scores'].append(confidence)
            results['uncertainty_scores'].append(uncertainty)
            results['decision_factors'].append(decision_factors)
            
        return results
        
    def _analyze_decision_factors(self, sample, prediction):
        """
        Analyze factors contributing to the prediction decision.
        
        Parameters:
        -----------
        sample : array-like
            Test sample
        prediction : int
            Predicted class
            
        Returns:
        --------
        dict : Decision factors analysis
        """
        sample_normalized = self.scaler.transform([sample])[0]
        
        # Find similar samples of the same class
        same_class_mask = self.y_train == prediction
        same_class_samples = self.X_train[same_class_mask]
        
        if len(same_class_samples) == 0:
            return {'top_features': [], 'feature_contributions': {}}
        
        # Calculate feature-wise similarities
        feature_similarities = []
        for i in range(len(sample_normalized)):
            similarities = 1 - np.abs(same_class_samples[:, i] - sample_normalized[i])
            avg_similarity = np.mean(similarities)
            feature_similarities.append(avg_similarity)
        
        # Get top contributing features
        top_indices = np.argsort(feature_similarities)[-5:][::-1]
        top_features = [(self.feature_names[i], feature_similarities[i]) for i in top_indices]
        
        # Group contributions by feature type
        group_contributions = {}
        for i, similarity in enumerate(feature_similarities):
            group = self.feature_groups.get(i, 'other')
            if group not in group_contributions:
                group_contributions[group] = []
            group_contributions[group].append(similarity)
        
        # Average contributions by group
        for group in group_contributions:
            group_contributions[group] = np.mean(group_contributions[group])
        
        return {
            'top_features': top_features,
            'feature_contributions': group_contributions
        }

class ModelComparator:
    """
    Comprehensive model comparison framework for evaluating classifier performance.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.results = {}
        self.models = {}
        self.logger = logging.getLogger('ModelComparator')
        
    def add_model(self, name, model):
        """Add a model to comparison."""
        self.models[name] = model
        
    def evaluate_models(self, X, y, cv_folds=5, feature_names=None):
        """
        Evaluate all models using cross-validation.
        
        Parameters:
        -----------
        X : array-like
            Features
        y : array-like
            Labels
        cv_folds : int, default=5
            Number of cross-validation folds
        feature_names : list, optional
            Feature names
        """
        self.logger.info(f"Evaluating {len(self.models)} models with {cv_folds}-fold CV...")
        
        # Setup cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        # Evaluation metrics
        metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        
        for model_name, model in self.models.items():
            self.logger.info(f"Evaluating {model_name}...")
            
            model_results = {
                'cv_scores': {},
                'mean_scores': {},
                'std_scores': {},
                'detailed_results': []
            }
            
            # Cross-validation evaluation
            for metric in metrics:
                try:
                    scores = cross_val_score(model, X, y, cv=cv, scoring=metric, n_jobs=-1)
                    model_results['cv_scores'][metric] = scores
                    model_results['mean_scores'][metric] = np.mean(scores)
                    model_results['std_scores'][metric] = np.std(scores)
                except Exception as e:
                    self.logger.warning(f"Error evaluating {metric} for {model_name}: {e}")
                    model_results['cv_scores'][metric] = np.array([0.0] * cv_folds)
                    model_results['mean_scores'][metric] = 0.0
                    model_results['std_scores'][metric] = 0.0
            
            # Detailed fold-by-fold evaluation
            fold_results = []
            for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
                X_train_fold, X_test_fold = X[train_idx], X[test_idx]
                y_train_fold, y_test_fold = y[train_idx], y[test_idx]
                
                try:
                    # Fit and predict
                    if hasattr(model, 'fit'):
                        if isinstance(model, EnhancedFuzzyKNN):
                            model.fit(X_train_fold, y_train_fold, feature_names)
                        else:
                            model.fit(X_train_fold, y_train_fold)
                    
                    y_pred = model.predict(X_test_fold)
                    
                    # Calculate detailed metrics
                    fold_result = {
                        'fold': fold + 1,
                        'accuracy': accuracy_score(y_test_fold, y_pred),
                        'precision': precision_score(y_test_fold, y_pred, average='macro', zero_division=0),
                        'recall': recall_score(y_test_fold, y_pred, average='macro', zero_division=0),
                        'f1': f1_score(y_test_fold, y_pred, average='macro', zero_division=0),
                        'confusion_matrix': confusion_matrix(y_test_fold, y_pred).tolist()
                    }
                    
                    # AUC-ROC if possible
                    try:
                        if hasattr(model, 'predict_proba'):
                            y_proba = model.predict_proba(X_test_fold)[:, 1]
                        elif hasattr(model, 'decision_function'):
                            y_proba = model.decision_function(X_test_fold)
                        else:
                            y_proba = None
                            
                        if y_proba is not None:
                            fold_result['auc_roc'] = roc_auc_score(y_test_fold, y_proba)
                    except:
                        fold_result['auc_roc'] = None
                        
                    fold_results.append(fold_result)
                    
                except Exception as e:
                    self.logger.warning(f"Error in fold {fold + 1} for {model_name}: {e}")
                    fold_results.append({
                        'fold': fold + 1,
                        'accuracy': 0.0,
                        'precision': 0.0,
                        'recall': 0.0,
                        'f1': 0.0,
                        'confusion_matrix': [[0, 0], [0, 0]],
                        'auc_roc': None
                    })
            
            model_results['detailed_results'] = fold_results
            self.results[model_name] = model_results
            
        self.logger.info("Model evaluation completed.")
        
    def generate_comparison_report(self):
        """Generate comprehensive comparison report."""
        if not self.results:
            return "No evaluation results available."
            
        report = []
        report.append("📊 MODEL COMPARISON REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Summary table
        report.append("📈 PERFORMANCE SUMMARY:")
        report.append("-" * 40)
        
        summary_data = []
        for model_name, results in self.results.items():
            summary_data.append([
                model_name,
                f"{results['mean_scores']['accuracy']:.4f} ± {results['std_scores']['accuracy']:.4f}",
                f"{results['mean_scores']['precision_macro']:.4f} ± {results['std_scores']['precision_macro']:.4f}",
                f"{results['mean_scores']['recall_macro']:.4f} ± {results['std_scores']['recall_macro']:.4f}",
                f"{results['mean_scores']['f1_macro']:.4f} ± {results['std_scores']['f1_macro']:.4f}"
            ])
        
        # Create performance table
        headers = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
        col_widths = [max(len(str(row[i])) for row in [headers] + summary_data) + 2 for i in range(len(headers))]
        
        # Print headers
        header_row = ''.join(headers[i].ljust(col_widths[i]) for i in range(len(headers)))
        report.append(header_row)
        report.append('-' * len(header_row))
        
        # Print data rows
        for row in summary_data:
            data_row = ''.join(str(row[i]).ljust(col_widths[i]) for i in range(len(row)))
            report.append(data_row)
        
        report.append("")
        
        # Best model identification
        best_model = max(self.results.keys(), 
                        key=lambda x: self.results[x]['mean_scores']['f1_macro'])
        report.append(f"🏆 BEST PERFORMING MODEL: {best_model}")
        report.append(f"   F1-Score: {self.results[best_model]['mean_scores']['f1_macro']:.4f}")
        report.append("")
        
        # Statistical significance testing
        report.append("📊 STATISTICAL SIGNIFICANCE TESTING:")
        report.append("-" * 40)
        
        model_names = list(self.results.keys())
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                model1, model2 = model_names[i], model_names[j]
                scores1 = self.results[model1]['cv_scores']['f1_macro']
                scores2 = self.results[model2]['cv_scores']['f1_macro']
                
                # Paired t-test
                t_stat, p_value = stats.ttest_rel(scores1, scores2)
                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                
                report.append(f"{model1} vs {model2}: p={p_value:.4f} {significance}")
        
        report.append("")
        report.append("Significance levels: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
        
        return '\n'.join(report)
        
    def plot_comparison(self, save_path=None):
        """Create visualization plots for model comparison."""
        if not self.results:
            print("No results to plot.")
            return
            
        try:
            # Setup plotting
            plt.style.use('default')  # Use default instead of seaborn
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Model Comparison Analysis', fontsize=16, fontweight='bold')
            
            # Extract data for plotting
            models = list(self.results.keys())
            metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
            metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            
            # Plot 1: Performance comparison
            ax1 = axes[0, 0]
            x = np.arange(len(models))
            width = 0.2
            
            for i, (metric, name) in enumerate(zip(metrics, metric_names)):
                means = [self.results[model]['mean_scores'][metric] for model in models]
                stds = [self.results[model]['std_scores'][metric] for model in models]
                ax1.bar(x + i * width, means, width, label=name, yerr=stds, capsize=3)
            
            ax1.set_xlabel('Models')
            ax1.set_ylabel('Score')
            ax1.set_title('Performance Comparison')
            ax1.set_xticks(x + width * 1.5)
            ax1.set_xticklabels(models, rotation=45)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: F1-Score distribution
            ax2 = axes[0, 1]
            f1_data = [self.results[model]['cv_scores']['f1_macro'] for model in models]
            ax2.boxplot(f1_data, labels=models)
            ax2.set_title('F1-Score Distribution (CV Folds)')
            ax2.set_ylabel('F1-Score')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Model stability (coefficient of variation)
            ax3 = axes[1, 0]
            stability_scores = []
            for model in models:
                cv_coeff = self.results[model]['std_scores']['f1_macro'] / self.results[model]['mean_scores']['f1_macro']
                stability_scores.append(cv_coeff)
            
            bars = ax3.bar(models, stability_scores, color='lightcoral')
            ax3.set_title('Model Stability (Lower is Better)')
            ax3.set_ylabel('Coefficient of Variation')
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, score in zip(bars, stability_scores):
                height = bar.get_height()
                ax3.annotate(f'{score:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
            
            # Plot 4: Simple bar chart for best model
            ax4 = axes[1, 1]
            best_model = max(self.results.keys(), key=lambda x: self.results[x]['mean_scores']['f1_macro'])
            best_scores = [self.results[best_model]['mean_scores'][metric] for metric in metrics]
            
            bars = ax4.bar(metric_names, best_scores, color='lightblue')
            ax4.set_title(f'Best Model Performance: {best_model}')
            ax4.set_ylabel('Score')
            ax4.set_ylim(0, 1)
            ax4.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, score in zip(bars, best_scores):
                height = bar.get_height()
                ax4.annotate(f'{score:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Comparison plots saved to: {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            print(f"Error creating plots: {e}")
            self.logger.warning(f"Could not create visualization plots: {e}")

class ExperimentRunner:
    """
    Main experiment runner for Enhanced Fuzzy K-NN evaluation.
    """
    
    def __init__(self, data_path, output_dir='fuzzy_knn_results'):
        self.data_path = data_path
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'{self.output_dir}/experiment_log_{self.timestamp}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('ExperimentRunner')
        
    def load_data(self):
        """Load and prepare the dataset."""
        self.logger.info(f"Loading data from: {self.data_path}")
        
        # Load the enhanced dataset
        df = pd.read_csv(self.data_path)
        
        # Separate features and target
        feature_columns = [col for col in df.columns if col not in ['nim', 'berprestasi', 'performance_tier']]
        X = df[feature_columns].values
        y = df['berprestasi'].values
        
        # Handle any missing values
        if np.any(np.isnan(X)):
            self.logger.warning("Found missing values, filling with mean...")
            X = np.nan_to_num(X, nan=np.nanmean(X, axis=0))
        
        self.logger.info(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
        self.logger.info(f"Class distribution: {np.bincount(y)}")
        
        return X, y, feature_columns
        
    def run_experiment(self):
        """Run the complete experiment."""
        self.logger.info("Starting Enhanced Fuzzy K-NN experiment...")
        
        # Load data
        X, y, feature_names = self.load_data()
        
        # Initialize models
        models = {
            'Enhanced_Fuzzy_KNN': EnhancedFuzzyKNN(adaptive_params=True),
            'Standard_KNN': KNeighborsClassifier(n_neighbors=5),
            'Naive_Bayes': GaussianNB(),
            'SVM_RBF': SVC(kernel='rbf', probability=True, random_state=42),
            'Random_Forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        
        # Create comparator
        comparator = ModelComparator(random_state=42)
        for name, model in models.items():
            comparator.add_model(name, model)
        
        # Evaluate models
        comparator.evaluate_models(X, y, cv_folds=5, feature_names=feature_names)
        
        # Generate reports
        self.logger.info("Generating comparison report...")
        report = comparator.generate_comparison_report()
        print(report)
        
        # Save report
        report_path = f'{self.output_dir}/comparison_report_{self.timestamp}.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # Create plots
        try:
            plot_path = f'{self.output_dir}/model_comparison_{self.timestamp}.png'
            comparator.plot_comparison(save_path=plot_path)
        except Exception as e:
            self.logger.warning(f"Could not create plots: {e}")
            self.logger.info("Continuing without plots...")
        
        # Detailed analysis of Enhanced Fuzzy K-NN
        self._analyze_fuzzy_knn(X, y, feature_names)
        
        # Export results
        self._export_results(comparator.results)
        
        self.logger.info(f"Experiment completed! Results saved to: {self.output_dir}")
        
    def _analyze_fuzzy_knn(self, X, y, feature_names):
        """Detailed analysis of Enhanced Fuzzy K-NN performance."""
        self.logger.info("Performing detailed Enhanced Fuzzy K-NN analysis...")
        
        # Create and train the model
        fuzzy_knn = EnhancedFuzzyKNN(adaptive_params=True)
        
        # Split data for detailed analysis
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        fuzzy_knn.fit(X_train, y_train, feature_names)
        
        # Predictions with confidence
        detailed_results = fuzzy_knn.predict_with_confidence(X_test)
        
        # Analyze decision factors
        analysis_results = {
            'model_parameters': {
                'optimal_k': fuzzy_knn.k,
                'fuzzy_m': fuzzy_knn.m,
                'feature_weights': fuzzy_knn.feature_weights
            },
            'prediction_analysis': {
                'predictions': detailed_results['predictions'].tolist(),
                'confidence_scores': detailed_results['confidence_scores'],
                'uncertainty_scores': detailed_results['uncertainty_scores'],
                'mean_confidence': np.mean(detailed_results['confidence_scores']),
                'mean_uncertainty': np.mean(detailed_results['uncertainty_scores'])
            },
            'feature_importance': self._analyze_feature_importance(detailed_results['decision_factors'])
        }
        
        # Save detailed analysis
        analysis_path = f'{self.output_dir}/fuzzy_knn_analysis_{self.timestamp}.json'
        with open(analysis_path, 'w', encoding='utf-8') as f:
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            # Apply conversion recursively
            def recursive_convert(obj):
                if isinstance(obj, dict):
                    return {k: recursive_convert(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [recursive_convert(item) for item in obj]
                else:
                    return convert_numpy(obj)
            
            converted_results = recursive_convert(analysis_results)
            json.dump(converted_results, f, indent=2)
        
        self.logger.info(f"Detailed analysis saved to: {analysis_path}")
        
    def _analyze_feature_importance(self, decision_factors_list):
        """Analyze feature importance across all predictions."""
        feature_contributions = {}
        group_contributions = {}
        
        for factors in decision_factors_list:
            # Feature-level contributions
            for feature, importance in factors['top_features']:
                if feature not in feature_contributions:
                    feature_contributions[feature] = []
                feature_contributions[feature].append(importance)
            
            # Group-level contributions
            for group, contribution in factors['feature_contributions'].items():
                if group not in group_contributions:
                    group_contributions[group] = []
                group_contributions[group].append(contribution)
        
        # Calculate average contributions
        avg_feature_contributions = {
            feature: np.mean(contributions) 
            for feature, contributions in feature_contributions.items()
        }
        
        avg_group_contributions = {
            group: np.mean(contributions)
            for group, contributions in group_contributions.items()
        }
        
        return {
            'feature_importance': avg_feature_contributions,
            'group_importance': avg_group_contributions,
            'top_features': sorted(avg_feature_contributions.items(), 
                                 key=lambda x: x[1], reverse=True)[:10]
        }
        
    def _export_results(self, results):
        """Export detailed results to JSON."""
        results_path = f'{self.output_dir}/detailed_results_{self.timestamp}.json'
        
        # Convert results for JSON serialization
        exportable_results = {}
        for model_name, model_results in results.items():
            exportable_results[model_name] = {
                'mean_scores': model_results['mean_scores'],
                'std_scores': model_results['std_scores'],
                'cv_scores': {metric: scores.tolist() for metric, scores in model_results['cv_scores'].items()},
                'detailed_results': model_results['detailed_results']
            }
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(exportable_results, f, indent=2)
        
        self.logger.info(f"Detailed results exported to: {results_path}")

def main():
    """Main execution function."""
    print("🧠 Enhanced Fuzzy K-NN for Student Achievement Classification")
    print("=" * 65)
    
    # Configuration
    data_path = "enhanced_clean_data/combined_enhanced_20250913_144318.csv"
    output_dir = "fuzzy_knn_results"
    
    # Check if data file exists
    if not os.path.exists(data_path):
        print(f"❌ Error: Data file not found at {data_path}")
        print("Please ensure the enhanced dataset is available.")
        return
    
    # Run experiment
    try:
        runner = ExperimentRunner(data_path, output_dir)
        runner.run_experiment()
        print("\n✅ Experiment completed successfully!")
        print(f"📁 Results available in: {output_dir}/")
        
    except Exception as e:
        print(f"❌ Experiment failed: {str(e)}")
        logging.error(f"Experiment failed: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
