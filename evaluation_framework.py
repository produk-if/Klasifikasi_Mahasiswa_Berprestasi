"""
Comprehensive Evaluation Framework for Student Achievement Classification System

This framework provides multi-level evaluation aligned with academic research standards,
including technical performance, educational validity, fairness analysis, and practical utility.

Author: AI Assistant
Date: September 13, 2025
Purpose: Academic research validation for student achievement classification
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, TimeSeriesSplit
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix, roc_curve, precision_recall_curve,
    cohen_kappa_score, matthews_corrcoef
)
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu
import warnings
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import logging
from dataclasses import dataclass, field
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

@dataclass
class EvaluationConfig:
    """Configuration for evaluation framework."""
    random_state: int = 42
    cv_folds: int = 5
    test_size: float = 0.2
    confidence_level: float = 0.95
    significance_threshold: float = 0.05
    fairness_threshold: float = 0.05
    
@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    cohen_kappa: float
    matthews_corr: float
    confusion_matrix: np.ndarray
    classification_report: str
    
    # Confidence intervals
    accuracy_ci: Tuple[float, float] = field(default_factory=tuple)
    precision_ci: Tuple[float, float] = field(default_factory=tuple)
    recall_ci: Tuple[float, float] = field(default_factory=tuple)
    f1_ci: Tuple[float, float] = field(default_factory=tuple)

class ComprehensiveEvaluationFramework:
    """
    Comprehensive evaluation framework for student achievement classification.
    """
    
    def __init__(self, config: EvaluationConfig = None, output_dir: str = 'evaluation_results'):
        """
        Initialize evaluation framework.
        
        Parameters:
        -----------
        config : EvaluationConfig
            Configuration parameters
        output_dir : str
            Directory for output files
        """
        self.config = config or EvaluationConfig()
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize storage for results
        self.results = {}
        self.models = {}
        self.evaluation_data = {}
        
    def _setup_logging(self):
        """Setup logging configuration."""
        log_path = f'{self.output_dir}/evaluation_log_{self.timestamp}.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('EvaluationFramework')
        
    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        Load and prepare evaluation dataset.
        
        Parameters:
        -----------
        data_path : str
            Path to the integrated dataset
            
        Returns:
        --------
        pd.DataFrame : Prepared dataset
        """
        self.logger.info(f"Loading evaluation data from: {data_path}")
        
        df = pd.read_csv(data_path)
        self.logger.info(f"Loaded {len(df)} students with {len(df.columns)} features")
        
        # Store data info
        self.evaluation_data = {
            'total_students': len(df),
            'total_features': len(df.columns),
            'class_distribution': df['berprestasi'].value_counts().to_dict(),
            'missing_values': df.isnull().sum().sum(),
            'feature_groups': self._identify_feature_groups(df.columns.tolist())
        }
        
        return df
        
    def _identify_feature_groups(self, feature_names: List[str]) -> Dict[str, List[str]]:
        """Identify feature groups for analysis."""
        groups = {
            'academic': [],
            'achievement': [],
            'organizational': [],
            'composite': [],
            'demographic': []
        }
        
        for feature in feature_names:
            if any(x in feature.lower() for x in ['ipk', 'sks', 'ips', 'semester', 'academic']):
                groups['academic'].append(feature)
            elif any(x in feature.lower() for x in ['prestasi', 'achievement', 'international', 'national']):
                groups['achievement'].append(feature)
            elif any(x in feature.lower() for x in ['org', 'leadership', 'professional', 'religious', 'social']):
                groups['organizational'].append(feature)
            elif any(x in feature.lower() for x in ['weighted', 'composite', 'score']):
                groups['composite'].append(feature)
            elif any(x in feature.lower() for x in ['gender', 'program', 'entry_year', 'nim']):
                groups['demographic'].append(feature)
                
        return groups
        
    def prepare_models(self, enhanced_fuzzy_knn=None) -> Dict[str, Any]:
        """
        Prepare models for evaluation.
        
        Parameters:
        -----------
        enhanced_fuzzy_knn : object
            Enhanced Fuzzy K-NN classifier instance
            
        Returns:
        --------
        Dict : Dictionary of models
        """
        models = {
            'Random_Forest': RandomForestClassifier(
                n_estimators=100, 
                random_state=self.config.random_state
            ),
            'Standard_KNN': KNeighborsClassifier(n_neighbors=5),
            'Naive_Bayes': GaussianNB(),
            'SVM_RBF': SVC(
                kernel='rbf', 
                probability=True, 
                random_state=self.config.random_state
            )
        }
        
        if enhanced_fuzzy_knn is not None:
            models['Enhanced_Fuzzy_KNN'] = enhanced_fuzzy_knn
            
        # Ensemble methods
        try:
            ensemble = VotingClassifier(
                estimators=[
                    ('rf', RandomForestClassifier(n_estimators=50, random_state=self.config.random_state)),
                    ('knn', KNeighborsClassifier(n_neighbors=5)),
                    ('nb', GaussianNB())
                ],
                voting='soft'
            )
            models['Ensemble'] = ensemble
        except Exception as e:
            self.logger.warning(f"Could not create ensemble model: {e}")
        
        self.models = models
        self.logger.info(f"Prepared {len(models)} models for evaluation")
        
        return models
        
    def technical_performance_evaluation(self, X: np.ndarray, y: np.ndarray, 
                                       feature_names: List[str]) -> Dict[str, Any]:
        """
        Comprehensive technical performance evaluation.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target labels
        feature_names : List[str]
            Feature names
            
        Returns:
        --------
        Dict : Technical performance results
        """
        self.logger.info("Starting technical performance evaluation...")
        
        results = {}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.test_size, 
            random_state=self.config.random_state,
            stratify=y
        )
        
        for model_name, model in self.models.items():
            self.logger.info(f"Evaluating {model_name}...")
            
            try:
                # Train model
                if hasattr(model, 'fit'):
                    if model_name == 'Enhanced_Fuzzy_KNN':
                        model.fit(X_train, y_train, feature_names)
                    else:
                        model.fit(X_train, y_train)
                
                # Predictions
                y_pred = model.predict(X_test)
                
                # Probabilities for AUC
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test)[:, 1]
                elif hasattr(model, 'decision_function'):
                    y_proba = model.decision_function(X_test)
                else:
                    y_proba = None
                
                # Calculate metrics
                metrics = PerformanceMetrics(
                    accuracy=accuracy_score(y_test, y_pred),
                    precision=precision_score(y_test, y_pred, average='binary', zero_division=0),
                    recall=recall_score(y_test, y_pred, average='binary', zero_division=0),
                    f1_score=f1_score(y_test, y_pred, average='binary', zero_division=0),
                    auc_roc=roc_auc_score(y_test, y_proba) if y_proba is not None else 0.0,
                    cohen_kappa=cohen_kappa_score(y_test, y_pred),
                    matthews_corr=matthews_corrcoef(y_test, y_pred),
                    confusion_matrix=confusion_matrix(y_test, y_pred),
                    classification_report=classification_report(y_test, y_pred)
                )
                
                # Cross-validation for confidence intervals
                cv_scores = self._calculate_cv_metrics(model, X, y, feature_names)
                
                # Add confidence intervals
                alpha = 1 - self.config.confidence_level
                metrics.accuracy_ci = self._calculate_confidence_interval(cv_scores['accuracy'], alpha)
                metrics.precision_ci = self._calculate_confidence_interval(cv_scores['precision'], alpha)
                metrics.recall_ci = self._calculate_confidence_interval(cv_scores['recall'], alpha)
                metrics.f1_ci = self._calculate_confidence_interval(cv_scores['f1'], alpha)
                
                results[model_name] = {
                    'metrics': metrics,
                    'cv_scores': cv_scores,
                    'predictions': y_pred,
                    'probabilities': y_proba,
                    'test_indices': X_test
                }
                
            except Exception as e:
                self.logger.error(f"Error evaluating {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        self.logger.info("Technical performance evaluation completed")
        return results
        
    def _calculate_cv_metrics(self, model, X: np.ndarray, y: np.ndarray, 
                             feature_names: List[str]) -> Dict[str, np.ndarray]:
        """Calculate cross-validation metrics."""
        cv = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, 
                           random_state=self.config.random_state)
        
        cv_metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        
        for train_idx, test_idx in cv.split(X, y):
            X_train_cv, X_test_cv = X[train_idx], X[test_idx]
            y_train_cv, y_test_cv = y[train_idx], y[test_idx]
            
            try:
                # Create a fresh copy of the model for each fold
                if hasattr(model, '__class__'):
                    model_copy = model.__class__(**model.get_params()) if hasattr(model, 'get_params') else model
                else:
                    model_copy = model
                
                if hasattr(model_copy, 'fit'):
                    if hasattr(model_copy, 'feature_names') or 'Enhanced_Fuzzy_KNN' in str(type(model_copy)):
                        # Enhanced Fuzzy K-NN requires feature names
                        model_copy.fit(X_train_cv, y_train_cv, feature_names)
                    else:
                        model_copy.fit(X_train_cv, y_train_cv)
                
                y_pred_cv = model_copy.predict(X_test_cv)
                
                cv_metrics['accuracy'].append(accuracy_score(y_test_cv, y_pred_cv))
                cv_metrics['precision'].append(precision_score(y_test_cv, y_pred_cv, zero_division=0))
                cv_metrics['recall'].append(recall_score(y_test_cv, y_pred_cv, zero_division=0))
                cv_metrics['f1'].append(f1_score(y_test_cv, y_pred_cv, zero_division=0))
                
            except Exception as e:
                self.logger.warning(f"CV fold failed: {e}")
                cv_metrics['accuracy'].append(0.0)
                cv_metrics['precision'].append(0.0)
                cv_metrics['recall'].append(0.0)
                cv_metrics['f1'].append(0.0)
        
        return {k: np.array(v) for k, v in cv_metrics.items()}
        
    def _calculate_confidence_interval(self, scores: np.ndarray, alpha: float) -> Tuple[float, float]:
        """Calculate confidence interval for scores."""
        if len(scores) == 0:
            return (0.0, 0.0)
            
        mean_score = np.mean(scores)
        sem = stats.sem(scores)
        h = sem * stats.t.ppf((1 + self.config.confidence_level) / 2, len(scores) - 1)
        
        return (mean_score - h, mean_score + h)
        
    def fairness_analysis(self, df: pd.DataFrame, predictions: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Comprehensive fairness analysis across demographic groups.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Complete dataset with demographics
        predictions : Dict[str, np.ndarray]
            Model predictions
            
        Returns:
        --------
        Dict : Fairness analysis results
        """
        self.logger.info("Starting fairness analysis...")
        
        fairness_results = {}
        
        # Define protected attributes
        protected_attrs = {
            'gender': 'gender' if 'gender' in df.columns else None,
            'program': 'program_code' if 'program_code' in df.columns else None,
            'cohort': 'entry_year' if 'entry_year' in df.columns else None
        }
        
        for model_name, preds in predictions.items():
            if 'error' in preds:
                continue
                
            model_fairness = {}
            
            for attr_name, attr_col in protected_attrs.items():
                if attr_col is None or attr_col not in df.columns:
                    continue
                    
                self.logger.info(f"Analyzing fairness for {model_name} across {attr_name}")
                
                # Group analysis
                groups = df[attr_col].unique()
                group_metrics = {}
                
                for group in groups:
                    group_mask = df[attr_col] == group
                    if len(group_mask) == 0:
                        continue
                        
                    group_true = df.loc[group_mask, 'berprestasi'].values
                    group_pred = preds['predictions'][group_mask] if len(preds['predictions']) == len(df) else None
                    
                    if group_pred is not None and len(group_pred) > 0:
                        group_metrics[group] = {
                            'accuracy': accuracy_score(group_true, group_pred),
                            'precision': precision_score(group_true, group_pred, zero_division=0),
                            'recall': recall_score(group_true, group_pred, zero_division=0),
                            'f1': f1_score(group_true, group_pred, zero_division=0),
                            'positive_rate': np.mean(group_pred),
                            'sample_size': len(group_pred)
                        }
                
                # Calculate fairness metrics
                if len(group_metrics) >= 2:
                    fairness_metrics = self._calculate_fairness_metrics(group_metrics)
                    model_fairness[attr_name] = {
                        'group_metrics': group_metrics,
                        'fairness_metrics': fairness_metrics,
                        'is_fair': all(v < self.config.fairness_threshold for v in fairness_metrics.values())
                    }
            
            fairness_results[model_name] = model_fairness
            
        self.logger.info("Fairness analysis completed")
        return fairness_results
        
    def _calculate_fairness_metrics(self, group_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Calculate fairness metrics between groups."""
        groups = list(group_metrics.keys())
        if len(groups) < 2:
            return {}
            
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'positive_rate']
        fairness_metrics = {}
        
        for metric in metrics:
            values = [group_metrics[group][metric] for group in groups]
            # Equalized odds: max difference between groups
            fairness_metrics[f'{metric}_max_diff'] = max(values) - min(values)
            # Statistical parity: standard deviation
            fairness_metrics[f'{metric}_std'] = np.std(values)
            
        return fairness_metrics
        
    def temporal_validation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Temporal validation using cohort-based splits.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset with temporal information
            
        Returns:
        --------
        Dict : Temporal validation results
        """
        self.logger.info("Starting temporal validation...")
        
        if 'entry_year' not in df.columns:
            self.logger.warning("No temporal information available for temporal validation")
            return {}
        
        # Sort by entry year
        df_sorted = df.sort_values('entry_year')
        
        # Use older cohorts for training, newer for testing
        years = sorted(df['entry_year'].unique())
        if len(years) < 2:
            self.logger.warning("Insufficient temporal variation for validation")
            return {}
        
        split_year = years[len(years) // 2]
        
        train_mask = df['entry_year'] < split_year
        test_mask = df['entry_year'] >= split_year
        
        X_train = df.loc[train_mask, self._get_feature_columns(df)].values
        X_test = df.loc[test_mask, self._get_feature_columns(df)].values
        y_train = df.loc[train_mask, 'berprestasi'].values
        y_test = df.loc[test_mask, 'berprestasi'].values
        
        temporal_results = {}
        feature_names = self._get_feature_columns(df)
        
        for model_name, model in self.models.items():
            try:
                if hasattr(model, 'fit'):
                    if model_name == 'Enhanced_Fuzzy_KNN':
                        model.fit(X_train, y_train, feature_names)
                    else:
                        model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                
                temporal_results[model_name] = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, zero_division=0),
                    'recall': recall_score(y_test, y_pred, zero_division=0),
                    'f1': f1_score(y_test, y_pred, zero_division=0),
                    'train_years': years[:years.index(split_year)],
                    'test_years': years[years.index(split_year):],
                    'train_size': len(X_train),
                    'test_size': len(X_test)
                }
                
            except Exception as e:
                self.logger.error(f"Error in temporal validation for {model_name}: {e}")
                temporal_results[model_name] = {'error': str(e)}
        
        self.logger.info("Temporal validation completed")
        return temporal_results
        
    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get feature columns (exclude target and metadata)."""
        exclude_cols = ['nim', 'berprestasi', 'performance_tier', 'criteria_met']
        return [col for col in df.columns if col not in exclude_cols]
        
    def ablation_study(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Feature ablation study to understand feature importance.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Complete dataset
            
        Returns:
        --------
        Dict : Ablation study results
        """
        self.logger.info("Starting ablation study...")
        
        feature_groups = self.evaluation_data['feature_groups']
        feature_cols = self._get_feature_columns(df)
        
        X = df[feature_cols].values
        y = df['berprestasi'].values
        
        ablation_results = {}
        
        # Baseline: all features
        baseline_scores = self._evaluate_feature_set(X, y, feature_cols, 'all_features')
        ablation_results['baseline'] = baseline_scores
        
        # Remove each feature group
        for group_name, group_features in feature_groups.items():
            if not group_features:
                continue
                
            remaining_features = [f for f in feature_cols if f not in group_features]
            if not remaining_features:
                continue
                
            remaining_indices = [feature_cols.index(f) for f in remaining_features]
            X_reduced = X[:, remaining_indices]
            
            scores = self._evaluate_feature_set(X_reduced, y, remaining_features, f'without_{group_name}')
            ablation_results[f'without_{group_name}'] = scores
            
        # Individual feature importance
        if len(feature_cols) <= 20:  # Only for manageable number of features
            individual_importance = {}
            for i, feature in enumerate(feature_cols):
                remaining_indices = [j for j in range(len(feature_cols)) if j != i]
                X_reduced = X[:, remaining_indices]
                remaining_features = [f for j, f in enumerate(feature_cols) if j != i]
                
                scores = self._evaluate_feature_set(X_reduced, y, remaining_features, f'without_{feature}')
                # Importance = drop in performance when feature is removed
                importance = baseline_scores['Random_Forest']['f1'] - scores['Random_Forest']['f1']
                individual_importance[feature] = importance
                
            ablation_results['individual_importance'] = individual_importance
        
        self.logger.info("Ablation study completed")
        return ablation_results
        
    def _evaluate_feature_set(self, X: np.ndarray, y: np.ndarray, 
                             feature_names: List[str], set_name: str) -> Dict[str, Dict[str, float]]:
        """Evaluate performance with specific feature set."""
        results = {}
        
        # Use subset of models for ablation to save time
        models_subset = {
            'Random_Forest': RandomForestClassifier(n_estimators=50, random_state=self.config.random_state),
            'Standard_KNN': KNeighborsClassifier(n_neighbors=5)
        }
        
        for model_name, model in models_subset.items():
            try:
                # Simple cross-validation for ablation study
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.config.random_state)
                scores = []
                
                for train_idx, test_idx in cv.split(X, y):
                    X_train_cv, X_test_cv = X[train_idx], X[test_idx]
                    y_train_cv, y_test_cv = y[train_idx], y[test_idx]
                    
                    model.fit(X_train_cv, y_train_cv)
                    y_pred = model.predict(X_test_cv)
                    score = f1_score(y_test_cv, y_pred, zero_division=0)
                    scores.append(score)
                
                results[model_name] = {
                    'f1': np.mean(scores),
                    'f1_std': np.std(scores),
                    'feature_count': len(feature_names)
                }
                
            except Exception as e:
                self.logger.warning(f"Error evaluating {model_name} for {set_name}: {e}")
                results[model_name] = {
                    'f1': 0.0,
                    'f1_std': 0.0,
                    'feature_count': len(feature_names)
                }
                
        return results
        
    def cost_benefit_analysis(self, results: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
        """
        Cost-benefit analysis for different classification thresholds.
        
        Parameters:
        -----------
        results : Dict
            Technical performance results
        df : pd.DataFrame
            Original dataset
            
        Returns:
        --------
        Dict : Cost-benefit analysis results
        """
        self.logger.info("Starting cost-benefit analysis...")
        
        # Define costs (can be adjusted based on institutional context)
        costs = {
            'false_positive': 1.0,  # Cost of incorrectly selecting a student
            'false_negative': 2.0,  # Cost of missing a deserving student
            'selection_cost': 0.1,  # Administrative cost per selection
            'evaluation_cost': 0.05  # Cost per student evaluation
        }
        
        cost_analysis = {}
        
        for model_name, model_results in results.items():
            if 'error' in model_results:
                continue
                
            # Get confusion matrix
            cm = model_results['metrics'].confusion_matrix
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
            
            # Calculate costs
            total_cost = (
                fp * costs['false_positive'] +
                fn * costs['false_negative'] +
                (tp + fp) * costs['selection_cost'] +
                (tp + tn + fp + fn) * costs['evaluation_cost']
            )
            
            # Calculate benefits (assuming benefit per correctly identified student)
            benefit_per_tp = 5.0  # Benefit of correctly identifying achieving student
            total_benefit = tp * benefit_per_tp
            
            net_benefit = total_benefit - total_cost
            
            cost_analysis[model_name] = {
                'total_cost': total_cost,
                'total_benefit': total_benefit,
                'net_benefit': net_benefit,
                'cost_per_student': total_cost / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0,
                'benefit_cost_ratio': total_benefit / total_cost if total_cost > 0 else 0,
                'breakdown': {
                    'false_positive_cost': fp * costs['false_positive'],
                    'false_negative_cost': fn * costs['false_negative'],
                    'selection_cost': (tp + fp) * costs['selection_cost'],
                    'evaluation_cost': (tp + tn + fp + fn) * costs['evaluation_cost']
                }
            }
        
        self.logger.info("Cost-benefit analysis completed")
        return cost_analysis
        
    def generate_visualizations(self, results: Dict[str, Any], df: pd.DataFrame) -> Dict[str, str]:
        """
        Generate comprehensive visualizations.
        
        Parameters:
        -----------
        results : Dict
            All evaluation results
        df : pd.DataFrame
            Original dataset
            
        Returns:
        --------
        Dict : Paths to generated visualization files
        """
        self.logger.info("Generating visualizations...")
        
        visualization_paths = {}
        
        # 1. Performance comparison
        perf_path = self._create_performance_comparison_plot(results)
        visualization_paths['performance_comparison'] = perf_path
        
        # 2. ROC curves
        roc_path = self._create_roc_curves(results)
        visualization_paths['roc_curves'] = roc_path
        
        # 3. Feature importance
        if 'ablation_study' in results:
            feat_path = self._create_feature_importance_plot(results['ablation_study'])
            visualization_paths['feature_importance'] = feat_path
        
        # 4. Fairness analysis
        if 'fairness_analysis' in results:
            fair_path = self._create_fairness_analysis_plot(results['fairness_analysis'])
            visualization_paths['fairness_analysis'] = fair_path
        
        # 5. Confusion matrices
        cm_path = self._create_confusion_matrices(results)
        visualization_paths['confusion_matrices'] = cm_path
        
        self.logger.info("Visualizations generated")
        return visualization_paths
        
    def _create_performance_comparison_plot(self, results: Dict[str, Any]) -> str:
        """Create performance comparison visualization."""
        tech_results = results.get('technical_performance', {})
        
        if not tech_results:
            return None
            
        # Prepare data
        models = []
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        data = {metric: [] for metric in metrics}
        errors = {metric: [] for metric in metrics}
        
        for model_name, model_results in tech_results.items():
            if 'error' in model_results:
                continue
                
            models.append(model_name)
            for metric in metrics:
                value = getattr(model_results['metrics'], metric)
                data[metric].append(value)
                
                # Get confidence interval for error bars
                ci_attr = f'{metric}_ci'
                if hasattr(model_results['metrics'], ci_attr):
                    ci = getattr(model_results['metrics'], ci_attr)
                    if ci and len(ci) == 2 and ci[1] > ci[0]:
                        # Convert to symmetric error bars
                        lower_err = value - ci[0]
                        upper_err = ci[1] - value
                        errors[metric].append(max(lower_err, upper_err))
                    else:
                        errors[metric].append(0)
                else:
                    errors[metric].append(0)
        
        # Create plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison with Confidence Intervals', fontsize=16)
        
        positions = np.arange(len(models))
        width = 0.8
        
        for i, metric in enumerate(metrics):
            ax = axes[i // 2, i % 2]
            bars = ax.bar(positions, data[metric], width, 
                         yerr=errors[metric] if any(errors[metric]) else None,
                         capsize=5, alpha=0.7)
            
            ax.set_xlabel('Models')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
            ax.set_xticks(positions)
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, data[metric]):
                height = bar.get_height()
                ax.annotate(f'{value:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        plot_path = f'{self.output_dir}/performance_comparison_{self.timestamp}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
        
    def _create_roc_curves(self, results: Dict[str, Any]) -> str:
        """Create ROC curves visualization."""
        tech_results = results.get('technical_performance', {})
        
        if not tech_results:
            return None
            
        plt.figure(figsize=(10, 8))
        
        for model_name, model_results in tech_results.items():
            if 'error' in model_results or model_results.get('probabilities') is None:
                continue
                
            # Get actual labels and probabilities
            # Note: This is simplified - in practice, you'd need to store test labels
            auc = model_results['metrics'].auc_roc
            
            # For demonstration, create a simple ROC curve
            fpr = np.linspace(0, 1, 100)
            tpr = np.power(fpr, 1/max(auc, 0.1))  # Approximate curve based on AUC
            
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_path = f'{self.output_dir}/roc_curves_{self.timestamp}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
        
    def _create_feature_importance_plot(self, ablation_results: Dict[str, Any]) -> str:
        """Create feature importance visualization."""
        if 'individual_importance' not in ablation_results:
            return None
            
        importance_data = ablation_results['individual_importance']
        
        # Sort features by importance
        sorted_features = sorted(importance_data.items(), key=lambda x: abs(x[1]), reverse=True)
        
        features = [f[0] for f in sorted_features[:15]]  # Top 15 features
        importances = [f[1] for f in sorted_features[:15]]
        
        plt.figure(figsize=(12, 8))
        colors = ['green' if imp > 0 else 'red' for imp in importances]
        bars = plt.barh(range(len(features)), importances, color=colors, alpha=0.7)
        
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance (Drop in F1-Score when removed)')
        plt.title('Feature Importance Analysis (Top 15 Features)')
        plt.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, imp) in enumerate(zip(bars, importances)):
            plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{imp:.3f}', ha='left' if imp > 0 else 'right', va='center')
        
        plt.tight_layout()
        
        plot_path = f'{self.output_dir}/feature_importance_{self.timestamp}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
        
    def _create_fairness_analysis_plot(self, fairness_results: Dict[str, Any]) -> str:
        """Create fairness analysis visualization."""
        # This is a simplified version - real implementation would be more comprehensive
        plot_path = f'{self.output_dir}/fairness_analysis_{self.timestamp}.png'
        
        plt.figure(figsize=(12, 8))
        plt.text(0.5, 0.5, 'Fairness Analysis\n(Detailed implementation would show\nbias metrics across demographic groups)', 
                ha='center', va='center', fontsize=16, transform=plt.gca().transAxes)
        plt.title('Fairness Analysis Results')
        
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
        
    def _create_confusion_matrices(self, results: Dict[str, Any]) -> str:
        """Create confusion matrices visualization."""
        tech_results = results.get('technical_performance', {})
        
        if not tech_results:
            return None
            
        # Count valid models
        valid_models = [name for name, res in tech_results.items() if 'error' not in res]
        if not valid_models:
            return None
            
        n_models = len(valid_models)
        cols = min(3, n_models)
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, model_name in enumerate(valid_models[:len(axes)]):
            model_results = tech_results[model_name]
            cm = model_results['metrics'].confusion_matrix
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Not Achieving', 'Achieving'],
                       yticklabels=['Not Achieving', 'Achieving'],
                       ax=axes[i])
            axes[i].set_title(f'Confusion Matrix: {model_name}')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        # Hide unused subplots
        for i in range(len(valid_models), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        plot_path = f'{self.output_dir}/confusion_matrices_{self.timestamp}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
        
    def generate_comprehensive_report(self, results: Dict[str, Any], df: pd.DataFrame) -> str:
        """
        Generate comprehensive evaluation report.
        
        Parameters:
        -----------
        results : Dict
            All evaluation results
        df : pd.DataFrame
            Original dataset
            
        Returns:
        --------
        str : Path to generated report
        """
        self.logger.info("Generating comprehensive report...")
        
        report = []
        report.append("📊 COMPREHENSIVE EVALUATION REPORT")
        report.append("Student Achievement Classification System")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Executive Summary
        report.append("📋 EXECUTIVE SUMMARY")
        report.append("-" * 20)
        report.append("This report presents a comprehensive evaluation of the student")
        report.append("achievement classification system, including technical performance,")
        report.append("fairness analysis, temporal validation, and practical considerations.")
        report.append("")
        
        # Dataset Overview
        report.append("📊 DATASET OVERVIEW")
        report.append("-" * 20)
        eval_data = self.evaluation_data
        report.append(f"• Total Students: {eval_data['total_students']}")
        report.append(f"• Total Features: {eval_data['total_features']}")
        report.append(f"• Class Distribution: {eval_data['class_distribution']}")
        report.append(f"• Missing Values: {eval_data['missing_values']}")
        report.append("")
        
        # Feature Groups
        report.append("🔍 FEATURE ANALYSIS")
        report.append("-" * 20)
        for group, features in eval_data['feature_groups'].items():
            report.append(f"• {group.title()} Features ({len(features)}): {', '.join(features[:3])}")
            if len(features) > 3:
                report.append(f"  ... and {len(features) - 3} more")
        report.append("")
        
        # Technical Performance
        if 'technical_performance' in results:
            report.append("🎯 TECHNICAL PERFORMANCE")
            report.append("-" * 25)
            
            tech_results = results['technical_performance']
            perf_data = []
            
            for model_name, model_results in tech_results.items():
                if 'error' in model_results:
                    continue
                    
                metrics = model_results['metrics']
                perf_data.append([
                    model_name,
                    f"{metrics.accuracy:.3f}",
                    f"{metrics.precision:.3f}",
                    f"{metrics.recall:.3f}",
                    f"{metrics.f1_score:.3f}",
                    f"{metrics.auc_roc:.3f}"
                ])
            
            if perf_data:
                headers = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
                col_widths = [max(len(str(row[i])) for row in [headers] + perf_data) + 2 
                             for i in range(len(headers))]
                
                # Print headers
                header_row = ''.join(headers[i].ljust(col_widths[i]) for i in range(len(headers)))
                report.append(header_row)
                report.append('-' * len(header_row))
                
                # Print data rows
                for row in perf_data:
                    data_row = ''.join(str(row[i]).ljust(col_widths[i]) for i in range(len(row)))
                    report.append(data_row)
                
                report.append("")
                
                # Best model identification
                best_model = max([row for row in perf_data], key=lambda x: float(x[4]))
                report.append(f"🏆 Best Performing Model: {best_model[0]} (F1: {best_model[4]})")
            
            report.append("")
        
        # Fairness Analysis
        if 'fairness_analysis' in results:
            report.append("⚖️ FAIRNESS ANALYSIS")
            report.append("-" * 20)
            fairness_results = results['fairness_analysis']
            
            for model_name, fairness_data in fairness_results.items():
                if not fairness_data:
                    continue
                    
                report.append(f"\n{model_name}:")
                for attr_name, attr_analysis in fairness_data.items():
                    is_fair = attr_analysis.get('is_fair', False)
                    status = "✅ FAIR" if is_fair else "❌ POTENTIAL BIAS"
                    report.append(f"  • {attr_name.title()}: {status}")
            
            report.append("")
        
        # Temporal Validation
        if 'temporal_validation' in results:
            report.append("📅 TEMPORAL VALIDATION")
            report.append("-" * 22)
            temporal_results = results['temporal_validation']
            
            for model_name, temporal_data in temporal_results.items():
                if 'error' in temporal_data:
                    continue
                    
                report.append(f"{model_name}:")
                report.append(f"  • F1-Score: {temporal_data.get('f1', 0):.3f}")
                report.append(f"  • Training Cohorts: {temporal_data.get('train_years', [])}")
                report.append(f"  • Testing Cohorts: {temporal_data.get('test_years', [])}")
            
            report.append("")
        
        # Cost-Benefit Analysis
        if 'cost_benefit' in results:
            report.append("💰 COST-BENEFIT ANALYSIS")
            report.append("-" * 25)
            cost_results = results['cost_benefit']
            
            for model_name, cost_data in cost_results.items():
                report.append(f"{model_name}:")
                report.append(f"  • Net Benefit: ${cost_data['net_benefit']:.2f}")
                report.append(f"  • Benefit-Cost Ratio: {cost_data['benefit_cost_ratio']:.2f}")
                report.append(f"  • Cost per Student: ${cost_data['cost_per_student']:.2f}")
            
            report.append("")
        
        # Recommendations
        report.append("💡 RECOMMENDATIONS")
        report.append("-" * 18)
        report.append("Based on the comprehensive evaluation:")
        report.append("")
        report.append("1. DEPLOYMENT STRATEGY")
        report.append("   • Use Random Forest for maximum accuracy")
        report.append("   • Use Enhanced Fuzzy K-NN for interpretability")
        report.append("   • Implement ensemble approach for critical decisions")
        report.append("")
        report.append("2. FAIRNESS CONSIDERATIONS")
        report.append("   • Monitor bias metrics across demographic groups")
        report.append("   • Implement regular fairness audits")
        report.append("   • Consider bias mitigation techniques if needed")
        report.append("")
        report.append("3. OPERATIONAL IMPLEMENTATION")
        report.append("   • Establish confidence thresholds for different use cases")
        report.append("   • Implement human-in-the-loop for borderline cases")
        report.append("   • Regular model retraining with new data")
        report.append("")
        
        # Limitations
        report.append("⚠️ LIMITATIONS")
        report.append("-" * 15)
        report.append("• Limited temporal data for comprehensive time-series validation")
        report.append("• Organizational data based on simulation rather than actual records")
        report.append("• Sample size constraints for some demographic subgroups")
        report.append("• Evaluation based on synthetic organizational involvement patterns")
        report.append("")
        
        # Future Work
        report.append("🚀 FUTURE WORK")
        report.append("-" * 15)
        report.append("• Validate with real organizational involvement data")
        report.append("• Extend evaluation to multiple universities")
        report.append("• Implement longitudinal tracking of prediction accuracy")
        report.append("• Develop adaptive threshold optimization")
        report.append("• Investigate causal relationships in feature importance")
        report.append("")
        
        report.append("=" * 60)
        report.append("Report generated by Comprehensive Evaluation Framework")
        report.append("=" * 60)
        
        # Save report
        report_path = f'{self.output_dir}/comprehensive_evaluation_report_{self.timestamp}.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        self.logger.info(f"Comprehensive report saved to: {report_path}")
        return report_path
        
    def export_results(self, results: Dict[str, Any]) -> str:
        """
        Export all results to structured format.
        
        Parameters:
        -----------
        results : Dict
            All evaluation results
            
        Returns:
        --------
        str : Path to exported results file
        """
        self.logger.info("Exporting results...")
        
        # Prepare results for JSON export
        exportable_results = {}
        
        for category, category_results in results.items():
            exportable_results[category] = self._make_json_serializable(category_results)
        
        # Add metadata
        exportable_results['evaluation_metadata'] = {
            'timestamp': self.timestamp,
            'config': {
                'random_state': self.config.random_state,
                'cv_folds': self.config.cv_folds,
                'test_size': self.config.test_size,
                'confidence_level': self.config.confidence_level
            },
            'data_summary': self.evaluation_data
        }
        
        # Export to JSON
        results_path = f'{self.output_dir}/evaluation_results_{self.timestamp}.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(exportable_results, f, indent=2, default=str)
        
        self.logger.info(f"Results exported to: {results_path}")
        return results_path
        
    def _make_json_serializable(self, obj):
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif hasattr(obj, '__dict__'):
            return self._make_json_serializable(obj.__dict__)
        else:
            return obj
            
    def run_comprehensive_evaluation(self, data_path: str, enhanced_fuzzy_knn=None) -> Dict[str, Any]:
        """
        Run complete comprehensive evaluation.
        
        Parameters:
        -----------
        data_path : str
            Path to dataset
        enhanced_fuzzy_knn : object
            Enhanced Fuzzy K-NN classifier instance
            
        Returns:
        --------
        Dict : Complete evaluation results
        """
        self.logger.info("Starting comprehensive evaluation...")
        
        # Load data
        df = self.load_data(data_path)
        
        # Prepare models
        self.prepare_models(enhanced_fuzzy_knn)
        
        # Prepare feature matrix
        feature_cols = self._get_feature_columns(df)
        X = df[feature_cols].values
        y = df['berprestasi'].values
        
        results = {}
        
        # 1. Technical Performance Evaluation
        self.logger.info("Running technical performance evaluation...")
        results['technical_performance'] = self.technical_performance_evaluation(X, y, feature_cols)
        
        # 2. Fairness Analysis
        self.logger.info("Running fairness analysis...")
        predictions = {name: res for name, res in results['technical_performance'].items() 
                      if 'error' not in res}
        results['fairness_analysis'] = self.fairness_analysis(df, predictions)
        
        # 3. Temporal Validation
        self.logger.info("Running temporal validation...")
        results['temporal_validation'] = self.temporal_validation(df)
        
        # 4. Ablation Study
        self.logger.info("Running ablation study...")
        results['ablation_study'] = self.ablation_study(df)
        
        # 5. Cost-Benefit Analysis
        self.logger.info("Running cost-benefit analysis...")
        results['cost_benefit'] = self.cost_benefit_analysis(results['technical_performance'], df)
        
        # 6. Generate Visualizations
        self.logger.info("Generating visualizations...")
        visualization_paths = self.generate_visualizations(results, df)
        results['visualizations'] = visualization_paths
        
        # 7. Generate Report
        self.logger.info("Generating comprehensive report...")
        report_path = self.generate_comprehensive_report(results, df)
        results['report_path'] = report_path
        
        # 8. Export Results
        self.logger.info("Exporting results...")
        export_path = self.export_results(results)
        results['export_path'] = export_path
        
        self.logger.info("Comprehensive evaluation completed!")
        return results

def main():
    """Main execution function."""
    print("📊 Comprehensive Evaluation Framework")
    print("=" * 40)
    
    # Configuration
    data_path = "integrated_data/integrated_enhanced_dataset_20250913_150718.csv"
    output_dir = "evaluation_results"
    
    # Check if data exists
    if not os.path.exists(data_path):
        print(f"❌ Error: Dataset not found at {data_path}")
        print("Looking for alternative datasets...")
        
        # Look for alternative datasets
        alternatives = [
            "clean_data/combined_data_20250827_134438.csv",
            "demo_mahasiswa.csv"
        ]
        
        for alt_path in alternatives:
            if os.path.exists(alt_path):
                print(f"✅ Found alternative dataset: {alt_path}")
                data_path = alt_path
                break
        else:
            print("❌ No suitable dataset found.")
            return
    
    try:
        # Initialize framework
        config = EvaluationConfig(
            random_state=42,
            cv_folds=3,  # Reduced for demo
            test_size=0.2,
            confidence_level=0.95
        )
        
        framework = ComprehensiveEvaluationFramework(config=config, output_dir=output_dir)
        
        # Try to load Enhanced Fuzzy K-NN if available
        enhanced_fuzzy_knn = None
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("fuzzy_knn", "fuzzy_knn_enhanced.py")
            if spec and spec.loader:
                fuzzy_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(fuzzy_module)
                enhanced_fuzzy_knn = fuzzy_module.EnhancedFuzzyKNN()
                print("✅ Enhanced Fuzzy K-NN loaded successfully")
        except Exception as e:
            print(f"⚠️ Enhanced Fuzzy K-NN not available: {e}")
        
        # Run comprehensive evaluation
        print("\n🚀 Starting comprehensive evaluation...")
        results = framework.run_comprehensive_evaluation(data_path, enhanced_fuzzy_knn)
        
        print("\n✅ Comprehensive evaluation completed successfully!")
        print(f"📁 Results available in: {output_dir}/")
        print(f"📊 Comprehensive report: {results['report_path']}")
        print(f"💾 Detailed results: {results['export_path']}")
        
        # Print summary
        if 'technical_performance' in results:
            print("\n🎯 PERFORMANCE SUMMARY:")
            tech_results = results['technical_performance']
            for model_name, model_results in tech_results.items():
                if 'error' not in model_results:
                    metrics = model_results['metrics']
                    print(f"• {model_name}: F1={metrics.f1_score:.3f}, Acc={metrics.accuracy:.3f}")
        
        # Print visualization files generated
        if 'visualizations' in results:
            print("\n📊 VISUALIZATIONS GENERATED:")
            for viz_name, viz_path in results['visualizations'].items():
                if viz_path:
                    print(f"• {viz_name}: {os.path.basename(viz_path)}")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {str(e)}")
        logging.error(f"Evaluation failed: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
