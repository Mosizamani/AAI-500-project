"""
Himalayan Expedition Analysis Module

This module provides reusable functions for analyzing Himalayan expedition data
and predicting summit success. Implements best practices including cross-validation
and proper data validation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.feature_selection import f_classif
import warnings


class ExpeditionAnalyzer:
    """Main class for analyzing Himalayan expedition data."""
    
    def __init__(self, output_dir='overleaf'):
        """
        Initialize the analyzer.
        
        Args:
            output_dir (str): Directory to save visualization outputs
        """
        self.output_dir = output_dir
        self.bad_features = [
            'peak_id', 'peak_name', 'nationality', 'host_cntr', 'other_cntrs', 
            'sponsor', 'leaders', 'team_asc_1', 'team_asc_2', 'team_asc_3', 
            'team_asc_4', 'is_disputed', 'is_claim', 'is_standard_rte', 
            'other_smts', 'approach', 'bc_arrived', 'bc_left', 'total_days', 
            'is_traverse', 'is_ski_snowboard', 'is_parapente', 'term_note', 
            'summit_day', 'time', 'summit_days', 'mbrs_deaths', 'high_camps',
            'hired_abc', 'hired_summits', 'hired_deaths', 'rope_fixed',
            'is_no_hired_abc', 'is_o2_not_used', 'is_o2_used', 'is_o2_descent',
            'is_o2_sleeping', 'is_o2_medical', 'is_o2_unkwn', 'had_o2',
            'camp_sites', 'accidents', 'achievements', 'agency', 'members',
            'rte_2_name', 'rte_3_name', 'rte_4_name', 'rte_1_name', 'exp_result'
        ]
        
        # Set plotting style
        sns.set(style="whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
    
    def validate_data(self, df):
        """
        Validate input data structure and quality.
        
        Args:
            df (pd.DataFrame): Raw expedition data
            
        Returns:
            dict: Validation results and warnings
        """
        validation_results = {
            'errors': [],
            'warnings': [],
            'info': []
        }
        
        # Check required columns
        required_cols = ['mbrs_summited', 'max_elev_reached', 'season']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            validation_results['errors'].append(f"Missing required columns: {missing_cols}")
        
        # Check data types and ranges
        if 'max_elev_reached' in df.columns:
            if df['max_elev_reached'].dtype not in ['int64', 'float64']:
                validation_results['warnings'].append("max_elev_reached should be numeric")
            
            if df['max_elev_reached'].min() < 0:
                validation_results['errors'].append("max_elev_reached contains negative values")
        
        # Check for excessive missing data
        missing_pct = df.isnull().sum() / len(df) * 100
        high_missing = missing_pct[missing_pct > 50]
        if not high_missing.empty:
            validation_results['warnings'].append(
                f"Columns with >50% missing data: {high_missing.index.tolist()}"
            )
        
        validation_results['info'].append(f"Dataset shape: {df.shape}")
        validation_results['info'].append(f"Total missing values: {df.isnull().sum().sum()}")
        
        return validation_results
    
    def preprocess_data(self, df, altitude_threshold=7000):
        """
        Preprocess expedition data for analysis.
        
        Args:
            df (pd.DataFrame): Raw expedition data
            altitude_threshold (int): Minimum altitude for analysis
            
        Returns:
            pd.DataFrame: Preprocessed data ready for modeling
        """
        # Validate data first
        validation = self.validate_data(df)
        if validation['errors']:
            raise ValueError(f"Data validation failed: {validation['errors']}")
        
        # Print warnings
        for warning in validation['warnings']:
            warnings.warn(warning)
        
        # Make a copy to avoid modifying original
        df_processed = df.copy()
        
        # Remove bad features
        existing_bad_features = [col for col in self.bad_features if col in df_processed.columns]
        df_processed = df_processed.drop(columns=existing_bad_features)
        
        # Filter for expeditions above altitude threshold
        df_processed = df_processed[df_processed['max_elev_reached'] > altitude_threshold].copy()
        df_processed = df_processed.drop(columns=['max_elev_reached'])
        
        # Create binary success variable
        df_processed['success'] = (df_processed['mbrs_summited'] > 0).astype(int)
        df_processed = df_processed.drop(columns=['mbrs_summited'])
        
        # Remove year column if present (temporal effects handled separately)
        if 'year' in df_processed.columns:
            df_processed = df_processed.drop(columns=['year'])
        
        # One-hot encode categorical variables
        if 'season' in df_processed.columns:
            df_processed = pd.get_dummies(df_processed, columns=['season'], prefix='season')
        
        # Drop rows with NaN values
        initial_shape = df_processed.shape
        df_processed = df_processed.dropna()
        final_shape = df_processed.shape
        
        print(f"Data preprocessing complete:")
        print(f"  Altitude threshold: >{altitude_threshold}m")
        print(f"  Shape change: {initial_shape} -> {final_shape}")
        print(f"  Rows removed due to NaN: {initial_shape[0] - final_shape[0]}")
        
        return df_processed
    
    def rank_features(self, X, y):
        """
        Rank features using ANOVA F-test.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            
        Returns:
            pd.DataFrame: Feature ranking with F-scores and p-values
        """
        f_scores, p_values = f_classif(X, y)
        feature_ranking = pd.DataFrame({
            'Feature': X.columns,
            'F-Score': f_scores,
            'P-Value': p_values
        }).sort_values(by='F-Score', ascending=False)
        
        print("\nANOVA F-test Feature Ranking:")
        print(feature_ranking.head(10))
        
        return feature_ranking
    
    def train_and_evaluate_model_cv(self, X, y, cv_folds=5, random_state=42):
        """
        Train and evaluate model using cross-validation (addresses README recommendation).
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            cv_folds (int): Number of cross-validation folds
            random_state (int): Random state for reproducibility
            
        Returns:
            dict: Model performance metrics and fitted model
        """
        # Stratified K-Fold for balanced splits
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        
        # Create and evaluate model with cross-validation
        model = LogisticRegression(max_iter=1000, random_state=random_state)
        
        # Cross-validation scores
        cv_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
        cv_roc_scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc')
        
        print(f"\n{cv_folds}-Fold Cross-Validation Results:")
        print(f"Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        print(f"ROC AUC: {cv_roc_scores.mean():.3f} (+/- {cv_roc_scores.std() * 2:.3f})")
        
        # Train final model on full dataset for detailed evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Fit model
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        # Detailed evaluation
        print(f"\nHold-out Test Set Results:")
        print(f"Test set size: {len(y_test)}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Calculate additional metrics
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Peirce Skill Score and Heidke Skill Score
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        pss = tpr - fpr
        
        numerator = 2 * (tp * tn - fp * fn)
        denominator = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
        hss = numerator / denominator if denominator != 0 else 0
        
        print(f"Peirce Skill Score (PSS): {pss:.3f}")
        print(f"Heidke Skill Score (HSS): {hss:.3f}")
        
        # ROC curve
        fpr_curve, tpr_curve, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr_curve, tpr_curve)
        
        results = {
            'model': model,
            'scaler': scaler,
            'cv_accuracy_mean': cv_scores.mean(),
            'cv_accuracy_std': cv_scores.std(),
            'cv_roc_auc_mean': cv_roc_scores.mean(),
            'cv_roc_auc_std': cv_roc_scores.std(),
            'test_accuracy': (y_pred == y_test).mean(),
            'test_roc_auc': roc_auc,
            'pss': pss,
            'hss': hss,
            'confusion_matrix': cm,
            'fpr': fpr_curve,
            'tpr': tpr_curve,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_prob': y_prob
        }
        
        return results
    
    def plot_confusion_matrix(self, cm, save_name="Confusion_Matrix.png"):
        """Create and save confusion matrix visualization."""
        plt.figure(figsize=(6, 6))
        plt.imshow(cm, cmap='rainbow')
        plt.colorbar(label='Count', shrink=0.65)
        plt.xticks([0, 1], ['Failure', 'Success'])
        plt.yticks([0, 1], ['Failure', 'Success'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title("Confusion Matrix")
        
        # Add text annotations
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, f'{cm[i, j]}', ha='center', va='center',
                        color='white' if cm[i, j] > cm.max() / 2 else 'black')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{save_name}', dpi=300, format='png', bbox_inches='tight')
        plt.show()
    
    def plot_roc_curve(self, fpr, tpr, roc_auc, save_name="ROC_Curve.png"):
        """Create and save ROC curve visualization."""
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{self.output_dir}/{save_name}', dpi=300, format='png', bbox_inches='tight')
        plt.show()


def load_and_analyze_expeditions(data_path='data/expeditions.csv', output_dir='overleaf'):
    """
    Main function to load and analyze expedition data with improved methodology.
    
    Args:
        data_path (str): Path to expeditions CSV file
        output_dir (str): Directory for saving outputs
        
    Returns:
        tuple: (processed_data, model_results, analyzer)
    """
    # Initialize analyzer
    analyzer = ExpeditionAnalyzer(output_dir=output_dir)
    
    # Load data
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    print(f"Raw data shape: {df.shape}")
    
    # Preprocess data
    df_processed = analyzer.preprocess_data(df)
    
    # Split features and target
    y = df_processed['success']
    X = df_processed.drop(columns=['success'])
    
    print(f"\nFinal dataset for modeling:")
    print(f"  Features: {X.shape[1]}")
    print(f"  Samples: {X.shape[0]}")
    print(f"  Success rate: {y.mean():.3f}")
    
    # Feature ranking
    feature_ranking = analyzer.rank_features(X, y)
    
    # Train and evaluate model with cross-validation
    results = analyzer.train_and_evaluate_model_cv(X, y)
    
    # Create visualizations
    analyzer.plot_confusion_matrix(results['confusion_matrix'])
    analyzer.plot_roc_curve(results['fpr'], results['tpr'], results['test_roc_auc'])
    
    return df_processed, results, analyzer


if __name__ == "__main__":
    # Example usage
    data, model_results, analyzer = load_and_analyze_expeditions()
    print("\nAnalysis complete. Check the overleaf/ directory for visualizations.")