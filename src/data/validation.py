"""
Data validation module for the MLOps pipeline.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import json

from src.utils.logger import get_logger
from src.utils.database import db_manager
from src.utils.helpers import get_timestamp

logger = get_logger(__name__)


class DataValidator:
    """Data validation class for quality checks."""
    
    def __init__(self, config: Dict):
        """
        Initialize data validator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.validation_config = config.get('data', {}).get('validation', {})
        self.max_missing_percentage = self.validation_config.get('max_missing_percentage', 0.1)
        self.min_rows = self.validation_config.get('min_rows', 1000)
        self.required_columns = self.validation_config.get('required_columns', [])
    
    def validate_schema(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate dataframe schema.
        
        Args:
            df: Input dataframe
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check required columns
        missing_columns = set(self.required_columns) - set(df.columns)
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
        
        # Check minimum number of rows
        if len(df) < self.min_rows:
            errors.append(f"Insufficient data: {len(df)} rows (minimum: {self.min_rows})")
        
        # Check for empty dataframe
        if df.empty:
            errors.append("Dataframe is empty")
        
        is_valid = len(errors) == 0
        logger.info(f"Schema validation: {'PASSED' if is_valid else 'FAILED'}")
        
        return is_valid, errors
    
    def validate_data_quality(self, df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate data quality.
        
        Args:
            df: Input dataframe
            
        Returns:
            Tuple of (is_valid, quality_metrics)
        """
        quality_metrics = {}
        errors = []
        
        # Missing values check
        missing_percentage = (df.isnull().sum() / len(df)).max()
        quality_metrics['max_missing_percentage'] = missing_percentage
        
        if missing_percentage > self.max_missing_percentage:
            errors.append(f"Too many missing values: {missing_percentage:.2%} (max: {self.max_missing_percentage:.2%})")
        
        # Duplicate rows check
        duplicate_count = df.duplicated().sum()
        duplicate_percentage = duplicate_count / len(df)
        quality_metrics['duplicate_percentage'] = duplicate_percentage
        quality_metrics['duplicate_count'] = duplicate_count
        
        # Data type consistency
        quality_metrics['data_types'] = df.dtypes.to_dict()
        
        # Numerical columns statistics
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            col_stats = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'missing_count': df[col].isnull().sum()
            }
            quality_metrics[f'{col}_stats'] = col_stats
            
            # Check for outliers (using IQR method)
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
            quality_metrics[f'{col}_outliers'] = outliers
        
        # Categorical columns statistics
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            col_stats = {
                'unique_count': df[col].nunique(),
                'most_frequent': df[col].mode().iloc[0] if not df[col].mode().empty else None,
                'missing_count': df[col].isnull().sum()
            }
            quality_metrics[f'{col}_stats'] = col_stats
        
        # Overall quality score
        quality_score = 1.0 - missing_percentage - (duplicate_percentage * 0.5)
        quality_metrics['overall_quality_score'] = max(0.0, quality_score)
        
        is_valid = len(errors) == 0
        logger.info(f"Data quality validation: {'PASSED' if is_valid else 'FAILED'}")
        
        return is_valid, quality_metrics
    
    def validate_target_distribution(self, df: pd.DataFrame, target_col: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate target variable distribution.
        
        Args:
            df: Input dataframe
            target_col: Target column name
            
        Returns:
            Tuple of (is_valid, distribution_metrics)
        """
        if target_col not in df.columns:
            return False, {'error': f'Target column {target_col} not found'}
        
        distribution_metrics = {}
        errors = []
        
        target_series = df[target_col]
        
        # Basic statistics
        distribution_metrics['unique_values'] = target_series.nunique()
        distribution_metrics['value_counts'] = target_series.value_counts().to_dict()
        distribution_metrics['missing_count'] = target_series.isnull().sum()
        
        # Check for class imbalance (for classification)
        if target_series.nunique() <= 10:  # Assume classification if few unique values
            value_counts = target_series.value_counts()
            min_class_ratio = value_counts.min() / value_counts.max()
            distribution_metrics['class_imbalance_ratio'] = min_class_ratio
            
            if min_class_ratio < 0.1:  # Less than 10% representation
                errors.append(f"Severe class imbalance detected: {min_class_ratio:.2%}")
        
        # Check for target leakage (constant values)
        if target_series.nunique() == 1:
            errors.append("Target variable has only one unique value")
        
        is_valid = len(errors) == 0
        logger.info(f"Target distribution validation: {'PASSED' if is_valid else 'FAILED'}")
        
        return is_valid, distribution_metrics
    
    def validate_feature_correlations(self, df: pd.DataFrame, target_col: str, 
                                    correlation_threshold: float = 0.95) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate feature correlations.
        
        Args:
            df: Input dataframe
            target_col: Target column name
            correlation_threshold: Threshold for high correlation
            
        Returns:
            Tuple of (is_valid, correlation_metrics)
        """
        correlation_metrics = {}
        errors = []
        
        # Get numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numerical_cols:
            numerical_cols.remove(target_col)
        
        if len(numerical_cols) < 2:
            return True, {'message': 'Not enough numerical features for correlation analysis'}
        
        # Calculate correlation matrix
        corr_matrix = df[numerical_cols].corr()
        
        # Find highly correlated features
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = abs(corr_matrix.iloc[i, j])
                if corr_value > correlation_threshold:
                    high_corr_pairs.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        correlation_metrics['high_correlation_pairs'] = high_corr_pairs
        correlation_metrics['correlation_threshold'] = correlation_threshold
        
        if high_corr_pairs:
            errors.append(f"Found {len(high_corr_pairs)} highly correlated feature pairs")
        
        # Calculate feature-target correlations
        if target_col in df.columns and df[target_col].dtype in [np.number]:
            target_correlations = df[numerical_cols + [target_col]].corr()[target_col].drop(target_col)
            correlation_metrics['target_correlations'] = target_correlations.to_dict()
        
        is_valid = len(errors) == 0
        logger.info(f"Feature correlation validation: {'PASSED' if is_valid else 'FAILED'}")
        
        return is_valid, correlation_metrics
    
    def run_full_validation(self, df: pd.DataFrame, target_col: str = None) -> Dict[str, Any]:
        """
        Run full data validation suite.
        
        Args:
            df: Input dataframe
            target_col: Target column name (optional)
            
        Returns:
            Validation results dictionary
        """
        logger.info("Starting full data validation")
        
        validation_results = {
            'timestamp': datetime.now().isoformat(),
            'dataset_shape': df.shape,
            'validation_passed': True,
            'errors': [],
            'warnings': []
        }
        
        # Schema validation
        schema_valid, schema_errors = self.validate_schema(df)
        validation_results['schema_validation'] = {
            'passed': schema_valid,
            'errors': schema_errors
        }
        if not schema_valid:
            validation_results['validation_passed'] = False
            validation_results['errors'].extend(schema_errors)
        
        # Data quality validation
        quality_valid, quality_metrics = self.validate_data_quality(df)
        validation_results['quality_validation'] = {
            'passed': quality_valid,
            'metrics': quality_metrics
        }
        if not quality_valid:
            validation_results['validation_passed'] = False
        
        # Target distribution validation (if target column provided)
        if target_col:
            target_valid, target_metrics = self.validate_target_distribution(df, target_col)
            validation_results['target_validation'] = {
                'passed': target_valid,
                'metrics': target_metrics
            }
            if not target_valid:
                validation_results['validation_passed'] = False
            
            # Feature correlation validation
            corr_valid, corr_metrics = self.validate_feature_correlations(df, target_col)
            validation_results['correlation_validation'] = {
                'passed': corr_valid,
                'metrics': corr_metrics
            }
            if not corr_valid:
                validation_results['warnings'].extend(['High feature correlations detected'])
        
        # Log validation results to database
        self._log_validation_results(validation_results)
        
        logger.info(f"Data validation completed: {'PASSED' if validation_results['validation_passed'] else 'FAILED'}")
        
        return validation_results
    
    def _log_validation_results(self, results: Dict[str, Any]):
        """
        Log validation results to database.
        
        Args:
            results: Validation results dictionary
        """
        # Extract key metrics for database logging
        metrics = {
            'validation_passed': int(results['validation_passed']),
            'error_count': len(results['errors']),
            'warning_count': len(results['warnings']),
            'dataset_rows': results['dataset_shape'][0],
            'dataset_columns': results['dataset_shape'][1]
        }
        
        # Add quality metrics if available
        if 'quality_validation' in results:
            quality_metrics = results['quality_validation']['metrics']
            if 'overall_quality_score' in quality_metrics:
                metrics['quality_score'] = quality_metrics['overall_quality_score']
            if 'max_missing_percentage' in quality_metrics:
                metrics['missing_percentage'] = quality_metrics['max_missing_percentage']
            if 'duplicate_percentage' in quality_metrics:
                metrics['duplicate_percentage'] = quality_metrics['duplicate_percentage']
        
        # Log to database
        db_manager.log_data_quality(
            dataset_name="validation_results",
            metrics=metrics,
            data_version=get_timestamp()
        )


def main():
    """Main function for data validation CLI."""
    import argparse
    from src.utils.helpers import load_config
    
    parser = argparse.ArgumentParser(description="Data Validation")
    parser.add_argument("--input", required=True, help="Input data file")
    parser.add_argument("--target", help="Target column name")
    parser.add_argument("--config", default="config/config.yaml", help="Configuration file")
    parser.add_argument("--output", help="Output validation report file")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Load data
    df = pd.read_csv(args.input)
    
    # Initialize validator
    validator = DataValidator(config)
    
    # Run validation
    results = validator.run_full_validation(df, args.target)
    
    # Save results if output specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Validation results saved to {args.output}")
    
    # Print summary
    print(f"Validation {'PASSED' if results['validation_passed'] else 'FAILED'}")
    if results['errors']:
        print(f"Errors: {len(results['errors'])}")
        for error in results['errors']:
            print(f"  - {error}")
    if results['warnings']:
        print(f"Warnings: {len(results['warnings'])}")
        for warning in results['warnings']:
            print(f"  - {warning}")


if __name__ == "__main__":
    main()
