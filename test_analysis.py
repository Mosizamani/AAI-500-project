"""
Unit tests for the Himalayan expedition analysis module.

Run with: python -m pytest test_analysis.py -v
"""

import pytest
import pandas as pd
import numpy as np
from analysis_utils import ExpeditionAnalyzer


class TestExpeditionAnalyzer:
    """Test cases for the ExpeditionAnalyzer class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.analyzer = ExpeditionAnalyzer()
        
        # Create sample test data
        self.sample_data = pd.DataFrame({
            'peak_id': ['EVER', 'EVER', 'AMAD', 'AMAD'],
            'mbrs_summited': [2, 0, 1, 3],
            'max_elev_reached': [8848, 7500, 6812, 6812],
            'season': ['Spring', 'Autumn', 'Spring', 'Winter'],
            'is_o2_climbing': [1, 0, 1, 1],
            'total_mbrs': [10, 8, 5, 7],
            'year': [2020, 2021, 2019, 2020]
        })
    
    def test_validate_data_success(self):
        """Test data validation with valid data."""
        validation = self.analyzer.validate_data(self.sample_data)
        
        assert len(validation['errors']) == 0
        assert 'Dataset shape:' in validation['info'][0]
    
    def test_validate_data_missing_columns(self):
        """Test data validation with missing required columns."""
        incomplete_data = self.sample_data.drop(columns=['mbrs_summited'])
        validation = self.analyzer.validate_data(incomplete_data)
        
        assert len(validation['errors']) > 0
        assert 'Missing required columns' in validation['errors'][0]
    
    def test_validate_data_negative_elevation(self):
        """Test data validation with invalid elevation values."""
        invalid_data = self.sample_data.copy()
        invalid_data.loc[0, 'max_elev_reached'] = -100
        validation = self.analyzer.validate_data(invalid_data)
        
        assert any('negative values' in error for error in validation['errors'])
    
    def test_preprocess_data_basic(self):
        """Test basic data preprocessing."""
        processed = self.analyzer.preprocess_data(self.sample_data, altitude_threshold=6000)
        
        # Check that success column was created
        assert 'success' in processed.columns
        assert 'mbrs_summited' not in processed.columns
        
        # Check altitude filtering
        assert len(processed) == 4  # All samples above 6000m
        
        # Check one-hot encoding of season
        season_cols = [col for col in processed.columns if col.startswith('season_')]
        assert len(season_cols) > 0
    
    def test_preprocess_data_altitude_filtering(self):
        """Test altitude threshold filtering."""
        processed = self.analyzer.preprocess_data(self.sample_data, altitude_threshold=7000)
        
        # Should only include expeditions above 7000m (EVER samples)
        assert len(processed) == 2
    
    def test_preprocess_data_success_variable(self):
        """Test creation of binary success variable."""
        processed = self.analyzer.preprocess_data(self.sample_data, altitude_threshold=6000)
        
        # Check success variable is binary
        assert set(processed['success'].unique()) <= {0, 1}
        
        # Verify logic: mbrs_summited > 0 should be success = 1
        # Original data: [2, 0, 1, 3] -> expected success: [1, 0, 1, 1]
        expected_success = [1, 0, 1, 1]
        assert processed['success'].tolist() == expected_success
    
    def test_bad_features_removal(self):
        """Test that bad features are properly removed."""
        processed = self.analyzer.preprocess_data(self.sample_data, altitude_threshold=6000)
        
        # Check that bad features were removed
        bad_features_present = [feat for feat in self.analyzer.bad_features 
                              if feat in processed.columns]
        assert len(bad_features_present) == 0
    
    def test_rank_features_basic(self):
        """Test feature ranking functionality."""
        # Create simple test data for ranking
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [5, 4, 3, 2, 1],  # Inversely correlated
            'feature3': [1, 1, 1, 1, 1]   # Constant (should rank low)
        })
        y = pd.Series([0, 0, 1, 1, 1])
        
        ranking = self.analyzer.rank_features(X, y)
        
        # Check output structure
        assert isinstance(ranking, pd.DataFrame)
        assert 'Feature' in ranking.columns
        assert 'F-Score' in ranking.columns
        assert 'P-Value' in ranking.columns
        assert len(ranking) == 3
        
        # Constant feature should have lowest F-score
        constant_feature_score = ranking[ranking['Feature'] == 'feature3']['F-Score'].iloc[0]
        assert constant_feature_score < ranking['F-Score'].max()


class TestDataIntegrity:
    """Test data integrity and edge cases."""
    
    def test_empty_dataframe(self):
        """Test handling of empty dataframe."""
        analyzer = ExpeditionAnalyzer()
        empty_df = pd.DataFrame()
        
        validation = analyzer.validate_data(empty_df)
        assert len(validation['errors']) > 0
    
    def test_all_missing_values(self):
        """Test handling of dataframe with all missing values."""
        analyzer = ExpeditionAnalyzer()
        missing_df = pd.DataFrame({
            'mbrs_summited': [np.nan, np.nan],
            'max_elev_reached': [np.nan, np.nan],
            'season': [np.nan, np.nan]
        })
        
        validation = analyzer.validate_data(missing_df)
        assert len(validation['warnings']) > 0


@pytest.fixture
def sample_expeditions_data():
    """Fixture providing realistic sample expedition data."""
    return pd.DataFrame({
        'peak_id': ['EVER'] * 100,
        'mbrs_summited': np.random.randint(0, 10, 100),
        'max_elev_reached': np.random.randint(7000, 8850, 100),
        'season': np.random.choice(['Spring', 'Autumn', 'Summer', 'Winter'], 100),
        'is_o2_climbing': np.random.randint(0, 2, 100),
        'total_mbrs': np.random.randint(1, 20, 100),
        'year': np.random.randint(1990, 2023, 100)
    })


def test_full_pipeline_integration(sample_expeditions_data):
    """Integration test for the full analysis pipeline."""
    analyzer = ExpeditionAnalyzer()
    
    # Test full preprocessing pipeline
    processed = analyzer.preprocess_data(sample_expeditions_data, altitude_threshold=7000)
    
    # Verify processed data structure
    assert 'success' in processed.columns
    assert len(processed) > 0
    assert processed['success'].dtype == int
    
    # Test feature ranking
    y = processed['success']
    X = processed.drop(columns=['success'])
    
    if len(X.columns) > 0:  # Only test if features remain after preprocessing
        ranking = analyzer.rank_features(X, y)
        assert len(ranking) == len(X.columns)


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])