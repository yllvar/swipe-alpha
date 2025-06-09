import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from typing import Dict, List

class DatingPortfolioOptimizer:
    def __init__(self, historical_data: pd.DataFrame):
        """
        Initialize with historical match data
        
        Args:
            historical_data: DataFrame containing match history with features
                            like reply_rate, date_rate, message_length, etc.
        """
        self.data = historical_data
        self.features = ['reply_rate', 'date_rate', 'message_length', 'response_time']
        
    def calculate_returns(self) -> pd.Series:
        """Calculate expected returns for each match type"""
        # Composite score based on engagement metrics
        self.data['expected_return'] = (
            0.4 * self.data['reply_rate'] +
            0.3 * self.data['date_rate'] +
            0.2 * (1 - self.data['response_time']) +
            0.1 * np.log(self.data['message_length'])
        )
        return self.data['expected_return']
    
    def calculate_covariance(self) -> pd.DataFrame:
        """Calculate covariance matrix between different profile types"""
        return self.data[self.features].cov()
    
    def optimize_portfolio(self) -> Dict:
        """Optimize match portfolio using Markowitz mean-variance optimization"""
        mu = self.calculate_returns()
        S = self.calculate_covariance()
        
        # Use PyPortfolioOpt for optimization
        ef = EfficientFrontier(mu, S)
        weights = ef.max_sharpe()
        cleaned_weights = ef.clean_weights()
        
        # Get performance metrics
        performance = ef.portfolio_performance(verbose=True)
        
        return {
            'weights': cleaned_weights,
            'expected_return': performance[0],
            'volatility': performance[1],
            'sharpe_ratio': performance[2]
        }
    
    def black_litterman_adjustment(self, prior_returns: pd.Series, views: Dict) -> Dict:
        """
        Apply Black-Litterman model for subjective views
        
        Args:
            prior_returns: Prior expected returns
            views: Dictionary of views on certain assets
                   Example: {'profile_type1': 0.2, 'profile_type2': -0.1}
        """
        from pypfopt.black_litterman import BlackLittermanModel
        from pypfopt import objective_functions
        
        # Create covariance matrix
        cov_matrix = self.calculate_covariance()
        
        # Initialize BL model
        bl = BlackLittermanModel(cov_matrix, pi=prior_returns, absolute_views=views)
        
        # Posterior estimate of returns
        ret_bl = bl.bl_returns()
        
        # Optimize portfolio with BL returns
        ef = EfficientFrontier(ret_bl, cov_matrix)
        ef.add_objective(objective_functions.L2_reg)
        weights = ef.max_sharpe()
        
        return {
            'bl_returns': ret_bl,
            'weights': ef.clean_weights(),
            'performance': ef.portfolio_performance()
        }