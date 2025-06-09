import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Dict

class DatingOptionsPricing:
    def __init__(self, historical_data: pd.DataFrame):
        """
        Initialize options pricing model for dating matches
        
        Args:
            historical_data: DataFrame containing match history with timestamps
        """
        self.data = historical_data
        
    def calculate_time_decay(self, current_time: pd.Timestamp, match_time: pd.Timestamp) -> float:
        """Calculate time decay factor for a match"""
        hours_passed = (current_time - match_time).total_seconds() / 3600
        return np.exp(-0.1 * hours_passed)  # Exponential decay
    
    def black_scholes(self, S: float, K: float, T: float, r: float, sigma: float) -> Dict:
        """
        Calculate Black-Scholes option prices for dating matches
        
        Args:
            S: Current match "price" (interest level)
            K: Strike price (minimum acceptable interest)
            T: Time to expiration (in years fraction)
            r: Risk-free rate (base interest decay)
            sigma: Volatility of interest
        """
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        return {
            'call_price': call_price,  # Value of pursuing the match
            'put_price': put_price,    # Value of letting it expire
            'delta': norm.cdf(d1),      # Sensitivity to interest changes
            'theta': - (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - 
                    r * K * np.exp(-r * T) * norm.cdf(d2)  # Time decay
        }
    
    def price_match_options(self, match_data: Dict) -> Dict:
        """
        Price match options based on interaction data
        
        Args:
            match_data: Dictionary containing match information
                        (last_message_time, message_count, reply_rate, etc.)
        """
        current_time = pd.Timestamp.now()
        match_time = pd.to_datetime(match_data['last_message_time'])
        
        # Calculate parameters
        S = match_data.get('reply_rate', 0.5)  # Current interest level
        K = 0.3  # Minimum acceptable interest
        T = (7 - (current_time - match_time).days) / 365  # Time decay over a week
        r = 0.05  # Base decay rate
        sigma = 0.2  # Volatility estimate
        
        # Calculate option prices
        option_values = self.black_scholes(S, K, T, r, sigma)
        
        return {
            **option_values,
            'time_decay': self.calculate_time_decay(current_time, match_time),
            'recommendation': 'message' if option_values['call_price'] > 0.5 else 'wait'
        }
    
    def evaluate_portfolio(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Evaluate all matches using options pricing model"""
        results = []
        for _, row in matches.iterrows():
            if pd.isna(row['last_message_time']):
                continue
                
            match_data = {
                'reply_rate': row.get('reply_rate', 0.5),
                'last_message_time': row['last_message_time'],
                'message_count': row.get('message_count', 1)
            }
            
            pricing = self.price_match_options(match_data)
            results.append({
                'match_id': row['match_id'],
                **pricing
            })
        
        return pd.DataFrame(results)