import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from tqdm import tqdm

class DatingMonteCarlo:
    def __init__(self, historical_data: pd.DataFrame):
        """
        Initialize Monte Carlo simulator with historical data
        
        Args:
            historical_data: DataFrame containing past match outcomes
        """
        self.data = historical_data
        self.simulation_results = None
        
    def simulate_conversation_outcomes(self, n_simulations: int = 1000) -> Dict:
        """Simulate conversation outcomes based on historical patterns"""
        # Calculate probabilities from historical data
        reply_prob = self.data['replied'].mean()
        date_prob = self.data[self.data['replied']]['dated'].mean()
        ghost_prob = self.data[self.data['dated']]['ghosted'].mean()
        
        # Run simulations
        results = []
        for _ in tqdm(range(n_simulations)):
            # Simulate conversation path
            replied = np.random.binomial(1, reply_prob)
            dated = replied and np.random.binomial(1, date_prob)
            ghosted = dated and np.random.binomial(1, ghost_prob)
            
            results.append({
                'replied': replied,
                'dated': dated,
                'ghosted': ghosted,
                'success': dated and not ghosted
            })
        
        self.simulation_results = pd.DataFrame(results)
        return self.simulation_results
    
    def optimize_message_strategy(self, strategies: List[Dict]) -> Dict:
        """
        Compare different messaging strategies via simulation
        
        Args:
            strategies: List of strategy dictionaries with parameters
                        Example: [{'opener_type': 'question', 'length': 'short', ...}]
        """
        strategy_results = {}
        
        for strategy in strategies:
            # Adjust probabilities based on strategy parameters
            base_reply = self.data['replied'].mean()
            
            # Example adjustments (would be data-driven in real implementation)
            if strategy.get('opener_type') == 'question':
                reply_prob = base_reply * 1.2  # 20% boost for questions
            elif strategy.get('opener_type') == 'humor':
                reply_prob = base_reply * 1.15
            else:
                reply_prob = base_reply
            
            if strategy.get('length') == 'medium':
                reply_prob *= 1.1
            
            # Run simulations with adjusted probabilities
            results = []
            for _ in range(100):  # 100 sims per strategy
                replied = np.random.binomial(1, min(reply_prob, 0.99))  # Cap at 99%
                dated = replied and np.random.binomial(1, self.data[self.data['replied']]['dated'].mean())
                ghosted = dated and np.random.binomial(1, self.data[self.data['dated']]['ghosted'].mean())
                
                results.append({
                    'reply_rate': replied,
                    'date_rate': dated,
                    'ghost_rate': ghosted,
                    'success_rate': dated and not ghosted
                })
            
            # Aggregate results
            df = pd.DataFrame(results)
            strategy_results[str(strategy)] = {
                'mean_reply': df['reply_rate'].mean(),
                'mean_date': df['date_rate'].mean(),
                'mean_ghost': df['ghost_rate'].mean(),
                'mean_success': df['success_rate'].mean(),
                'std_success': df['success_rate'].std()
            }
        
        return strategy_results
    
    def time_value_simulation(self, decay_rate: float = 0.1) -> pd.DataFrame:
        """
        Simulate the time-value decay of matches
        
        Args:
            decay_rate: Exponential decay rate for match interest over time
        """
        # Generate time series data
        days = 30
        time_values = []
        
        for day in range(days):
            # Probability decays exponentially over time
            prob = np.exp(-decay_rate * day)
            replies = np.random.binomial(1, prob, size=1000)  # 1000 simulations per day
            time_values.append({
                'day': day,
                'reply_prob': prob,
                'mean_replies': replies.mean(),
                'std_replies': replies.std()
            })
        
        return pd.DataFrame(time_values)