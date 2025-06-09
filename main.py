import pandas as pd
import numpy as np
from datetime import datetime
from api.client import TinderClient
from core.portfolio_optimizer import DatingPortfolioOptimizer
from core.alpha_model import AlphaSignalModel
from core.monte_carlo import DatingMonteCarlo
from utils.visualization import plot_portfolio_performance
import json
import os

class SwipeAlpha:
    def __init__(self, config_path: str = 'config.json'):
        """
        Main application class for Swipe Alpha
        
        Args:
            config_path: Path to configuration file with API credentials
        """
        self.config = self.load_config(config_path)
        self.client = None
        self.data = None
        self.portfolio = None
        self.alpha_model = None
        
    def load_config(self, path: str) -> Dict:
        """Load configuration file"""
        with open(path) as f:
            return json.load(f)
    
    def authenticate(self) -> None:
        """Authenticate with Tinder API"""
        self.client = TinderClient(
            fb_token=self.config['facebook_token'],
            fb_id=self.config['facebook_id']
        )
        print(f"Authenticated as {self.client.user.name}")
    
    def collect_data(self, user_limit: int = 100, match_limit: int = 100) -> None:
        """Collect initial data from Tinder"""
        print("Collecting nearby users...")
        users = self.client.get_nearby_users(limit=user_limit)
        
        print("Collecting match history...")
        matches = self.client.get_match_history(limit=match_limit)
        
        # Create DataFrame
        users_df = pd.DataFrame(users)
        matches_df = pd.DataFrame(matches)
        
        # Calculate some metrics
        if len(matches_df) > 0:
            matches_df['message_count'] = matches_df['messages'].apply(len)
            matches_df['replied'] = matches_df['message_count'] > 1
            matches_df['response_time'] = matches_df['messages'].apply(
                lambda x: (datetime.fromisoformat(x[1]['sent_at']) - 
                          datetime.fromisoformat(x[0]['sent_at'])).total_seconds() / 3600
                if len(x) > 1 else np.nan
            )
        
        self.data = {
            'users': users_df,
            'matches': matches_df
        }
        
        # Save raw data
        os.makedirs('data/raw', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        users_df.to_csv(f'data/raw/users_{timestamp}.csv', index=False)
        matches_df.to_csv(f'data/raw/matches_{timestamp}.csv', index=False)
        
        print(f"Collected {len(users_df)} users and {len(matches_df)} matches")
    
    def optimize_portfolio(self) -> None:
        """Run portfolio optimization on matches"""
        if not self.data or 'matches' not in self.data:
            raise ValueError("No match data available. Run collect_data() first.")
        
        # Prepare data for optimization
        match_df = self.data['matches'].copy()
        
        # Calculate some metrics (in a real app, these would be more sophisticated)
        match_df['reply_rate'] = match_df['replied'].astype(float)
        match_df['date_rate'] = np.random.uniform(0, 0.5, len(match_df))  # Placeholder
        match_df['message_length'] = match_df['messages'].apply(
            lambda x: np.mean([len(m['body']) for m in x]) if x else 0
        )
        
        # Initialize and run optimizer
        optimizer = DatingPortfolioOptimizer(match_df)
        self.portfolio = optimizer.optimize_portfolio()
        
        print("\nPortfolio Optimization Results:")
        print(f"Expected Return: {self.portfolio['expected_return']:.2%}")
        print(f"Volatility: {self.portfolio['volatility']:.2%}")
        print(f"Sharpe Ratio: {self.portfolio['sharpe_ratio']:.2f}")
        
        # Visualize results
        plot_portfolio_performance(self.portfolio)
    
    def train_alpha_model(self) -> None:
        """Train model to detect alpha signals in profiles"""
        if not self.data or 'matches' not in self.data:
            raise ValueError("No match data available. Run collect_data() first.")
        
        # Prepare data (in a real app, this would be more comprehensive)
        df = self.data['matches'].copy()
        
        # Add some placeholder features for demonstration
        df['bio'] = df['name'].apply(lambda x: f"Sample bio for {x}")
        df['message_text'] = df['messages'].apply(
            lambda x: ' '.join([m['body'] for m in x]) if x else ''
        )
        df['has_job'] = np.random.binomial(1, 0.7, len(df))
        df['has_education'] = np.random.binomial(1, 0.8, len(df))
        df['spotify_connected'] = np.random.binomial(1, 0.4, len(df))
        df['date_rate'] = np.random.uniform(0, 1, len(df))  # Placeholder target
        
        # Train model
        self.alpha_model = AlphaSignalModel(df)
        results = self.alpha_model.train_model()
        
        print("\nAlpha Model Training Results:")
        print(f"Train Accuracy: {results['train_accuracy']:.2%}")
        print(f"Test Accuracy: {results['test_accuracy']:.2%}")
    
    def run_monte_carlo_simulations(self) -> None:
        """Run Monte Carlo simulations for dating scenarios"""
        if not self.data or 'matches' not in self.data:
            raise ValueError("No match data available. Run collect_data() first.")
        
        # Prepare data
        df = self.data['matches'].copy()
        df['replied'] = df['message_count'] > 1
        df['dated'] = np.random.binomial(1, 0.3, len(df))  # Placeholder
        df['ghosted'] = np.where(
            df['dated'], 
            np.random.binomial(1, 0.4, len(df)),  # 40% ghost rate after dates
            np.nan
        )
        
        # Run simulations
        simulator = DatingMonteCarlo(df)
        sim_results = simulator.simulate_conversation_outcomes(n_simulations=1000)
        
        print("\nMonte Carlo Simulation Results:")
        print(f"Average Reply Rate: {sim_results['replied'].mean():.2%}")
        print(f"Average Date Rate: {sim_results['dated'].mean():.2%}")
        print(f"Average Ghost Rate: {sim_results['ghosted'].mean():.2%}")
        print(f"Average Success Rate: {sim_results['success'].mean():.2%}")
        
        # Compare strategies
        strategies = [
            {'opener_type': 'question', 'length': 'medium', 'time': 'evening'},
            {'opener_type': 'humor', 'length': 'short', 'time': 'night'},
            {'opener_type': 'compliment', 'length': 'long', 'time': 'afternoon'}
        ]
        
        strategy_results = simulator.optimize_message_strategy(strategies)
        print("\nStrategy Comparison:")
        for strategy, metrics in strategy_results.items():
            print(f"\nStrategy: {strategy}")
            print(f"Success Rate: {metrics['mean_success']:.2%} ± {metrics['std_success']:.2%}")
    
    def run(self) -> None:
        """Main execution method"""
        try:
            print("=== Swipe Alpha - Quantitative Dating Optimization ===")
            
            # Step 1: Authenticate
            self.authenticate()
            
            # Step 2: Collect data
            self.collect_data(user_limit=50, match_limit=50)
            
            # Step 3: Portfolio optimization
            self.optimize_portfolio()
            
            # Step 4: Alpha signal detection
            self.train_alpha_model()
            
            # Step 5: Monte Carlo simulations
            self.run_monte_carlo_simulations()
            
            print("\n=== Analysis Complete ===")
            
        except Exception as e:
            print(f"Error running Swipe Alpha: {e}")

if __name__ == "__main__":
    app = SwipeAlpha()
    app.run()