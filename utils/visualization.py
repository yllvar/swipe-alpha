import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict

def plot_portfolio_performance(portfolio: Dict) -> None:
    """Visualize portfolio optimization results"""
    weights = pd.Series(portfolio['weights'])
    weights = weights[weights > 0].sort_values(ascending=False)
    
    plt.figure(figsize=(12, 6))
    
    # Plot weights
    plt.subplot(1, 2, 1)
    weights.plot(kind='bar', color='skyblue')
    plt.title('Optimal Portfolio Weights')
    plt.ylabel('Weight')
    plt.xticks(rotation=45)
    
    # Plot performance metrics
    plt.subplot(1, 2, 2)
    metrics = pd.Series({
        'Expected Return': portfolio['expected_return'],
        'Volatility': portfolio['volatility'],
        'Sharpe Ratio': portfolio['sharpe_ratio']
    })
    metrics.plot(kind='bar', color=['green', 'red', 'blue'])
    plt.title('Portfolio Performance')
    
    plt.tight_layout()
    plt.show()

def plot_alpha_feature_importance(importances: pd.Series, feature_names: List[str]) -> None:
    """Plot feature importances from alpha model"""
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances, y=feature_names, palette='viridis')
    plt.title('Alpha Signal Feature Importances')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.show()

def plot_monte_carlo_results(results: pd.DataFrame) -> None:
    """Visualize Monte Carlo simulation results"""
    plt.figure(figsize=(10, 6))
    
    # Plot distribution of success rates
    sns.histplot(results['success'], kde=True, stat='probability')
    plt.title('Distribution of Successful Connections')
    plt.xlabel('Success (1 = dated without ghosting)')
    plt.ylabel('Probability')
    
    plt.show()