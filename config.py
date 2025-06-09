import json
import os
from typing import Dict

class Config:
    def __init__(self, config_file: str = 'config.json'):
        """
        Initialize configuration
        
        Args:
            config_file: Path to JSON configuration file
        """
        self.config_file = config_file
        self.data = self._load_config()
        
    def _load_config(self) -> Dict:
        """Load configuration from file"""
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"Config file {self.config_file} not found")
            
        with open(self.config_file) as f:
            return json.load(f)
    
    def get(self, key: str, default=None):
        """Get configuration value"""
        return self.data.get(key, default)
    
    def get_tinder_credentials(self) -> Dict:
        """Get Tinder API credentials"""
        return {
            'facebook_token': self.get('facebook_token'),
            'facebook_id': self.get('facebook_id')
        }
    
    def get_analysis_settings(self) -> Dict:
        """Get analysis configuration"""
        return {
            'swipe_limit': self.get('swipe_limit', 100),
            'match_limit': self.get('match_limit', 100),
            'simulation_days': self.get('simulation_days', 30),
            'decay_rate': self.get('decay_rate', 0.1)
        }