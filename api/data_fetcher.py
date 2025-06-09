import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List
import time

class TinderDataFetcher:
    def __init__(self, client):
        """
        Initialize data fetcher with Tinder client
        
        Args:
            client: Authenticated TinderClient instance
        """
        self.client = client
        
    def fetch_comprehensive_data(self, days: int = 30) -> Dict:
        """
        Fetch comprehensive data including users, matches, and messages
        
        Args:
            days: Number of days of history to fetch
        """
        print("Fetching comprehensive Tinder data...")
        
        # Get nearby users
        users = self.client.get_nearby_users(limit=100)
        
        # Get match history with messages
        matches = self.client.get_match_history(limit=100)
        
        # Get swipe history (approximated)
        swipe_history = self._generate_swipe_history(days)
        
        return {
            'users': pd.DataFrame(users),
            'matches': pd.DataFrame(matches),
            'swipes': pd.DataFrame(swipe_history),
            'fetched_at': datetime.now().isoformat()
        }
    
    def _generate_swipe_history(self, days: int) -> List[Dict]:
        """Generate simulated swipe history (Tinder API doesn't provide this directly)"""
        history = []
        for i in range(days * 20):  # Assume 20 swipes per day
            history.append({
                'timestamp': (datetime.now() - timedelta(days=days) + 
                             timedelta(hours=i)).isoformat(),
                'direction': 'right' if i % 3 != 0 else 'left',  # 66% right swipe rate
                'time_of_day': ['morning', 'afternoon', 'evening', 'night'][i % 4],
                'day_of_week': (datetime.now() - timedelta(days=days) + 
                               timedelta(hours=i)).strftime('%A')
            })
        return history
    
    def fetch_realtime_updates(self, last_fetch: str) -> Dict:
        """
        Fetch new data since last fetch time
        
        Args:
            last_fetch: ISO format timestamp of last fetch
        """
        last_time = datetime.fromisoformat(last_fetch)
        now = datetime.now()
        
        # Simulate new data (in a real app, would use actual API)
        new_matches = []
        for i in range(3):  # 3 new matches
            new_matches.append({
                'match_id': f"new_{i}_{now.timestamp()}",
                'person_id': f"person_{i}",
                'name': f"New Match {i}",
                'created_at': now.isoformat(),
                'is_super_like': False,
                'is_boost_match': False,
                'is_fast_match': True,
                'message_count': 0
            })
        
        return {
            'new_matches': pd.DataFrame(new_matches),
            'new_messages': pd.DataFrame(),  # Would be populated in real implementation
            'new_swipes': pd.DataFrame()     # Would be populated in real implementation
        }