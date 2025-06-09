import pandas as pd
import numpy as np
from typing import Dict, List

class DatingMetricsCalculator:
    def __init__(self, data: Dict):
        """
        Initialize with dating data
        
        Args:
            data: Dictionary containing users, matches, swipes DataFrames
        """
        self.users = data.get('users', pd.DataFrame())
        self.matches = data.get('matches', pd.DataFrame())
        self.swipes = data.get('swipes', pd.DataFrame())
        
    def calculate_basic_metrics(self) -> Dict:
        """Calculate basic dating metrics"""
        match_rate = len(self.matches) / len(self.swipes[self.swipes['direction'] == 'right']) \
                    if len(self.swipes) > 0 else 0
                    
        reply_rate = self.matches['message_count'].apply(lambda x: 1 if x > 1 else 0).mean() \
                    if len(self.matches) > 0 else 0
                    
        return {
            'total_swipes': len(self.swipes),
            'right_swipes': len(self.swipes[self.swipes['direction'] == 'right']),
            'left_swipes': len(self.swipes[self.swipes['direction'] == 'left']),
            'total_matches': len(self.matches),
            'match_rate': match_rate,
            'average_reply_rate': reply_rate,
            'average_messages_per_match': self.matches['message_count'].mean() \
                                         if len(self.matches) > 0 else 0,
            'super_like_rate': self.matches['is_super_like'].mean() \
                              if len(self.matches) > 0 else 0
        }
    
    def calculate_time_based_metrics(self) -> Dict:
        """Calculate metrics by time of day and day of week"""
        if len(self.swipes) == 0:
            return {}
            
        time_metrics = {}
        
        # Swipe metrics by time of day
        for time_of_day in ['morning', 'afternoon', 'evening', 'night']:
            subset = self.swipes[self.swipes['time_of_day'] == time_of_day]
            if len(subset) > 0:
                time_metrics[f'{time_of_day}_swipes'] = len(subset)
                time_metrics[f'{time_of_day}_match_rate'] = \
                    len(self.matches) / len(subset[subset['direction'] == 'right']) \
                    if len(subset[subset['direction'] == 'right']) > 0 else 0
        
        # Swipe metrics by day of week
        for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                   'Friday', 'Saturday', 'Sunday']:
            subset = self.swipes[self.swipes['day_of_week'] == day]
            if len(subset) > 0:
                time_metrics[f'{day}_swipes'] = len(subset)
                time_metrics[f'{day}_match_rate'] = \
                    len(self.matches) / len(subset[subset['direction'] == 'right']) \
                    if len(subset[subset['direction'] == 'right']) > 0 else 0
        
        return time_metrics
    
    def calculate_conversation_metrics(self) -> Dict:
        """Calculate metrics about message quality"""
        if len(self.matches) == 0:
            return {}
            
        # Calculate response times
        response_times = []
        for _, match in self.matches.iterrows():
            if len(match['messages']) > 1:
                first_msg = match['messages'][0]['sent_at']
                second_msg = match['messages'][1]['sent_at']
                response_time = (datetime.fromisoformat(second_msg) - 
                                datetime.fromisoformat(first_msg)).total_seconds() / 3600
                response_times.append(response_time)
        
        return {
            'average_response_time_hours': np.mean(response_times) if response_times else 0,
            'median_response_time_hours': np.median(response_times) if response_times else 0,
            'message_length_chars': self._calculate_avg_message_length(),
            'messages_per_match': self.matches['message_count'].mean(),
            'initiator_rate': self.matches['messages'].apply(
                lambda x: 1 if x and x[0]['sender'] == 'self' else 0).mean()
        }
    
    def _calculate_avg_message_length(self) -> float:
        """Calculate average message length in characters"""
        lengths = []
        for _, match in self.matches.iterrows():
            for msg in match['messages']:
                lengths.append(len(msg['body']))
        return np.mean(lengths) if lengths else 0
    
    def generate_full_report(self) -> Dict:
        """Generate comprehensive metrics report"""
        return {
            'basic_metrics': self.calculate_basic_metrics(),
            'time_metrics': self.calculate_time_based_metrics(),
            'conversation_metrics': self.calculate_conversation_metrics()
        }