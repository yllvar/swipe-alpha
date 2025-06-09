import pynder
from datetime import datetime
import time
import json
import os
from typing import Dict, List, Optional

class TinderClient:
    def __init__(self, fb_token: str, fb_id: str):
        """
        Initialize Tinder API client using Facebook credentials
        
        Args:
            fb_token: Facebook access token
            fb_id: Facebook user ID
        """
        self.session = pynder.Session(facebook_token=fb_token, facebook_id=fb_id)
        self.user = self.session.profile
        
    def get_nearby_users(self, limit: int = 100) -> List[Dict]:
        """Fetch nearby users with basic profile info"""
        users = []
        for user in self.session.nearby_users(limit=limit):
            users.append({
                'id': user.id,
                'name': user.name,
                'age': user.age,
                'bio': user.bio,
                'distance_km': user.distance_km,
                'photos': [photo.url for photo in user.photos],
                'jobs': [job.get('title', {}).get('name', '') for job in user.jobs],
                'schools': [school.name for school in user.schools],
                'common_connections': user.common_connections,
                'common_interests': user.common_interests,
                'common_friends': user.common_friends,
                'instagram': user.instagram_username,
                'spotify': user.spotify_connected,
                'last_online': user.ping_time.isoformat() if user.ping_time else None
            })
        return users
    
    def get_match_history(self, limit: int = 100) -> List[Dict]:
        """Get historical match data"""
        matches = []
        for match in self.session.matches(limit=limit):
            messages = []
            for message in match.messages:
                messages.append({
                    'sender': message.sender,
                    'body': message.body,
                    'sent_at': message.sent_at.isoformat()
                })
            
            matches.append({
                'match_id': match.id,
                'person_id': match.user.id,
                'name': match.user.name,
                'messages': messages,
                'created_at': match.created_at.isoformat(),
                'is_super_like': match.is_super_like,
                'is_boost_match': match.is_boost_match,
                'is_fast_match': match.is_fast_match,
                'message_count': len(messages),
                'last_message_time': messages[-1]['sent_at'] if messages else None
            })
        return matches
    
    def swipe_right(self, user_id: str) -> bool:
        """Like a profile"""
        try:
            user = self.session.user_info(user_id)
            user.like()
            return True
        except Exception as e:
            print(f"Error swiping right on {user_id}: {e}")
            return False
    
    def send_message(self, match_id: str, message: str) -> bool:
        """Send message to a match"""
        try:
            match = next(m for m in self.session.matches() if m.id == match_id)
            match.message(message)
            return True
        except Exception as e:
            print(f"Error sending message to {match_id}: {e}")
            return False