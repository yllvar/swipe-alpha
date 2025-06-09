import pandas as pd
import numpy as np
import spacy
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict, List, Tuple

class ProfileNLPAnalyzer:
    def __init__(self):
        """
        Initialize NLP analyzer for profile bios and messages
        """
        self.nlp = spacy.load('en_core_web_sm')
        self.vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        
    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of profile bio or message"""
        blob = TextBlob(text)
        return {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity,
            'sentiment': 'positive' if blob.sentiment.polarity > 0 else 
                        'negative' if blob.sentiment.polarity < 0 else 'neutral'
        }
    
    def extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text using spaCy"""
        doc = self.nlp(text)
        phrases = [chunk.text for chunk in doc.noun_chunks]
        return list(set(phrases))  # Remove duplicates
    
    def calculate_readability(self, text: str) -> float:
        """Calculate Flesch reading ease score"""
        sentences = text.count('.') + text.count('!') + text.count('?')
        words = len(text.split())
        syllables = sum([self.count_syllables(word) for word in text.split()])
        
        if words == 0 or sentences == 0:
            return 0
            
        return 206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)
    
    def count_syllables(self, word: str) -> int:
        """Count syllables in a word (approximate)"""
        word = word.lower()
        count = 0
        vowels = "aeiouy"
        if word[0] in vowels:
            count += 1
        for index in range(1, len(word)):
            if word[index] in vowels and word[index - 1] not in vowels:
                count += 1
        if word.endswith('e'):
            count -= 1
        if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
            count += 1
        if count == 0:
            count += 1
        return count
    
    def vectorize_text(self, texts: List[str]) -> pd.DataFrame:
        """Convert text to TF-IDF vectors"""
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        return pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=self.vectorizer.get_feature_names_out()
        )
    
    def analyze_profiles(self, profiles: pd.DataFrame) -> pd.DataFrame:
        """Perform comprehensive NLP analysis on profiles"""
        results = []
        for _, row in profiles.iterrows():
            bio = row.get('bio', '')
            analysis = {
                'profile_id': row.get('id'),
                'bio_length': len(bio),
                **self.analyze_sentiment(bio),
                'key_phrases': self.extract_key_phrases(bio),
                'readability': self.calculate_readability(bio),
                'word_count': len(bio.split())
            }
            results.append(analysis)
        
        return pd.DataFrame(results)