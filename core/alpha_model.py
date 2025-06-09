import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import Dict, List, Tuple

class AlphaSignalModel:
    def __init__(self, data: pd.DataFrame):
        """
        Initialize with match data to detect alpha signals
        
        Args:
            data: DataFrame containing match outcomes and features
        """
        self.data = data
        self.model = None
        self.feature_importances_ = None
        
    def preprocess_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for modeling"""
        # Define features and target
        text_features = ['bio', 'message_text']
        numeric_features = ['age', 'distance_km', 'common_connections']
        categorical_features = ['has_job', 'has_education', 'spotify_connected']
        
        # Create target variable (1 if led to date, 0 otherwise)
        self.data['target'] = (self.data['date_rate'] > 0.5).astype(int)
        
        # Split data
        X = self.data[text_features + numeric_features + categorical_features]
        y = self.data['target']
        
        return X, y
    
    def build_model(self) -> Pipeline:
        """Build machine learning pipeline to detect alpha signals"""
        # Preprocess different feature types
        text_transformer = TfidfVectorizer(max_features=100, stop_words='english')
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('text', text_transformer, ['bio', 'message_text']),
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        # Create pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        self.model = pipeline
        return pipeline
    
    def train_model(self, test_size: float = 0.2) -> Dict:
        """Train the alpha signal detection model"""
        X, y = self.preprocess_data()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42)
        
        model = self.build_model()
        model.fit(X_train, y_train)
        
        # Get feature importances
        if hasattr(model.named_steps['classifier'], 'feature_importances_'):
            importances = model.named_steps['classifier'].feature_importances_
            self.feature_importances_ = importances
        
        # Evaluate model
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        return {
            'model': model,
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'feature_importances': self.feature_importances_
        }
    
    def predict_alpha(self, new_profiles: pd.DataFrame) -> pd.DataFrame:
        """Predict alpha scores for new profiles"""
        if not self.model:
            raise ValueError("Model not trained. Call train_model() first.")
        
        predictions = self.model.predict_proba(new_profiles)
        new_profiles['alpha_score'] = predictions[:, 1]  # Probability of positive class
        return new_profiles