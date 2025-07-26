import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os
import shap
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class MatchPredictor:
    def __init__(self):
        self.models = {
            'xgboost': XGBClassifier(random_state=42, eval_metric='logloss'),
            'random_forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        self.best_model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        self.model_path = 'backend/models/saved_models/'
        
        # Ensure model directory exists
        os.makedirs(self.model_path, exist_ok=True)
    
    def preprocess_features(self, df, is_training=True):
        """Preprocess features for training or prediction"""
        # Create copy to avoid modifying original
        df = df.copy()
        
        # Categorical features to encode
        categorical_features = ['team1', 'team2', 'venue', 'format', 'toss_winner', 'toss_decision']
        
        # Encode categorical features
        for feature in categorical_features:
            if feature in df.columns:
                if is_training:
                    if feature not in self.label_encoders:
                        self.label_encoders[feature] = LabelEncoder()
                    df[feature] = self.label_encoders[feature].fit_transform(df[feature].astype(str))
                else:
                    if feature in self.label_encoders:
                        # Handle unseen categories
                        df[feature] = df[feature].astype(str)
                        unique_vals = set(self.label_encoders[feature].classes_)
                        df[feature] = df[feature].apply(lambda x: x if x in unique_vals else 'unknown')
                        
                        # Add 'unknown' to encoder if not present
                        if 'unknown' not in self.label_encoders[feature].classes_:
                            self.label_encoders[feature].classes_ = np.append(
                                self.label_encoders[feature].classes_, 'unknown')
                        
                        df[feature] = self.label_encoders[feature].transform(df[feature])
        
        # Feature engineering
        df = self.engineer_features(df)
        
        # Select numeric features for scaling
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'result' in numeric_features:
            numeric_features.remove('result')
        
        if is_training:
            df[numeric_features] = self.scaler.fit_transform(df[numeric_features])
        else:
            df[numeric_features] = self.scaler.transform(df[numeric_features])
        
        self.feature_names = numeric_features
        return df[numeric_features]
    
    def engineer_features(self, df):
        """Engineer additional features from existing data"""
        # Team strength differential (if rankings available)
        if 'team1_ranking' in df.columns and 'team2_ranking' in df.columns:
            df['ranking_diff'] = df['team1_ranking'] - df['team2_ranking']
        
        # Recent form differential
        if 'team1_recent_wins' in df.columns and 'team2_recent_wins' in df.columns:
            df['recent_form_diff'] = df['team1_recent_wins'] - df['team2_recent_wins']
        
        # Head-to-head advantage
        if 'h2h_team1_wins' in df.columns and 'h2h_total_matches' in df.columns:
            df['h2h_win_rate'] = df['h2h_team1_wins'] / (df['h2h_total_matches'] + 1)  # +1 to avoid division by zero
        
        # Home advantage
        if 'team1_home' in df.columns:
            df['home_advantage'] = df['team1_home'].astype(int)
        
        # Venue familiarity (matches played at venue)
        if 'team1_venue_matches' in df.columns and 'team2_venue_matches' in df.columns:
            df['venue_experience_diff'] = df['team1_venue_matches'] - df['team2_venue_matches']
        
        return df
    
    def create_sample_data(self):
        """Create sample training data for demonstration"""
        np.random.seed(42)
        
        teams = ['India', 'Australia', 'England', 'Pakistan', 'South Africa', 
                'New Zealand', 'West Indies', 'Sri Lanka', 'Bangladesh', 'Afghanistan']
        venues = ['Mumbai', 'Sydney', 'Lord\'s', 'Lahore', 'Cape Town', 
                 'Auckland', 'Kingston', 'Colombo', 'Dhaka', 'Kabul']
        formats = ['ODI', 'T20I', 'Test']
        
        n_samples = 1000
        data = []
        
        for i in range(n_samples):
            team1, team2 = np.random.choice(teams, 2, replace=False)
            venue = np.random.choice(venues)
            format_type = np.random.choice(formats)
            
            # Simulate team rankings (1-10, lower is better)
            team1_ranking = np.random.randint(1, 11)
            team2_ranking = np.random.randint(1, 11)
            
            # Simulate recent form (wins in last 5 matches)
            team1_recent_wins = np.random.randint(0, 6)
            team2_recent_wins = np.random.randint(0, 6)
            
            # Simulate head-to-head
            h2h_total = np.random.randint(5, 20)
            h2h_team1_wins = np.random.randint(0, h2h_total + 1)
            
            # Simulate toss
            toss_winner = np.random.choice([team1, team2])
            toss_decision = np.random.choice(['bat', 'bowl'])
            
            # Simulate home advantage
            team1_home = 1 if venue in ['Mumbai'] and team1 == 'India' else 0
            
            # Simulate venue experience
            team1_venue_matches = np.random.randint(0, 10)
            team2_venue_matches = np.random.randint(0, 10)
            
            # Create result based on features (with some randomness)
            # Better ranking, recent form, and home advantage increase win probability
            team1_score = (11 - team1_ranking) + team1_recent_wins + team1_home * 2
            team2_score = (11 - team2_ranking) + team2_recent_wins
            
            # Add some randomness
            team1_score += np.random.normal(0, 2)
            team2_score += np.random.normal(0, 2)
            
            result = 1 if team1_score > team2_score else 0
            
            data.append({
                'team1': team1,
                'team2': team2,
                'venue': venue,
                'format': format_type,
                'team1_ranking': team1_ranking,
                'team2_ranking': team2_ranking,
                'team1_recent_wins': team1_recent_wins,
                'team2_recent_wins': team2_recent_wins,
                'h2h_total_matches': h2h_total,
                'h2h_team1_wins': h2h_team1_wins,
                'toss_winner': toss_winner,
                'toss_decision': toss_decision,
                'team1_home': team1_home,
                'team1_venue_matches': team1_venue_matches,
                'team2_venue_matches': team2_venue_matches,
                'result': result  # 1 if team1 wins, 0 if team2 wins
            })
        
        return pd.DataFrame(data)
    
    def train(self, df=None):
        """Train the match prediction models"""
        if df is None:
            print("No training data provided. Creating sample data...")
            df = self.create_sample_data()
        
        print(f"Training with {len(df)} matches...")
        
        # Preprocess features
        X = self.preprocess_features(df, is_training=True)
        y = df['result']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        best_score = 0
        best_model_name = None
        
        print("\nTraining models...")
        
        # Train and evaluate each model
        for name, model in self.models.items():
            print(f"\n📊 Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            
            # Test set evaluation
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            print(f"  CV Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})")
            print(f"  Test Accuracy: {accuracy:.4f}")
            print(f"  Test Precision: {precision:.4f}")
            print(f"  Test Recall: {recall:.4f}")
            print(f"  Test F1: {f1:.4f}")
            
            # Save if best model
            if cv_scores.mean() > best_score:
                best_score = cv_scores.mean()
                best_model_name = name
                self.best_model = model
        
        print(f"\n🏆 Best model: {best_model_name} (CV Accuracy: {best_score:.4f})")
        
        # Save the best model and preprocessors
        self.save_model()
        
        return {
            'best_model': best_model_name,
            'best_score': best_score,
            'feature_names': self.feature_names
        }
    
    def predict(self, input_data):
        """Make match prediction"""
        if self.best_model is None:
            raise ValueError("No model trained. Please train the model first.")
        
        # Convert input to DataFrame if it's a dict
        if isinstance(input_data, dict):
            input_data = pd.DataFrame([input_data])
        
        # Preprocess
        X = self.preprocess_features(input_data, is_training=False)
        
        # Get prediction and probabilities
        prediction = self.best_model.predict(X)[0]
        probabilities = self.best_model.predict_proba(X)[0]
        
        # Get team names from original input
        team1 = input_data.iloc[0]['team1'] if 'team1' in input_data.columns else "Team 1"
        team2 = input_data.iloc[0]['team2'] if 'team2' in input_data.columns else "Team 2"
        
        winner = team1 if prediction == 1 else team2
        confidence = max(probabilities)
        
        # Generate explanation factors
        factors = self.get_prediction_factors(input_data.iloc[0], probabilities)
        
        return {
            'winner': winner,
            'confidence': float(confidence),
            'prob_team1': float(probabilities[1]),
            'prob_team2': float(probabilities[0]),
            'factors': factors
        }
    
    def get_prediction_factors(self, input_data, probabilities):
        """Generate explanation factors for the prediction"""
        factors = []
        
        # Analyze key factors
        if 'team1_ranking' in input_data and 'team2_ranking' in input_data:
            if input_data['team1_ranking'] < input_data['team2_ranking']:
                factors.append(f"{input_data['team1']} has better ranking ({input_data['team1_ranking']} vs {input_data['team2_ranking']})")
            else:
                factors.append(f"{input_data['team2']} has better ranking ({input_data['team2_ranking']} vs {input_data['team1_ranking']})")
        
        if 'team1_recent_wins' in input_data and 'team2_recent_wins' in input_data:
            if input_data['team1_recent_wins'] > input_data['team2_recent_wins']:
                factors.append(f"{input_data['team1']} has better recent form ({input_data['team1_recent_wins']}/5 vs {input_data['team2_recent_wins']}/5)")
            elif input_data['team2_recent_wins'] > input_data['team1_recent_wins']:
                factors.append(f"{input_data['team2']} has better recent form ({input_data['team2_recent_wins']}/5 vs {input_data['team1_recent_wins']}/5)")
        
        if 'team1_home' in input_data and input_data['team1_home'] == 1:
            factors.append(f"{input_data['team1']} has home advantage")
        
        return factors
    
    def get_feature_importance(self):
        """Get feature importance from the best model"""
        if self.best_model is None:
            return {}
        
        if hasattr(self.best_model, 'feature_importances_'):
            importance = self.best_model.feature_importances_
            feature_importance = dict(zip(self.feature_names, importance))
            # Sort by importance
            return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        
        return {}
    
    def save_model(self):
        """Save the trained model and preprocessors"""
        if self.best_model is not None:
            joblib.dump(self.best_model, f"{self.model_path}match_predictor_model.pkl")
            joblib.dump(self.label_encoders, f"{self.model_path}match_label_encoders.pkl")
            joblib.dump(self.scaler, f"{self.model_path}match_scaler.pkl")
            joblib.dump(self.feature_names, f"{self.model_path}match_feature_names.pkl")
            print(f"✅ Model saved to {self.model_path}")
    
    def load_model(self):
        """Load the trained model and preprocessors"""
        try:
            self.best_model = joblib.load(f"{self.model_path}match_predictor_model.pkl")
            self.label_encoders = joblib.load(f"{self.model_path}match_label_encoders.pkl")
            self.scaler = joblib.load(f"{self.model_path}match_scaler.pkl")
            self.feature_names = joblib.load(f"{self.model_path}match_feature_names.pkl")
            print("✅ Match prediction model loaded successfully!")
            return True
        except FileNotFoundError:
            print("⚠️ No saved model found. Will need to train first.")
            return False
    
    def load_or_train(self):
        """Load existing model or train a new one"""
        if not self.load_model():
            print("🔄 Training new match prediction model...")
            self.train()
        
    def hyperparameter_tuning(self, df=None):
        """Perform hyperparameter tuning for the models"""
        if df is None:
            df = self.create_sample_data()
        
        X = self.preprocess_features(df, is_training=True)
        y = df['result']
        
        # XGBoost hyperparameter tuning
        xgb_params = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.1, 0.2],
            'subsample': [0.8, 1.0]
        }
        
        print("🔍 Performing hyperparameter tuning for XGBoost...")
        xgb_grid = GridSearchCV(
            XGBClassifier(random_state=42, eval_metric='logloss'),
            xgb_params, cv=3, scoring='accuracy', n_jobs=-1
        )
        xgb_grid.fit(X, y)
        
        # Update the best XGBoost model
        self.models['xgboost'] = xgb_grid.best_estimator_
        
        print(f"✅ Best XGBoost parameters: {xgb_grid.best_params_}")
        print(f"✅ Best XGBoost score: {xgb_grid.best_score_:.4f}")
        
        return xgb_grid.best_estimator_