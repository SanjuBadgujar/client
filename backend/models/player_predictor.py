import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class PlayerPredictor:
    def __init__(self):
        # Models for different predictions
        self.runs_model = RandomForestRegressor(random_state=42, n_estimators=100)
        self.wickets_model = RandomForestRegressor(random_state=42, n_estimators=100)
        self.performance_classifier = RandomForestClassifier(random_state=42, n_estimators=100)
        
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        self.model_path = 'backend/models/saved_models/'
        
        # Ensure model directory exists
        os.makedirs(self.model_path, exist_ok=True)
    
    def preprocess_features(self, df, is_training=True):
        """Preprocess features for training or prediction"""
        df = df.copy()
        
        # Categorical features to encode
        categorical_features = ['player_name', 'team', 'opposition', 'venue', 'format', 'position']
        
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
        
        # Remove target variables from features
        target_cols = ['runs', 'wickets', 'performance_class', 'fantasy_points']
        for col in target_cols:
            if col in numeric_features:
                numeric_features.remove(col)
        
        if is_training:
            df[numeric_features] = self.scaler.fit_transform(df[numeric_features])
        else:
            df[numeric_features] = self.scaler.transform(df[numeric_features])
        
        self.feature_names = numeric_features
        return df[numeric_features]
    
    def engineer_features(self, df):
        """Engineer additional features from existing data"""
        # Calculate batting/bowling averages if available
        if 'career_runs' in df.columns and 'career_innings' in df.columns:
            df['batting_average'] = df['career_runs'] / (df['career_innings'] + 1)
        
        if 'career_wickets' in df.columns and 'career_matches' in df.columns:
            df['bowling_average'] = df['career_wickets'] / (df['career_matches'] + 1)
        
        # Recent form indicators
        if 'recent_runs' in df.columns and 'recent_innings' in df.columns:
            df['recent_avg'] = df['recent_runs'] / (df['recent_innings'] + 1)
        
        # Opposition strength factor
        if 'opposition_ranking' in df.columns:
            df['opposition_strength'] = 11 - df['opposition_ranking']  # Convert to strength (higher = stronger)
        
        # Venue familiarity
        if 'matches_at_venue' in df.columns:
            df['venue_familiarity'] = np.log1p(df['matches_at_venue'])
        
        # Format specialization
        if 'format_matches' in df.columns and 'total_matches' in df.columns:
            df['format_specialization'] = df['format_matches'] / (df['total_matches'] + 1)
        
        return df
    
    def create_sample_data(self):
        """Create sample player performance data"""
        np.random.seed(42)
        
        players = [
            'Virat Kohli', 'Rohit Sharma', 'Babar Azam', 'Steve Smith', 'Joe Root',
            'Kane Williamson', 'David Warner', 'Quinton de Kock', 'Ben Stokes', 'Hardik Pandya',
            'Jasprit Bumrah', 'Pat Cummins', 'Trent Boult', 'Kagiso Rabada', 'Shaheen Afridi'
        ]
        
        teams = ['India', 'Australia', 'England', 'Pakistan', 'South Africa', 
                'New Zealand', 'West Indies', 'Sri Lanka', 'Bangladesh', 'Afghanistan']
        
        venues = ['Mumbai', 'Sydney', 'Lord\'s', 'Lahore', 'Cape Town', 
                 'Auckland', 'Kingston', 'Colombo', 'Dhaka', 'Kabul']
        
        formats = ['ODI', 'T20I', 'Test']
        positions = ['Batsman', 'Bowler', 'All-rounder', 'Wicket-keeper']
        
        n_samples = 1500
        data = []
        
        for i in range(n_samples):
            player = np.random.choice(players)
            team = np.random.choice(teams)
            opposition = np.random.choice([t for t in teams if t != team])
            venue = np.random.choice(venues)
            format_type = np.random.choice(formats)
            position = np.random.choice(positions)
            
            # Player attributes (career stats)
            career_matches = np.random.randint(20, 200)
            career_runs = np.random.randint(500, 15000)
            career_innings = np.random.randint(20, 300)
            career_wickets = np.random.randint(0, 300) if position in ['Bowler', 'All-rounder'] else np.random.randint(0, 20)
            
            # Recent form (last 5 matches)
            recent_innings = np.random.randint(1, 6)
            recent_runs = np.random.randint(0, 300)
            recent_wickets = np.random.randint(0, 10) if position in ['Bowler', 'All-rounder'] else np.random.randint(0, 3)
            
            # Opposition strength
            opposition_ranking = np.random.randint(1, 11)
            
            # Venue familiarity
            matches_at_venue = np.random.randint(0, 20)
            
            # Format specialization
            format_matches = np.random.randint(10, career_matches)
            
            # Generate performance based on player attributes
            base_runs = 30 if format_type == 'T20I' else 50 if format_type == 'ODI' else 40
            base_wickets = 1 if position in ['Bowler', 'All-rounder'] else 0
            
            # Adjust based on recent form and opposition
            form_factor = recent_runs / (recent_innings * 20 + 1)
            opposition_factor = (11 - opposition_ranking) / 10
            venue_factor = min(matches_at_venue / 10, 1)
            
            # Predict runs
            runs = base_runs * (1 + form_factor) * (1 + opposition_factor * 0.3) * (1 + venue_factor * 0.1)
            runs = max(0, int(runs + np.random.normal(0, 15)))
            
            # Predict wickets
            if position in ['Bowler', 'All-rounder']:
                wickets = base_wickets * (1 + form_factor) * (1 + opposition_factor * 0.3)
                wickets = max(0, int(wickets + np.random.normal(0, 1)))
            else:
                wickets = 0
            
            # Performance classification
            if runs >= 50 or wickets >= 3:
                performance_class = 'High'
            elif runs >= 25 or wickets >= 1:
                performance_class = 'Medium'
            else:
                performance_class = 'Low'
            
            # Fantasy points calculation
            fantasy_points = runs + (wickets * 25) + (50 if runs >= 50 else 0) + (25 if wickets >= 3 else 0)
            
            data.append({
                'player_name': player,
                'team': team,
                'opposition': opposition,
                'venue': venue,
                'format': format_type,
                'position': position,
                'career_matches': career_matches,
                'career_runs': career_runs,
                'career_innings': career_innings,
                'career_wickets': career_wickets,
                'recent_innings': recent_innings,
                'recent_runs': recent_runs,
                'recent_wickets': recent_wickets,
                'opposition_ranking': opposition_ranking,
                'matches_at_venue': matches_at_venue,
                'format_matches': format_matches,
                'total_matches': career_matches,
                'runs': runs,
                'wickets': wickets,
                'performance_class': performance_class,
                'fantasy_points': fantasy_points
            })
        
        return pd.DataFrame(data)
    
    def train(self, df=None):
        """Train the player performance prediction models"""
        if df is None:
            print("No training data provided. Creating sample data...")
            df = self.create_sample_data()
        
        print(f"Training with {len(df)} player performances...")
        
        # Preprocess features
        X = self.preprocess_features(df, is_training=True)
        
        # Prepare target variables
        y_runs = df['runs']
        y_wickets = df['wickets']
        y_performance = df['performance_class']
        
        # Split data
        X_train, X_test, y_runs_train, y_runs_test = train_test_split(
            X, y_runs, test_size=0.2, random_state=42
        )
        _, _, y_wickets_train, y_wickets_test = train_test_split(
            X, y_wickets, test_size=0.2, random_state=42
        )
        _, _, y_perf_train, y_perf_test = train_test_split(
            X, y_performance, test_size=0.2, random_state=42
        )
        
        print("\nTraining models...")
        
        # Train runs prediction model
        print("📊 Training runs prediction model...")
        self.runs_model.fit(X_train, y_runs_train)
        runs_pred = self.runs_model.predict(X_test)
        runs_mae = mean_absolute_error(y_runs_test, runs_pred)
        runs_rmse = np.sqrt(mean_squared_error(y_runs_test, runs_pred))
        print(f"  Runs MAE: {runs_mae:.2f}")
        print(f"  Runs RMSE: {runs_rmse:.2f}")
        
        # Train wickets prediction model
        print("📊 Training wickets prediction model...")
        self.wickets_model.fit(X_train, y_wickets_train)
        wickets_pred = self.wickets_model.predict(X_test)
        wickets_mae = mean_absolute_error(y_wickets_test, wickets_pred)
        wickets_rmse = np.sqrt(mean_squared_error(y_wickets_test, wickets_pred))
        print(f"  Wickets MAE: {wickets_mae:.2f}")
        print(f"  Wickets RMSE: {wickets_rmse:.2f}")
        
        # Train performance classification model
        print("📊 Training performance classification model...")
        self.performance_classifier.fit(X_train, y_perf_train)
        perf_pred = self.performance_classifier.predict(X_test)
        perf_accuracy = accuracy_score(y_perf_test, perf_pred)
        print(f"  Performance Accuracy: {perf_accuracy:.4f}")
        
        # Save models
        self.save_model()
        
        return {
            'runs_mae': runs_mae,
            'runs_rmse': runs_rmse,
            'wickets_mae': wickets_mae,
            'wickets_rmse': wickets_rmse,
            'performance_accuracy': perf_accuracy
        }
    
    def predict(self, input_data):
        """Make player performance prediction"""
        if self.runs_model is None:
            raise ValueError("No model trained. Please train the model first.")
        
        # Convert input to DataFrame if it's a dict
        if isinstance(input_data, dict):
            input_data = pd.DataFrame([input_data])
        
        # Preprocess
        X = self.preprocess_features(input_data, is_training=False)
        
        # Get predictions
        runs_pred = max(0, int(self.runs_model.predict(X)[0]))
        wickets_pred = max(0, int(self.wickets_model.predict(X)[0]))
        performance_class = self.performance_classifier.predict(X)[0]
        
        # Get prediction probabilities for confidence
        perf_proba = max(self.performance_classifier.predict_proba(X)[0])
        
        # Calculate fantasy points
        fantasy_points = runs_pred + (wickets_pred * 25)
        if runs_pred >= 50:
            fantasy_points += 50
        if wickets_pred >= 3:
            fantasy_points += 25
        
        # Generate explanation factors
        factors = self.get_prediction_factors(input_data.iloc[0], runs_pred, wickets_pred)
        
        return {
            'runs': runs_pred,
            'wickets': wickets_pred,
            'performance_class': performance_class,
            'confidence': float(perf_proba),
            'fantasy_points': fantasy_points,
            'factors': factors
        }
    
    def get_prediction_factors(self, input_data, runs_pred, wickets_pred):
        """Generate explanation factors for the prediction"""
        factors = []
        
        # Analyze recent form
        if 'recent_runs' in input_data and 'recent_innings' in input_data:
            recent_avg = input_data['recent_runs'] / (input_data['recent_innings'] + 1)
            if recent_avg > 30:
                factors.append(f"Good recent form: {recent_avg:.1f} runs per innings in last {input_data['recent_innings']} matches")
            elif recent_avg < 15:
                factors.append(f"Poor recent form: {recent_avg:.1f} runs per innings in last {input_data['recent_innings']} matches")
        
        # Opposition strength
        if 'opposition_ranking' in input_data:
            if input_data['opposition_ranking'] <= 3:
                factors.append(f"Strong opposition (Rank {input_data['opposition_ranking']}) may limit performance")
            elif input_data['opposition_ranking'] >= 7:
                factors.append(f"Weaker opposition (Rank {input_data['opposition_ranking']}) may boost performance")
        
        # Venue familiarity
        if 'matches_at_venue' in input_data:
            if input_data['matches_at_venue'] >= 5:
                factors.append(f"Good venue familiarity: {input_data['matches_at_venue']} matches at {input_data.get('venue', 'venue')}")
            elif input_data['matches_at_venue'] == 0:
                factors.append(f"First time playing at {input_data.get('venue', 'venue')}")
        
        # Position-specific factors
        if 'position' in input_data:
            if input_data['position'] in ['Bowler', 'All-rounder'] and wickets_pred >= 2:
                factors.append(f"As a {input_data['position']}, good wicket-taking opportunity predicted")
        
        return factors
    
    def save_model(self):
        """Save the trained models and preprocessors"""
        joblib.dump(self.runs_model, f"{self.model_path}player_runs_model.pkl")
        joblib.dump(self.wickets_model, f"{self.model_path}player_wickets_model.pkl")
        joblib.dump(self.performance_classifier, f"{self.model_path}player_performance_classifier.pkl")
        joblib.dump(self.label_encoders, f"{self.model_path}player_label_encoders.pkl")
        joblib.dump(self.scaler, f"{self.model_path}player_scaler.pkl")
        joblib.dump(self.feature_names, f"{self.model_path}player_feature_names.pkl")
        print(f"✅ Player models saved to {self.model_path}")
    
    def load_model(self):
        """Load the trained models and preprocessors"""
        try:
            self.runs_model = joblib.load(f"{self.model_path}player_runs_model.pkl")
            self.wickets_model = joblib.load(f"{self.model_path}player_wickets_model.pkl")
            self.performance_classifier = joblib.load(f"{self.model_path}player_performance_classifier.pkl")
            self.label_encoders = joblib.load(f"{self.model_path}player_label_encoders.pkl")
            self.scaler = joblib.load(f"{self.model_path}player_scaler.pkl")
            self.feature_names = joblib.load(f"{self.model_path}player_feature_names.pkl")
            print("✅ Player prediction models loaded successfully!")
            return True
        except FileNotFoundError:
            print("⚠️ No saved player models found. Will need to train first.")
            return False
    
    def load_or_train(self):
        """Load existing models or train new ones"""
        if not self.load_model():
            print("🔄 Training new player prediction models...")
            self.train()
    
    def get_feature_importance(self, model_type='runs'):
        """Get feature importance from the specified model"""
        model = None
        if model_type == 'runs':
            model = self.runs_model
        elif model_type == 'wickets':
            model = self.wickets_model
        elif model_type == 'performance':
            model = self.performance_classifier
        
        if model is not None and hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            feature_importance = dict(zip(self.feature_names, importance))
            return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        
        return {}