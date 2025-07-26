from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import sqlite3
from models.match_predictor import MatchPredictor
from models.player_predictor import PlayerPredictor
from data.data_processor import DataProcessor
import plotly.graph_objects as go
import plotly.express as px
import json

app = Flask(__name__)
CORS(app)

# Initialize predictors
match_predictor = MatchPredictor()
player_predictor = PlayerPredictor()
data_processor = DataProcessor()

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

@app.route('/api/predict-match', methods=['POST'])
def predict_match():
    """Predict match winner"""
    try:
        data = request.json
        
        # Required fields
        required_fields = ['team1', 'team2', 'venue', 'format']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Process input data
        processed_data = data_processor.process_match_data(data)
        
        # Get prediction
        prediction = match_predictor.predict(processed_data)
        
        # Get feature importance for explainability
        feature_importance = match_predictor.get_feature_importance()
        
        response = {
            "predicted_winner": prediction['winner'],
            "confidence_score": prediction['confidence'],
            "probability_team1": prediction['prob_team1'],
            "probability_team2": prediction['prob_team2'],
            "feature_importance": feature_importance,
            "factors": prediction.get('factors', [])
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict-player', methods=['POST'])
def predict_player():
    """Predict player performance"""
    try:
        data = request.json
        
        # Required fields
        required_fields = ['player_name', 'team', 'opposition', 'venue', 'format']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Process input data
        processed_data = data_processor.process_player_data(data)
        
        # Get prediction
        prediction = player_predictor.predict(processed_data)
        
        response = {
            "player_name": data['player_name'],
            "predicted_runs": prediction.get('runs'),
            "predicted_wickets": prediction.get('wickets'),
            "performance_class": prediction.get('performance_class'),
            "confidence_score": prediction.get('confidence'),
            "fantasy_points": prediction.get('fantasy_points'),
            "key_factors": prediction.get('factors', [])
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/teams', methods=['GET'])
def get_teams():
    """Get list of available teams"""
    teams = data_processor.get_teams()
    return jsonify({"teams": teams})

@app.route('/api/venues', methods=['GET'])
def get_venues():
    """Get list of available venues"""
    venues = data_processor.get_venues()
    return jsonify({"venues": venues})

@app.route('/api/players', methods=['GET'])
def get_players():
    """Get list of available players"""
    team = request.args.get('team')
    players = data_processor.get_players(team)
    return jsonify({"players": players})

@app.route('/api/head-to-head', methods=['GET'])
def get_head_to_head():
    """Get head-to-head statistics between two teams"""
    team1 = request.args.get('team1')
    team2 = request.args.get('team2')
    
    if not team1 or not team2:
        return jsonify({"error": "Both team1 and team2 parameters are required"}), 400
    
    h2h_stats = data_processor.get_head_to_head_stats(team1, team2)
    return jsonify(h2h_stats)

@app.route('/api/player-stats', methods=['GET'])
def get_player_stats():
    """Get player statistics"""
    player_name = request.args.get('player_name')
    
    if not player_name:
        return jsonify({"error": "player_name parameter is required"}), 400
    
    stats = data_processor.get_player_stats(player_name)
    return jsonify(stats)

@app.route('/api/analytics/team-performance', methods=['GET'])
def team_performance_analytics():
    """Get team performance analytics with visualizations"""
    team = request.args.get('team')
    format_type = request.args.get('format', 'all')
    
    analytics = data_processor.get_team_analytics(team, format_type)
    
    # Create visualization
    fig = px.line(analytics['recent_form'], x='date', y='result', 
                  title=f'{team} Recent Performance')
    chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return jsonify({
        "analytics": analytics,
        "chart": chart_json
    })

@app.route('/api/train-models', methods=['POST'])
def train_models():
    """Retrain the ML models with latest data"""
    try:
        # This endpoint would be used to retrain models periodically
        match_predictor.train()
        player_predictor.train()
        
        return jsonify({
            "status": "success",
            "message": "Models retrained successfully",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Initialize database and models on startup
    try:
        data_processor.initialize_database()
        match_predictor.load_or_train()
        player_predictor.load_or_train()
        print("✅ Models loaded successfully!")
    except Exception as e:
        print(f"⚠️ Warning: Could not load models - {e}")
        print("Models will be trained on first prediction request.")
    
    app.run(debug=True, host='0.0.0.0', port=5000)