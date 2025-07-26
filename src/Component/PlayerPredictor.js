import React, { useState, useEffect } from 'react';
import './PlayerPredictor.css';

const PlayerPredictor = ({ teams, venues, players }) => {
  const [formData, setFormData] = useState({
    player_name: '',
    team: '',
    opposition: '',
    venue: '',
    format: 'ODI'
  });
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [playerStats, setPlayerStats] = useState(null);
  const [filteredPlayers, setFilteredPlayers] = useState(players);

  useEffect(() => {
    // Filter players by team when team is selected
    if (formData.team) {
      fetchPlayersByTeam(formData.team);
    } else {
      setFilteredPlayers(players);
    }
  }, [formData.team, players]);

  const fetchPlayersByTeam = async (team) => {
    try {
      const response = await fetch(`http://localhost:5000/api/players?team=${team}`);
      if (response.ok) {
        const data = await response.json();
        setFilteredPlayers(data.players);
      }
    } catch (error) {
      console.error('Error fetching players by team:', error);
      // Fallback to all players
      setFilteredPlayers(players);
    }
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));

    // Fetch player stats when player is selected
    if (name === 'player_name' && value) {
      fetchPlayerStats(value);
    }
  };

  const fetchPlayerStats = async (playerName) => {
    try {
      const response = await fetch(`http://localhost:5000/api/player-stats?player_name=${playerName}`);
      if (response.ok) {
        const data = await response.json();
        setPlayerStats(data);
      }
    } catch (error) {
      console.error('Error fetching player stats:', error);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!formData.player_name || !formData.team || !formData.opposition || !formData.venue) {
      alert('Please fill in all required fields');
      return;
    }

    if (formData.team === formData.opposition) {
      alert('Please select different teams');
      return;
    }

    setLoading(true);
    
    try {
      const response = await fetch('http://localhost:5000/api/predict-player', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });

      if (response.ok) {
        const data = await response.json();
        setPrediction(data);
      } else {
        const errorData = await response.json();
        alert(`Error: ${errorData.error}`);
      }
    } catch (error) {
      console.error('Error predicting player performance:', error);
      alert('Error connecting to prediction service');
    } finally {
      setLoading(false);
    }
  };

  const resetForm = () => {
    setFormData({
      player_name: '',
      team: '',
      opposition: '',
      venue: '',
      format: 'ODI'
    });
    setPrediction(null);
    setPlayerStats(null);
  };

  const getPerformanceColor = (performanceClass) => {
    switch (performanceClass) {
      case 'High': return '#4CAF50';
      case 'Medium': return '#FF9800';
      case 'Low': return '#f44336';
      default: return '#9E9E9E';
    }
  };

  return (
    <div className="player-predictor">
      <div className="predictor-container">
        <div className="form-section">
          <h2>👤 Player Performance Predictor</h2>
          <form onSubmit={handleSubmit} className="prediction-form">
            <div className="form-row">
              <div className="form-group">
                <label htmlFor="team">Team *</label>
                <select
                  id="team"
                  name="team"
                  value={formData.team}
                  onChange={handleInputChange}
                  required
                >
                  <option value="">Select Team</option>
                  {teams.map(team => (
                    <option key={team} value={team}>{team}</option>
                  ))}
                </select>
              </div>

              <div className="form-group">
                <label htmlFor="player_name">Player *</label>
                <select
                  id="player_name"
                  name="player_name"
                  value={formData.player_name}
                  onChange={handleInputChange}
                  required
                >
                  <option value="">Select Player</option>
                  {filteredPlayers.map(player => (
                    <option key={player} value={player}>{player}</option>
                  ))}
                </select>
              </div>
            </div>

            <div className="form-row">
              <div className="form-group">
                <label htmlFor="opposition">Opposition *</label>
                <select
                  id="opposition"
                  name="opposition"
                  value={formData.opposition}
                  onChange={handleInputChange}
                  required
                >
                  <option value="">Select Opposition</option>
                  {teams.filter(team => team !== formData.team).map(team => (
                    <option key={team} value={team}>{team}</option>
                  ))}
                </select>
              </div>

              <div className="form-group">
                <label htmlFor="venue">Venue *</label>
                <select
                  id="venue"
                  name="venue"
                  value={formData.venue}
                  onChange={handleInputChange}
                  required
                >
                  <option value="">Select Venue</option>
                  {venues.map(venue => (
                    <option key={venue} value={venue}>{venue}</option>
                  ))}
                </select>
              </div>
            </div>

            <div className="form-row">
              <div className="form-group">
                <label htmlFor="format">Format *</label>
                <select
                  id="format"
                  name="format"
                  value={formData.format}
                  onChange={handleInputChange}
                  required
                >
                  <option value="ODI">ODI</option>
                  <option value="T20I">T20I</option>
                  <option value="Test">Test</option>
                </select>
              </div>
            </div>

            <div className="form-actions">
              <button type="submit" disabled={loading} className="predict-btn">
                {loading ? '🔄 Predicting...' : '🎯 Predict Performance'}
              </button>
              <button type="button" onClick={resetForm} className="reset-btn">
                🔄 Reset
              </button>
            </div>
          </form>
        </div>

        {playerStats && (
          <div className="player-stats-section">
            <h3>Player Statistics</h3>
            <div className="stats-grid">
              <div className="stat-card">
                <span className="stat-title">Career Average</span>
                <span className="stat-value">{playerStats.batting_average?.toFixed(2) || 'N/A'}</span>
              </div>
              <div className="stat-card">
                <span className="stat-title">Recent Form</span>
                <span className="stat-value">{playerStats.recent_form?.recent_average?.toFixed(2) || 'N/A'}</span>
              </div>
              <div className="stat-card">
                <span className="stat-title">Total Runs</span>
                <span className="stat-value">{playerStats.career_stats?.total_runs || 'N/A'}</span>
              </div>
              <div className="stat-card">
                <span className="stat-title">Total Wickets</span>
                <span className="stat-value">{playerStats.career_stats?.total_wickets || 'N/A'}</span>
              </div>
            </div>
          </div>
        )}

        {prediction && (
          <div className="prediction-result">
            <h3>🎯 Performance Prediction</h3>
            
            <div className="prediction-header">
              <div className="player-info">
                <span className="player-name">{prediction.player_name}</span>
                <span 
                  className="performance-class"
                  style={{ backgroundColor: getPerformanceColor(prediction.performance_class) }}
                >
                  {prediction.performance_class} Performance
                </span>
              </div>
              <div className="confidence-score">
                <span className="confidence-label">Confidence:</span>
                <span className="confidence-value">{(prediction.confidence_score * 100).toFixed(1)}%</span>
              </div>
            </div>

            <div className="prediction-metrics">
              <div className="metric-card runs">
                <div className="metric-header">
                  <span className="metric-icon">🏏</span>
                  <span className="metric-title">Predicted Runs</span>
                </div>
                <div className="metric-value">{prediction.predicted_runs}</div>
              </div>

              <div className="metric-card wickets">
                <div className="metric-header">
                  <span className="metric-icon">🎯</span>
                  <span className="metric-title">Predicted Wickets</span>
                </div>
                <div className="metric-value">{prediction.predicted_wickets}</div>
              </div>

              <div className="metric-card fantasy">
                <div className="metric-header">
                  <span className="metric-icon">⭐</span>
                  <span className="metric-title">Fantasy Points</span>
                </div>
                <div className="metric-value">{prediction.fantasy_points}</div>
              </div>
            </div>

            {prediction.key_factors && prediction.key_factors.length > 0 && (
              <div className="prediction-factors">
                <h4>Key Factors:</h4>
                <ul>
                  {prediction.key_factors.map((factor, index) => (
                    <li key={index}>{factor}</li>
                  ))}
                </ul>
              </div>
            )}

            <div className="fantasy-breakdown">
              <h4>Fantasy Points Breakdown:</h4>
              <div className="breakdown-items">
                <div className="breakdown-item">
                  <span>Runs: {prediction.predicted_runs} × 1</span>
                  <span>{prediction.predicted_runs} pts</span>
                </div>
                <div className="breakdown-item">
                  <span>Wickets: {prediction.predicted_wickets} × 25</span>
                  <span>{prediction.predicted_wickets * 25} pts</span>
                </div>
                {prediction.predicted_runs >= 50 && (
                  <div className="breakdown-item bonus">
                    <span>50+ Runs Bonus</span>
                    <span>50 pts</span>
                  </div>
                )}
                {prediction.predicted_wickets >= 3 && (
                  <div className="breakdown-item bonus">
                    <span>3+ Wickets Bonus</span>
                    <span>25 pts</span>
                  </div>
                )}
                <div className="breakdown-total">
                  <span>Total Fantasy Points</span>
                  <span>{prediction.fantasy_points} pts</span>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default PlayerPredictor;