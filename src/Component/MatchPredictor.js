import React, { useState } from 'react';
import './MatchPredictor.css';

const MatchPredictor = ({ teams, venues }) => {
  const [formData, setFormData] = useState({
    team1: '',
    team2: '',
    venue: '',
    format: 'ODI',
    toss_winner: '',
    toss_decision: 'bat'
  });
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [h2hStats, setH2hStats] = useState(null);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));

    // Fetch head-to-head stats when both teams are selected
    if ((name === 'team1' || name === 'team2') && formData.team1 && formData.team2) {
      fetchH2HStats();
    }
  };

  const fetchH2HStats = async () => {
    if (!formData.team1 || !formData.team2) return;
    
    try {
      const response = await fetch(`http://localhost:5000/api/head-to-head?team1=${formData.team1}&team2=${formData.team2}`);
      if (response.ok) {
        const data = await response.json();
        setH2hStats(data);
      }
    } catch (error) {
      console.error('Error fetching H2H stats:', error);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!formData.team1 || !formData.team2 || !formData.venue) {
      alert('Please fill in all required fields');
      return;
    }

    if (formData.team1 === formData.team2) {
      alert('Please select different teams');
      return;
    }

    setLoading(true);
    
    try {
      const response = await fetch('http://localhost:5000/api/predict-match', {
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
      console.error('Error predicting match:', error);
      alert('Error connecting to prediction service');
    } finally {
      setLoading(false);
    }
  };

  const resetForm = () => {
    setFormData({
      team1: '',
      team2: '',
      venue: '',
      format: 'ODI',
      toss_winner: '',
      toss_decision: 'bat'
    });
    setPrediction(null);
    setH2hStats(null);
  };

  return (
    <div className="match-predictor">
      <div className="predictor-container">
        <div className="form-section">
          <h2>🏆 Match Outcome Predictor</h2>
          <form onSubmit={handleSubmit} className="prediction-form">
            <div className="form-row">
              <div className="form-group">
                <label htmlFor="team1">Team 1 *</label>
                <select
                  id="team1"
                  name="team1"
                  value={formData.team1}
                  onChange={handleInputChange}
                  required
                >
                  <option value="">Select Team 1</option>
                  {teams.map(team => (
                    <option key={team} value={team}>{team}</option>
                  ))}
                </select>
              </div>

              <div className="vs-indicator">
                <span>VS</span>
              </div>

              <div className="form-group">
                <label htmlFor="team2">Team 2 *</label>
                <select
                  id="team2"
                  name="team2"
                  value={formData.team2}
                  onChange={handleInputChange}
                  required
                >
                  <option value="">Select Team 2</option>
                  {teams.filter(team => team !== formData.team1).map(team => (
                    <option key={team} value={team}>{team}</option>
                  ))}
                </select>
              </div>
            </div>

            <div className="form-row">
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

            <div className="form-row">
              <div className="form-group">
                <label htmlFor="toss_winner">Toss Winner</label>
                <select
                  id="toss_winner"
                  name="toss_winner"
                  value={formData.toss_winner}
                  onChange={handleInputChange}
                >
                  <option value="">Not decided</option>
                  {formData.team1 && <option value={formData.team1}>{formData.team1}</option>}
                  {formData.team2 && <option value={formData.team2}>{formData.team2}</option>}
                </select>
              </div>

              <div className="form-group">
                <label htmlFor="toss_decision">Toss Decision</label>
                <select
                  id="toss_decision"
                  name="toss_decision"
                  value={formData.toss_decision}
                  onChange={handleInputChange}
                >
                  <option value="bat">Bat First</option>
                  <option value="bowl">Bowl First</option>
                </select>
              </div>
            </div>

            <div className="form-actions">
              <button type="submit" disabled={loading} className="predict-btn">
                {loading ? '🔄 Predicting...' : '🎯 Predict Winner'}
              </button>
              <button type="button" onClick={resetForm} className="reset-btn">
                🔄 Reset
              </button>
            </div>
          </form>
        </div>

        {h2hStats && (
          <div className="h2h-section">
            <h3>Head-to-Head Statistics</h3>
            <div className="h2h-stats">
              <div className="stat-item">
                <span className="stat-label">Total Matches:</span>
                <span className="stat-value">{h2hStats.total_matches}</span>
              </div>
              <div className="stat-item">
                <span className="stat-label">{formData.team1} Wins:</span>
                <span className="stat-value">{h2hStats.team1_wins}</span>
              </div>
              <div className="stat-item">
                <span className="stat-label">{formData.team2} Wins:</span>
                <span className="stat-value">{h2hStats.team2_wins}</span>
              </div>
              <div className="stat-item">
                <span className="stat-label">Win Percentage:</span>
                <span className="stat-value">{h2hStats.team1_win_percentage?.toFixed(1)}%</span>
              </div>
            </div>
          </div>
        )}

        {prediction && (
          <div className="prediction-result">
            <h3>🎯 Prediction Result</h3>
            <div className="winner-announcement">
              <div className="winner-team">
                <span className="winner-label">Predicted Winner:</span>
                <span className="winner-name">{prediction.predicted_winner}</span>
              </div>
              <div className="confidence-score">
                <span className="confidence-label">Confidence:</span>
                <span className="confidence-value">{(prediction.confidence_score * 100).toFixed(1)}%</span>
                <div className="confidence-bar">
                  <div 
                    className="confidence-fill" 
                    style={{ width: `${prediction.confidence_score * 100}%` }}
                  ></div>
                </div>
              </div>
            </div>

            <div className="team-probabilities">
              <div className="team-prob">
                <span className="team-name">{formData.team1}</span>
                <div className="prob-bar">
                  <div 
                    className="prob-fill team1" 
                    style={{ width: `${prediction.probability_team1 * 100}%` }}
                  ></div>
                </div>
                <span className="prob-value">{(prediction.probability_team1 * 100).toFixed(1)}%</span>
              </div>
              <div className="team-prob">
                <span className="team-name">{formData.team2}</span>
                <div className="prob-bar">
                  <div 
                    className="prob-fill team2" 
                    style={{ width: `${prediction.probability_team2 * 100}%` }}
                  ></div>
                </div>
                <span className="prob-value">{(prediction.probability_team2 * 100).toFixed(1)}%</span>
              </div>
            </div>

            {prediction.factors && prediction.factors.length > 0 && (
              <div className="prediction-factors">
                <h4>Key Factors:</h4>
                <ul>
                  {prediction.factors.map((factor, index) => (
                    <li key={index}>{factor}</li>
                  ))}
                </ul>
              </div>
            )}

            {prediction.feature_importance && Object.keys(prediction.feature_importance).length > 0 && (
              <div className="feature-importance">
                <h4>Feature Importance:</h4>
                <div className="importance-list">
                  {Object.entries(prediction.feature_importance)
                    .slice(0, 5)
                    .map(([feature, importance]) => (
                      <div key={feature} className="importance-item">
                        <span className="feature-name">{feature.replace(/_/g, ' ')}</span>
                        <div className="importance-bar">
                          <div 
                            className="importance-fill" 
                            style={{ width: `${importance * 100}%` }}
                          ></div>
                        </div>
                        <span className="importance-value">{(importance * 100).toFixed(1)}%</span>
                      </div>
                    ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default MatchPredictor;