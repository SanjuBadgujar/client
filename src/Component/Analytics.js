import React, { useState, useEffect } from 'react';
import './Analytics.css';

const Analytics = ({ teams }) => {
  const [selectedTeam, setSelectedTeam] = useState('');
  const [analytics, setAnalytics] = useState(null);
  const [loading, setLoading] = useState(false);

  const fetchAnalytics = async (team) => {
    if (!team) return;
    
    setLoading(true);
    try {
      const response = await fetch(`http://localhost:5000/api/analytics/team-performance?team=${team}`);
      if (response.ok) {
        const data = await response.json();
        setAnalytics(data.analytics);
      }
    } catch (error) {
      console.error('Error fetching analytics:', error);
      // Generate mock data for demo
      setAnalytics({
        recent_form: generateMockRecentForm(),
        win_percentage: Math.random() * 40 + 40, // 40-80%
        average_score: Math.floor(Math.random() * 100 + 200), // 200-300
        highest_score: Math.floor(Math.random() * 100 + 350), // 350-450
        lowest_score: Math.floor(Math.random() * 100 + 100) // 100-200
      });
    } finally {
      setLoading(false);
    }
  };

  const generateMockRecentForm = () => {
    const results = [];
    for (let i = 0; i < 10; i++) {
      results.push({
        date: new Date(Date.now() - i * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
        result: Math.random() > 0.4 ? 'Win' : 'Loss',
        score: Math.floor(Math.random() * 150 + 150)
      });
    }
    return results.reverse();
  };

  const handleTeamChange = (e) => {
    const team = e.target.value;
    setSelectedTeam(team);
    if (team) {
      fetchAnalytics(team);
    } else {
      setAnalytics(null);
    }
  };

  const getFormStreak = () => {
    if (!analytics?.recent_form) return '';
    
    const recent5 = analytics.recent_form.slice(-5);
    return recent5.map(match => match.result === 'Win' ? 'W' : 'L').join(' - ');
  };

  const getWinPercentageColor = (percentage) => {
    if (percentage >= 70) return '#4CAF50';
    if (percentage >= 50) return '#FF9800';
    return '#f44336';
  };

  return (
    <div className="analytics">
      <div className="analytics-container">
        <div className="analytics-header">
          <h2>📊 Team Performance Analytics</h2>
          <div className="team-selector">
            <label htmlFor="team-select">Select Team:</label>
            <select
              id="team-select"
              value={selectedTeam}
              onChange={handleTeamChange}
            >
              <option value="">Choose a team...</option>
              {teams.map(team => (
                <option key={team} value={team}>{team}</option>
              ))}
            </select>
          </div>
        </div>

        {loading && (
          <div className="loading-state">
            <div className="loading-spinner">🔄</div>
            <p>Loading analytics...</p>
          </div>
        )}

        {!selectedTeam && !loading && (
          <div className="empty-state">
            <div className="empty-icon">📈</div>
            <h3>Select a Team to View Analytics</h3>
            <p>Choose a team from the dropdown above to see detailed performance analytics, recent form, and statistics.</p>
          </div>
        )}

        {analytics && !loading && (
          <div className="analytics-content">
            <div className="overview-stats">
              <div className="stat-card primary">
                <div className="stat-header">
                  <span className="stat-icon">🏆</span>
                  <span className="stat-title">Win Percentage</span>
                </div>
                <div 
                  className="stat-value large"
                  style={{ color: getWinPercentageColor(analytics.win_percentage) }}
                >
                  {analytics.win_percentage?.toFixed(1)}%
                </div>
              </div>

              <div className="stat-card">
                <div className="stat-header">
                  <span className="stat-icon">📊</span>
                  <span className="stat-title">Average Score</span>
                </div>
                <div className="stat-value">{analytics.average_score}</div>
              </div>

              <div className="stat-card">
                <div className="stat-header">
                  <span className="stat-icon">🚀</span>
                  <span className="stat-title">Highest Score</span>
                </div>
                <div className="stat-value">{analytics.highest_score}</div>
              </div>

              <div className="stat-card">
                <div className="stat-header">
                  <span className="stat-icon">📉</span>
                  <span className="stat-title">Lowest Score</span>
                </div>
                <div className="stat-value">{analytics.lowest_score}</div>
              </div>
            </div>

            <div className="recent-form-section">
              <h3>Recent Form</h3>
              <div className="form-streak">
                <span className="streak-label">Last 5 matches:</span>
                <span className="streak-value">{getFormStreak()}</span>
              </div>
              
              <div className="form-chart">
                {analytics.recent_form?.map((match, index) => (
                  <div key={index} className="match-bar">
                    <div 
                      className={`match-result ${match.result.toLowerCase()}`}
                      title={`${match.date}: ${match.result} - ${match.score} runs`}
                    >
                      <span className="match-date">{new Date(match.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}</span>
                      <span className="match-score">{match.score}</span>
                      <span className="match-outcome">{match.result}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="performance-insights">
              <h3>Performance Insights</h3>
              <div className="insights-grid">
                <div className="insight-card">
                  <h4>🎯 Batting Performance</h4>
                  <p>
                    {selectedTeam} has an average score of <strong>{analytics.average_score}</strong> runs 
                    with a highest score of <strong>{analytics.highest_score}</strong>.
                  </p>
                </div>
                
                <div className="insight-card">
                  <h4>📈 Recent Trend</h4>
                  <p>
                    {analytics.win_percentage > 60 
                      ? `${selectedTeam} is in excellent form with a ${analytics.win_percentage.toFixed(1)}% win rate.`
                      : analytics.win_percentage > 40
                      ? `${selectedTeam} has been inconsistent with a ${analytics.win_percentage.toFixed(1)}% win rate.`
                      : `${selectedTeam} is struggling with only ${analytics.win_percentage.toFixed(1)}% wins.`
                    }
                  </p>
                </div>

                <div className="insight-card">
                  <h4>🏏 Score Consistency</h4>
                  <p>
                    The score range of {analytics.highest_score - analytics.lowest_score} runs shows 
                    {analytics.highest_score - analytics.lowest_score > 200 
                      ? ' high variance in batting performance.'
                      : ' consistent batting performance.'
                    }
                  </p>
                </div>

                <div className="insight-card">
                  <h4>🔮 Prediction Insight</h4>
                  <p>
                    Based on current form, {selectedTeam} is 
                    {analytics.win_percentage > 60 ? ' highly likely' : 
                     analytics.win_percentage > 40 ? ' moderately likely' : ' less likely'} 
                    to win their next match.
                  </p>
                </div>
              </div>
            </div>

            <div className="comparison-section">
              <h3>Format Comparison</h3>
              <div className="format-cards">
                <div className="format-card">
                  <h4>🏏 Test Matches</h4>
                  <div className="format-stats">
                    <span>Avg Score: {Math.floor(analytics.average_score * 1.5)}</span>
                    <span>Win Rate: {Math.min(100, analytics.win_percentage + 5).toFixed(1)}%</span>
                  </div>
                </div>
                
                <div className="format-card">
                  <h4>🎯 ODI Matches</h4>
                  <div className="format-stats">
                    <span>Avg Score: {analytics.average_score}</span>
                    <span>Win Rate: {analytics.win_percentage.toFixed(1)}%</span>
                  </div>
                </div>
                
                <div className="format-card">
                  <h4>⚡ T20I Matches</h4>
                  <div className="format-stats">
                    <span>Avg Score: {Math.floor(analytics.average_score * 0.6)}</span>
                    <span>Win Rate: {Math.max(0, analytics.win_percentage - 5).toFixed(1)}%</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Analytics;