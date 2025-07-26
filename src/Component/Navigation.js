import React from 'react';
import './Navigation.css';

const Navigation = ({ activeTab, setActiveTab }) => {
  const tabs = [
    { id: 'match', label: '🏆 Match Predictor', description: 'Predict match winners' },
    { id: 'player', label: '👤 Player Predictor', description: 'Predict player performance' },
    { id: 'analytics', label: '📊 Analytics', description: 'Team & player insights' }
  ];

  return (
    <nav className="navigation">
      <div className="nav-container">
        {tabs.map(tab => (
          <button
            key={tab.id}
            className={`nav-tab ${activeTab === tab.id ? 'active' : ''}`}
            onClick={() => setActiveTab(tab.id)}
          >
            <span className="tab-label">{tab.label}</span>
            <span className="tab-description">{tab.description}</span>
          </button>
        ))}
      </div>
    </nav>
  );
};

export default Navigation;