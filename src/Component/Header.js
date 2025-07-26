import React from 'react';
import './Header.css';

const Header = () => {
  return (
    <header className="header">
      <div className="header-content">
        <div className="logo">
          <span className="logo-icon">🏏</span>
          <h1>Cricket Predictor</h1>
        </div>
        <div className="tagline">
          <p>AI-Powered Match & Player Performance Predictions</p>
        </div>
      </div>
    </header>
  );
};

export default Header;