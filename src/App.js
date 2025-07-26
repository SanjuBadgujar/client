import React, { useState, useEffect } from 'react';
import './App.css';
import MatchPredictor from './Component/MatchPredictor';
import PlayerPredictor from './Component/PlayerPredictor';
import Analytics from './Component/Analytics';
import Header from './Component/Header';
import Navigation from './Component/Navigation';

function App() {
  const [activeTab, setActiveTab] = useState('match');
  const [teams, setTeams] = useState([]);
  const [venues, setVenues] = useState([]);
  const [players, setPlayers] = useState([]);

  useEffect(() => {
    // Fetch initial data from backend
    fetchInitialData();
  }, []);

  const fetchInitialData = async () => {
    try {
      // Fetch teams
      const teamsResponse = await fetch('http://localhost:5000/api/teams');
      if (teamsResponse.ok) {
        const teamsData = await teamsResponse.json();
        setTeams(teamsData.teams);
      }

      // Fetch venues
      const venuesResponse = await fetch('http://localhost:5000/api/venues');
      if (venuesResponse.ok) {
        const venuesData = await venuesResponse.json();
        setVenues(venuesData.venues);
      }

      // Fetch players
      const playersResponse = await fetch('http://localhost:5000/api/players');
      if (playersResponse.ok) {
        const playersData = await playersResponse.json();
        setPlayers(playersData.players);
      }
    } catch (error) {
      console.error('Error fetching initial data:', error);
      // Use fallback data if backend is not available
      setTeams(['India', 'Australia', 'England', 'Pakistan', 'South Africa', 'New Zealand', 'West Indies', 'Sri Lanka', 'Bangladesh', 'Afghanistan']);
      setVenues(['Wankhede Stadium', 'Sydney Cricket Ground', "Lord's Cricket Ground", 'Gaddafi Stadium', 'Newlands', 'Eden Park', 'Kensington Oval', 'R. Premadasa Stadium', 'Shere Bangla National Stadium', 'Sharjah Cricket Stadium']);
      setPlayers(['Virat Kohli', 'Rohit Sharma', 'Jasprit Bumrah', 'Steve Smith', 'Pat Cummins', 'Joe Root', 'Ben Stokes', 'Babar Azam', 'Shaheen Afridi', 'Quinton de Kock', 'Kane Williamson', 'Trent Boult', 'Jason Holder', 'Dimuth Karunaratne', 'Shakib Al Hasan', 'Rashid Khan']);
    }
  };

  const renderActiveComponent = () => {
    switch (activeTab) {
      case 'match':
        return <MatchPredictor teams={teams} venues={venues} />;
      case 'player':
        return <PlayerPredictor teams={teams} venues={venues} players={players} />;
      case 'analytics':
        return <Analytics teams={teams} />;
      default:
        return <MatchPredictor teams={teams} venues={venues} />;
    }
  };

  return (
    <div className="App">
      <Header />
      <Navigation activeTab={activeTab} setActiveTab={setActiveTab} />
      <main className="main-content">
        {renderActiveComponent()}
      </main>
    </div>
  );
}

export default App;
