import pandas as pd
import numpy as np
import sqlite3
import os
from datetime import datetime, timedelta
import json

class DataProcessor:
    def __init__(self, db_path='backend/data/cricket_data.db'):
        self.db_path = db_path
        self.ensure_data_directory()
        
    def ensure_data_directory(self):
        """Ensure the data directory exists"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
    
    def initialize_database(self):
        """Initialize SQLite database with cricket data tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create matches table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                team1 TEXT NOT NULL,
                team2 TEXT NOT NULL,
                venue TEXT NOT NULL,
                format TEXT NOT NULL,
                date TEXT,
                result TEXT,
                winner TEXT,
                margin TEXT,
                toss_winner TEXT,
                toss_decision TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create players table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS players (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                team TEXT NOT NULL,
                position TEXT NOT NULL,
                nationality TEXT,
                date_of_birth TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create player stats table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS player_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_name TEXT NOT NULL,
                match_id INTEGER,
                team TEXT,
                opposition TEXT,
                venue TEXT,
                format TEXT,
                date TEXT,
                runs INTEGER DEFAULT 0,
                balls_faced INTEGER DEFAULT 0,
                fours INTEGER DEFAULT 0,
                sixes INTEGER DEFAULT 0,
                wickets INTEGER DEFAULT 0,
                overs_bowled REAL DEFAULT 0,
                runs_conceded INTEGER DEFAULT 0,
                catches INTEGER DEFAULT 0,
                stumpings INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (match_id) REFERENCES matches (id)
            )
        ''')
        
        # Create teams table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS teams (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                country TEXT,
                current_ranking_test INTEGER,
                current_ranking_odi INTEGER,
                current_ranking_t20 INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create venues table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS venues (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                city TEXT,
                country TEXT,
                capacity INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        
        # Insert sample data if tables are empty
        cursor.execute("SELECT COUNT(*) FROM teams")
        if cursor.fetchone()[0] == 0:
            self.insert_sample_data(conn)
        
        conn.close()
        print("✅ Database initialized successfully!")
    
    def insert_sample_data(self, conn):
        """Insert sample cricket data"""
        cursor = conn.cursor()
        
        # Sample teams
        teams = [
            ('India', 'India', 1, 1, 1),
            ('Australia', 'Australia', 2, 2, 2),
            ('England', 'England', 3, 3, 3),
            ('Pakistan', 'Pakistan', 4, 4, 4),
            ('South Africa', 'South Africa', 5, 5, 5),
            ('New Zealand', 'New Zealand', 6, 6, 6),
            ('West Indies', 'West Indies', 7, 7, 7),
            ('Sri Lanka', 'Sri Lanka', 8, 8, 8),
            ('Bangladesh', 'Bangladesh', 9, 9, 9),
            ('Afghanistan', 'Afghanistan', 10, 10, 10)
        ]
        
        cursor.executemany('''
            INSERT INTO teams (name, country, current_ranking_test, current_ranking_odi, current_ranking_t20)
            VALUES (?, ?, ?, ?, ?)
        ''', teams)
        
        # Sample venues
        venues = [
            ('Wankhede Stadium', 'Mumbai', 'India', 33000),
            ('Sydney Cricket Ground', 'Sydney', 'Australia', 48000),
            ("Lord's Cricket Ground", 'London', 'England', 30000),
            ('Gaddafi Stadium', 'Lahore', 'Pakistan', 27000),
            ('Newlands', 'Cape Town', 'South Africa', 25000),
            ('Eden Park', 'Auckland', 'New Zealand', 50000),
            ('Kensington Oval', 'Bridgetown', 'Barbados', 28000),
            ('R. Premadasa Stadium', 'Colombo', 'Sri Lanka', 35000),
            ('Shere Bangla National Stadium', 'Dhaka', 'Bangladesh', 25000),
            ('Sharjah Cricket Stadium', 'Sharjah', 'UAE', 27000)
        ]
        
        cursor.executemany('''
            INSERT INTO venues (name, city, country, capacity)
            VALUES (?, ?, ?, ?)
        ''', venues)
        
        # Sample players
        players = [
            ('Virat Kohli', 'India', 'Batsman', 'India', '1988-11-05'),
            ('Rohit Sharma', 'India', 'Batsman', 'India', '1987-04-30'),
            ('Jasprit Bumrah', 'India', 'Bowler', 'India', '1993-12-06'),
            ('Steve Smith', 'Australia', 'Batsman', 'Australia', '1989-06-02'),
            ('Pat Cummins', 'Australia', 'Bowler', 'Australia', '1993-05-08'),
            ('Joe Root', 'England', 'Batsman', 'England', '1990-12-30'),
            ('Ben Stokes', 'England', 'All-rounder', 'England', '1991-06-04'),
            ('Babar Azam', 'Pakistan', 'Batsman', 'Pakistan', '1994-10-15'),
            ('Shaheen Afridi', 'Pakistan', 'Bowler', 'Pakistan', '2000-04-06'),
            ('Quinton de Kock', 'South Africa', 'Wicket-keeper', 'South Africa', '1992-12-17'),
            ('Kane Williamson', 'New Zealand', 'Batsman', 'New Zealand', '1990-08-08'),
            ('Trent Boult', 'New Zealand', 'Bowler', 'New Zealand', '1989-07-22'),
            ('Jason Holder', 'West Indies', 'All-rounder', 'Barbados', '1991-11-05'),
            ('Dimuth Karunaratne', 'Sri Lanka', 'Batsman', 'Sri Lanka', '1988-04-21'),
            ('Shakib Al Hasan', 'Bangladesh', 'All-rounder', 'Bangladesh', '1987-03-24'),
            ('Rashid Khan', 'Afghanistan', 'Bowler', 'Afghanistan', '1998-09-20')
        ]
        
        cursor.executemany('''
            INSERT INTO players (name, team, position, nationality, date_of_birth)
            VALUES (?, ?, ?, ?, ?)
        ''', players)
        
        conn.commit()
        print("✅ Sample data inserted successfully!")
    
    def process_match_data(self, input_data):
        """Process match data for prediction"""
        # Get additional data for the teams
        team1_stats = self.get_team_recent_form(input_data['team1'])
        team2_stats = self.get_team_recent_form(input_data['team2'])
        
        # Get head-to-head stats
        h2h = self.get_head_to_head_stats(input_data['team1'], input_data['team2'])
        
        # Get team rankings
        team1_ranking = self.get_team_ranking(input_data['team1'], input_data['format'])
        team2_ranking = self.get_team_ranking(input_data['team2'], input_data['format'])
        
        # Process the data
        processed_data = {
            'team1': input_data['team1'],
            'team2': input_data['team2'],
            'venue': input_data['venue'],
            'format': input_data['format'],
            'team1_ranking': team1_ranking,
            'team2_ranking': team2_ranking,
            'team1_recent_wins': team1_stats.get('recent_wins', 2),
            'team2_recent_wins': team2_stats.get('recent_wins', 2),
            'h2h_total_matches': h2h.get('total_matches', 10),
            'h2h_team1_wins': h2h.get('team1_wins', 5),
            'toss_winner': input_data.get('toss_winner', input_data['team1']),
            'toss_decision': input_data.get('toss_decision', 'bat'),
            'team1_home': 1 if self.is_home_venue(input_data['team1'], input_data['venue']) else 0,
            'team1_venue_matches': self.get_venue_experience(input_data['team1'], input_data['venue']),
            'team2_venue_matches': self.get_venue_experience(input_data['team2'], input_data['venue'])
        }
        
        return processed_data
    
    def process_player_data(self, input_data):
        """Process player data for prediction"""
        # Get player career stats
        career_stats = self.get_player_career_stats(input_data['player_name'])
        
        # Get recent form
        recent_form = self.get_player_recent_form(input_data['player_name'])
        
        # Get opposition ranking
        opposition_ranking = self.get_team_ranking(input_data['opposition'], input_data['format'])
        
        # Get venue experience
        venue_matches = self.get_player_venue_experience(input_data['player_name'], input_data['venue'])
        
        # Get format specialization
        format_stats = self.get_player_format_stats(input_data['player_name'], input_data['format'])
        
        processed_data = {
            'player_name': input_data['player_name'],
            'team': input_data['team'],
            'opposition': input_data['opposition'],
            'venue': input_data['venue'],
            'format': input_data['format'],
            'position': career_stats.get('position', 'Batsman'),
            'career_matches': career_stats.get('total_matches', 50),
            'career_runs': career_stats.get('total_runs', 2000),
            'career_innings': career_stats.get('total_innings', 60),
            'career_wickets': career_stats.get('total_wickets', 10),
            'recent_innings': recent_form.get('recent_innings', 3),
            'recent_runs': recent_form.get('recent_runs', 60),
            'recent_wickets': recent_form.get('recent_wickets', 2),
            'opposition_ranking': opposition_ranking,
            'matches_at_venue': venue_matches,
            'format_matches': format_stats.get('format_matches', 20),
            'total_matches': career_stats.get('total_matches', 50)
        }
        
        return processed_data
    
    def get_teams(self):
        """Get list of all teams"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM teams ORDER BY name")
        teams = [row[0] for row in cursor.fetchall()]
        conn.close()
        return teams
    
    def get_venues(self):
        """Get list of all venues"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM venues ORDER BY name")
        venues = [row[0] for row in cursor.fetchall()]
        conn.close()
        return venues
    
    def get_players(self, team=None):
        """Get list of players, optionally filtered by team"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if team:
            cursor.execute("SELECT name FROM players WHERE team = ? ORDER BY name", (team,))
        else:
            cursor.execute("SELECT name FROM players ORDER BY name")
        
        players = [row[0] for row in cursor.fetchall()]
        conn.close()
        return players
    
    def get_team_recent_form(self, team):
        """Get recent form for a team (last 5 matches)"""
        # For demo purposes, return simulated data
        # In real implementation, this would query actual match results
        return {
            'recent_wins': np.random.randint(0, 6),
            'recent_losses': np.random.randint(0, 6),
            'recent_avg_score': np.random.randint(150, 300)
        }
    
    def get_head_to_head_stats(self, team1, team2):
        """Get head-to-head statistics between two teams"""
        # For demo purposes, return simulated data
        total_matches = np.random.randint(10, 30)
        team1_wins = np.random.randint(0, total_matches + 1)
        
        return {
            'total_matches': total_matches,
            'team1_wins': team1_wins,
            'team2_wins': total_matches - team1_wins,
            'team1_win_percentage': (team1_wins / total_matches * 100) if total_matches > 0 else 0
        }
    
    def get_team_ranking(self, team, format_type):
        """Get team ranking for specific format"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        format_column = {
            'Test': 'current_ranking_test',
            'ODI': 'current_ranking_odi',
            'T20I': 'current_ranking_t20'
        }.get(format_type, 'current_ranking_odi')
        
        cursor.execute(f"SELECT {format_column} FROM teams WHERE name = ?", (team,))
        result = cursor.fetchone()
        conn.close()
        
        return result[0] if result else 5  # Default ranking
    
    def is_home_venue(self, team, venue):
        """Check if the venue is a home venue for the team"""
        # Simple mapping for demo - in real implementation, this would be more sophisticated
        home_venues = {
            'India': ['Wankhede Stadium', 'Eden Gardens', 'M. Chinnaswamy Stadium'],
            'Australia': ['Sydney Cricket Ground', 'Melbourne Cricket Ground'],
            'England': ["Lord's Cricket Ground", 'The Oval'],
            'Pakistan': ['Gaddafi Stadium', 'National Stadium'],
            'South Africa': ['Newlands', 'The Wanderers'],
            'New Zealand': ['Eden Park', 'Basin Reserve'],
            'West Indies': ['Kensington Oval', 'Queen\'s Park Oval'],
            'Sri Lanka': ['R. Premadasa Stadium', 'Galle International Stadium'],
            'Bangladesh': ['Shere Bangla National Stadium'],
            'Afghanistan': ['Sharjah Cricket Stadium']
        }
        
        return venue in home_venues.get(team, [])
    
    def get_venue_experience(self, team, venue):
        """Get number of matches played by team at venue"""
        # For demo purposes, return simulated data
        return np.random.randint(0, 15)
    
    def get_player_career_stats(self, player_name):
        """Get player career statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get basic player info
        cursor.execute("SELECT position FROM players WHERE name = ?", (player_name,))
        result = cursor.fetchone()
        
        if not result:
            # Return default stats for unknown players
            return {
                'position': 'Batsman',
                'total_matches': 50,
                'total_runs': 2000,
                'total_innings': 60,
                'total_wickets': 5
            }
        
        position = result[0]
        
        # Get aggregated stats from player_stats table
        cursor.execute('''
            SELECT 
                COUNT(*) as total_matches,
                SUM(runs) as total_runs,
                COUNT(CASE WHEN runs >= 0 THEN 1 END) as total_innings,
                SUM(wickets) as total_wickets
            FROM player_stats 
            WHERE player_name = ?
        ''', (player_name,))
        
        stats = cursor.fetchone()
        conn.close()
        
        return {
            'position': position,
            'total_matches': stats[0] if stats[0] else 50,
            'total_runs': stats[1] if stats[1] else 2000,
            'total_innings': stats[2] if stats[2] else 60,
            'total_wickets': stats[3] if stats[3] else 5
        }
    
    def get_player_recent_form(self, player_name):
        """Get player recent form (last 5 matches)"""
        # For demo purposes, return simulated data
        recent_innings = np.random.randint(1, 6)
        recent_runs = np.random.randint(0, 200)
        recent_wickets = np.random.randint(0, 8)
        
        return {
            'recent_innings': recent_innings,
            'recent_runs': recent_runs,
            'recent_wickets': recent_wickets,
            'recent_average': recent_runs / max(recent_innings, 1)
        }
    
    def get_player_venue_experience(self, player_name, venue):
        """Get player experience at specific venue"""
        # For demo purposes, return simulated data
        return np.random.randint(0, 12)
    
    def get_player_format_stats(self, player_name, format_type):
        """Get player statistics for specific format"""
        # For demo purposes, return simulated data
        return {
            'format_matches': np.random.randint(10, 80),
            'format_runs': np.random.randint(500, 5000),
            'format_wickets': np.random.randint(0, 100)
        }
    
    def get_player_stats(self, player_name):
        """Get comprehensive player statistics"""
        career_stats = self.get_player_career_stats(player_name)
        recent_form = self.get_player_recent_form(player_name)
        
        return {
            'career_stats': career_stats,
            'recent_form': recent_form,
            'batting_average': career_stats['total_runs'] / max(career_stats['total_innings'], 1),
            'bowling_average': career_stats['total_runs'] / max(career_stats['total_wickets'], 1) if career_stats['total_wickets'] > 0 else 0
        }
    
    def get_team_analytics(self, team, format_type='all'):
        """Get team analytics data"""
        # For demo purposes, return simulated analytics data
        dates = pd.date_range(end=datetime.now(), periods=20, freq='D')
        results = np.random.choice(['Win', 'Loss'], size=20, p=[0.6, 0.4])
        
        recent_form = pd.DataFrame({
            'date': dates,
            'result': results,
            'score': np.random.randint(150, 350, 20)
        })
        
        return {
            'recent_form': recent_form.to_dict('records'),
            'win_percentage': (results == 'Win').mean() * 100,
            'average_score': recent_form['score'].mean(),
            'highest_score': recent_form['score'].max(),
            'lowest_score': recent_form['score'].min()
        }
    
    def add_match_result(self, match_data):
        """Add a match result to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO matches (team1, team2, venue, format, date, result, winner, margin, toss_winner, toss_decision)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            match_data['team1'], match_data['team2'], match_data['venue'],
            match_data['format'], match_data.get('date'), match_data.get('result'),
            match_data.get('winner'), match_data.get('margin'),
            match_data.get('toss_winner'), match_data.get('toss_decision')
        ))
        
        conn.commit()
        conn.close()
    
    def add_player_performance(self, performance_data):
        """Add player performance data to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO player_stats 
            (player_name, team, opposition, venue, format, date, runs, balls_faced, 
             fours, sixes, wickets, overs_bowled, runs_conceded, catches, stumpings)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            performance_data['player_name'], performance_data['team'],
            performance_data['opposition'], performance_data['venue'],
            performance_data['format'], performance_data.get('date'),
            performance_data.get('runs', 0), performance_data.get('balls_faced', 0),
            performance_data.get('fours', 0), performance_data.get('sixes', 0),
            performance_data.get('wickets', 0), performance_data.get('overs_bowled', 0),
            performance_data.get('runs_conceded', 0), performance_data.get('catches', 0),
            performance_data.get('stumpings', 0)
        ))
        
        conn.commit()
        conn.close()
    
    def export_data_for_training(self, format_type=None):
        """Export data in format suitable for model training"""
        conn = sqlite3.connect(self.db_path)
        
        # Export match data
        match_query = "SELECT * FROM matches"
        if format_type:
            match_query += f" WHERE format = '{format_type}'"
        
        matches_df = pd.read_sql_query(match_query, conn)
        
        # Export player stats
        player_query = "SELECT * FROM player_stats"
        if format_type:
            player_query += f" WHERE format = '{format_type}'"
        
        players_df = pd.read_sql_query(player_query, conn)
        
        conn.close()
        
        return {
            'matches': matches_df,
            'player_stats': players_df
        }